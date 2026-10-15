from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .features import DATE_COL, FeatureConfig, ID_COL, LABEL_COL, build_feature_frame, filter_trainable_rows
from .metrics import evaluate_scored_frame


@dataclass(frozen=True)
class PortfolioConfig:
    output_dir: str = "./model/portfolio_direct"
    top_k: int = 5
    hidden_dim: int = 256
    dropout: float = 0.15
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    risk_lambda: float = 0.15
    concentration_gamma: float = 0.05
    entropy_beta: float = 0.01
    early_stopping_patience: int = 8
    val_size: int = 40
    n_val_folds: int = 3
    rolling_val_epochs: int = 20
    random_state: int = 42
    progress_path: str = ""


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DailyPortfolioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], mean: np.ndarray, std: np.ndarray):
        self.days = []
        for date, day in df.sort_values([DATE_COL, ID_COL]).groupby(DATE_COL, sort=True):
            x = ((day[feature_cols].to_numpy(np.float32) - mean) / std).astype(np.float32)
            y = day[LABEL_COL].to_numpy(np.float32)
            risk = day.get("risk_vol_20", pd.Series(0.03, index=day.index)).to_numpy(np.float32)
            codes = day[ID_COL].astype(str).tolist()
            if len(day) >= 10:
                self.days.append((date, x, y, risk, codes))

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, idx: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return self.days[idx]


def collate_daily(batch):
    max_n = max(item[1].shape[0] for item in batch)
    feat_dim = batch[0][1].shape[1]
    xs, ys, risks, masks, dates, codes = [], [], [], [], [], []
    for date, x, y, risk, code_list in batch:
        n = x.shape[0]
        x_pad = np.zeros((max_n, feat_dim), dtype=np.float32)
        y_pad = np.zeros(max_n, dtype=np.float32)
        risk_pad = np.zeros(max_n, dtype=np.float32)
        mask = np.zeros(max_n, dtype=np.float32)
        x_pad[:n] = x
        y_pad[:n] = y
        risk_pad[:n] = risk
        mask[:n] = 1.0
        xs.append(x_pad)
        ys.append(y_pad)
        risks.append(risk_pad)
        masks.append(mask)
        dates.append(date)
        codes.append(code_list)
    return (
        dates,
        torch.tensor(np.stack(xs)),
        torch.tensor(np.stack(ys)),
        torch.tensor(np.stack(risks)),
        torch.tensor(np.stack(masks)),
        codes,
    )


class PortfolioScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def portfolio_loss(scores: torch.Tensor, returns: torch.Tensor, risk: torch.Tensor, mask: torch.Tensor, cfg: PortfolioConfig) -> torch.Tensor:
    masked_scores = scores.masked_fill(mask <= 0, -1e9)
    weights = F.softmax(masked_scores, dim=1) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
    portfolio_return = (weights * returns).sum(dim=1)
    risk_penalty = (weights * risk.clamp_min(1e-4)).sum(dim=1)
    concentration = (weights.pow(2)).sum(dim=1)
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1)
    return (-portfolio_return + cfg.risk_lambda * risk_penalty + cfg.concentration_gamma * concentration - cfg.entropy_beta * entropy).mean()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _score_dataset(model: PortfolioScorer, dataset: DailyPortfolioDataset, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_daily)
    rows = []
    model.eval()
    with torch.no_grad():
        for dates, x, y, risk, mask, codes in loader:
            x = x.to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()
            valid = mask.squeeze(0).numpy() > 0
            for code, score, ret, rv in zip(codes[0], scores[valid], y.squeeze(0).numpy()[valid], risk.squeeze(0).numpy()[valid]):
                rows.append({DATE_COL: dates[0], ID_COL: code, "score": float(score), LABEL_COL: float(ret), "risk_vol_20": float(rv)})
    return pd.DataFrame(rows)


def make_rolling_folds(dates: list[pd.Timestamp], val_size: int, n_folds: int, embargo: int = 5) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    dates = list(pd.DatetimeIndex(pd.to_datetime(dates)).sort_values())
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    if n_folds <= 0 or len(dates) <= val_size + embargo + 20:
        return folds
    first_val_start = max(20 + embargo, len(dates) - n_folds * val_size)
    for val_start in range(first_val_start, len(dates), val_size):
        val_end = min(val_start + val_size, len(dates))
        train_end = max(0, val_start - embargo)
        train_dates = dates[:train_end]
        val_dates = dates[val_start:val_end]
        if len(train_dates) >= 20 and len(val_dates) > 0:
            folds.append((train_dates, val_dates))
    return folds[-n_folds:]


def _fit_portfolio_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PortfolioConfig,
    epochs: int,
    device: torch.device,
    seed_offset: int = 0,
    log_prefix: str = "portfolio",
) -> tuple[PortfolioScorer, dict[str, Any], pd.DataFrame]:
    torch.manual_seed(cfg.random_state + seed_offset)
    np.random.seed(cfg.random_state + seed_offset)

    mean = train_df[feature_cols].mean().to_numpy(np.float32)
    std = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0).to_numpy(np.float32)
    train_set = DailyPortfolioDataset(train_df, feature_cols, mean, std)
    val_set = DailyPortfolioDataset(val_df, feature_cols, mean, std)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_daily)

    model = PortfolioScorer(len(feature_cols), cfg.hidden_dim, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _, x, y, risk, mask, _ in train_loader:
            x, y, risk, mask = x.to(device), y.to(device), risk.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(x)
            loss = portfolio_loss(scores, y, risk, mask, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scored = _score_dataset(model, val_set, device)
        metrics = evaluate_scored_frame(
            val_scored,
            top_k=cfg.top_k,
            policy="score_softmax",
            temperature=0.2,
            max_weight=0.45,
            max_per_industry=2,
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses) if losses else 0.0), **metrics})
        print(f"{log_prefix} epoch {epoch}: loss={history[-1]['loss']:.6f}, val_return={metrics['portfolio_return']:.6f}")
        if metrics["portfolio_return"] > best_metric:
            best_metric = metrics["portfolio_return"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_scored = _score_dataset(model, val_set, device)
    result = {
        "mean": mean,
        "std": std,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_return": best_metric,
        "train_days": len(train_set),
        "val_days": len(val_set),
    }
    return model, result, best_val_scored


def train_portfolio_model(
    raw_df: pd.DataFrame,
    feature_cfg: FeatureConfig | None = None,
    portfolio_cfg: PortfolioConfig | None = None,
) -> dict[str, Any]:
    feature_cfg = feature_cfg or FeatureConfig()
    portfolio_cfg = portfolio_cfg or PortfolioConfig()
    output_dir = Path(portfolio_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(portfolio_cfg.progress_path) if portfolio_cfg.progress_path else output_dir / "progress.json"

    torch.manual_seed(portfolio_cfg.random_state)
    np.random.seed(portfolio_cfg.random_state)

    print("building direct-portfolio feature frame")
    _write_progress(progress_path, {"stage": "feature_build", "status": "running"})
    featured, feature_cols = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=True)
    trainable = filter_trainable_rows(featured, feature_cols, feature_cfg)
    dates = list(pd.DatetimeIndex(trainable[DATE_COL].unique()).sort_values())
    if len(dates) <= portfolio_cfg.val_size + 20:
        split = max(int(len(dates) * 0.8), 1)
    else:
        split = len(dates) - portfolio_cfg.val_size
    train_dates = dates[:split]
    val_dates = dates[split:]
    train_df = trainable[trainable[DATE_COL].isin(train_dates)].copy()
    val_df = trainable[trainable[DATE_COL].isin(val_dates)].copy()

    _write_progress(
        progress_path,
        {
            "stage": "training",
            "status": "running",
            "epochs": portfolio_cfg.epochs,
            "completed_epochs": 0,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "feature_count": len(feature_cols),
        },
    )

    device = _device()
    rolling_summaries = []
    rolling_oof_parts = []
    folds = make_rolling_folds(dates, portfolio_cfg.val_size, portfolio_cfg.n_val_folds)
    for fold_idx, (fold_train_dates, fold_val_dates) in enumerate(folds, start=1):
        fold_train = trainable[trainable[DATE_COL].isin(fold_train_dates)].copy()
        fold_val = trainable[trainable[DATE_COL].isin(fold_val_dates)].copy()
        print(
            f"portfolio fold {fold_idx}: train {min(fold_train_dates).date()}..{max(fold_train_dates).date()} "
            f"val {min(fold_val_dates).date()}..{max(fold_val_dates).date()}"
        )
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": fold_idx,
                "fold_count": len(folds),
                "completed_folds": fold_idx - 1,
                "train_rows": len(fold_train),
                "val_rows": len(fold_val),
            },
        )
        _, fold_result, fold_scored = _fit_portfolio_split(
            fold_train,
            fold_val,
            feature_cols,
            portfolio_cfg,
            epochs=min(portfolio_cfg.rolling_val_epochs, portfolio_cfg.epochs),
            device=device,
            seed_offset=fold_idx,
            log_prefix=f"portfolio fold {fold_idx}",
        )
        fold_metric = evaluate_scored_frame(
            fold_scored,
            top_k=portfolio_cfg.top_k,
            policy="score_softmax",
            temperature=0.2,
            max_weight=0.45,
            max_per_industry=2,
        )
        rolling_summaries.append(
            {
                "fold": fold_idx,
                "best_epoch": fold_result["best_epoch"],
                "best_val_return": fold_result["best_val_return"],
                **fold_metric,
            }
        )
        rolling_oof_parts.append(fold_scored.assign(fold=fold_idx))
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": fold_idx,
                "fold_count": len(folds),
                "completed_folds": fold_idx,
                "latest_fold_metric": rolling_summaries[-1],
            },
        )

    if rolling_summaries:
        pd.DataFrame(rolling_summaries).to_csv(output_dir / "rolling_validation.csv", index=False)
        pd.concat(rolling_oof_parts, ignore_index=True).to_csv(output_dir / "oof_scores.csv", index=False)

    print("portfolio final train")
    model, final_result, val_scored = _fit_portfolio_split(
        train_df,
        val_df,
        feature_cols,
        portfolio_cfg,
        epochs=portfolio_cfg.epochs,
        device=device,
        log_prefix="portfolio final",
    )
    history = final_result["history"]
    best_epoch = final_result["best_epoch"]
    best_metric = final_result["best_val_return"]
    mean = final_result["mean"]
    std = final_result["std"]
    val_scored.to_csv(output_dir / "validation_scores.csv", index=False)
    torch.save(model.state_dict(), output_dir / "portfolio_model.pth")
    artifact = {
        "feature_config": asdict(feature_cfg),
        "portfolio_config": asdict(portfolio_cfg),
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "history": history,
        "rolling_validation": rolling_summaries,
        "train_dates": (str(min(train_dates).date()), str(max(train_dates).date())),
        "val_dates": (str(min(val_dates).date()), str(max(val_dates).date())) if val_dates else None,
    }
    joblib.dump(artifact, output_dir / "portfolio_artifact.joblib")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    _write_progress(
        progress_path,
        {
            "stage": "finished",
            "status": "done",
            "epochs": portfolio_cfg.epochs,
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_return": best_metric,
            "artifact": str(output_dir / "portfolio_artifact.joblib"),
            "model": str(output_dir / "portfolio_model.pth"),
        },
    )
    return artifact
