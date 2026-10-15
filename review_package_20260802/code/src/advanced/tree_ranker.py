from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import DATE_COL, FeatureConfig, ID_COL, LABEL_COL, build_feature_frame, filter_trainable_rows
from .metrics import evaluate_scored_frame


@dataclass(frozen=True)
class TreeRankerConfig:
    output_dir: str = "./model/advanced_ranker"
    top_k: int = 5
    n_folds: int = 4
    val_size: int = 20
    embargo: int = 5
    min_train_dates: int = 120
    random_state: int = 42
    lgb_estimators: int = 700
    xgb_estimators: int = 550
    cat_estimators: int = 700
    learning_rate: float = 0.035
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 25
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 0.2
    max_rows_per_fold: int = 0
    weight_temperature: float = 0.02
    progress_path: str = ""
    relevance_mode: str = "rank"


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _date_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby(DATE_COL, sort=True).size().astype(int).tolist()


def _make_relevance(df: pd.DataFrame, top_k: int, mode: str = "rank") -> pd.Series:
    def _day(s: pd.Series) -> pd.Series:
        rank = s.rank(method="first", ascending=False)
        if mode == "burst":
            rel = np.zeros(len(s), dtype=int)
            rel += np.where(rank.to_numpy() <= 1, 10, 0)
            rel += np.where((rank.to_numpy() > 1) & (rank.to_numpy() <= 3), 7, 0)
            rel += np.where((rank.to_numpy() > 3) & (rank.to_numpy() <= top_k), 5, 0)
            rel += np.where((rank.to_numpy() > top_k) & (rank.to_numpy() <= 10), 2, 0)
            values = s.to_numpy(dtype=float)
            rel += np.where(values >= 0.10, 4, 0)
            rel += np.where((values >= 0.06) & (values < 0.10), 2, 0)
            rel += np.where((values >= 0.03) & (values < 0.06), 1, 0)
            rel -= np.where(values < 0.0, 1, 0)
            rel = np.clip(rel, 0, None)
        elif mode == "recall":
            rel = np.zeros(len(s), dtype=int)
            rank_values = rank.to_numpy()
            values = s.to_numpy(dtype=float)
            rel += np.where(rank_values <= 1, 8, 0)
            rel += np.where((rank_values > 1) & (rank_values <= 3), 6, 0)
            rel += np.where((rank_values > 3) & (rank_values <= top_k), 5, 0)
            rel += np.where((rank_values > top_k) & (rank_values <= 10), 3, 0)
            rel += np.where((rank_values > 10) & (rank_values <= 20), 1, 0)
            rel += np.where(values >= 0.10, 4, 0)
            rel += np.where((values >= 0.06) & (values < 0.10), 2, 0)
            rel += np.where((values >= 0.03) & (values < 0.06), 1, 0)
            rel -= np.where(values < -0.03, 1, 0)
            rel = np.clip(rel, 0, None)
        else:
            rel = np.maximum(0, top_k + 1 - rank.to_numpy()).astype(int)
        return pd.Series(rel, index=s.index)

    return df.groupby(DATE_COL, group_keys=False)[LABEL_COL].apply(_day).astype(int)


def make_rolling_folds(dates: list[pd.Timestamp], cfg: TreeRankerConfig) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    dates = list(pd.DatetimeIndex(pd.to_datetime(dates)).sort_values())
    n = len(dates)
    if n < cfg.min_train_dates + cfg.embargo + 5:
        split = max(int(n * 0.8), 1)
        return [(dates[:split], dates[split:])]

    first_val_start = max(cfg.min_train_dates + cfg.embargo, n - cfg.n_folds * cfg.val_size)
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    for val_start in range(first_val_start, n, cfg.val_size):
        val_end = min(val_start + cfg.val_size, n)
        train_end = max(0, val_start - cfg.embargo)
        train_dates = dates[:train_end]
        val_dates = dates[val_start:val_end]
        if len(train_dates) >= cfg.min_train_dates and len(val_dates) > 0:
            folds.append((train_dates, val_dates))
    return folds[-cfg.n_folds :]


def _maybe_downsample(df: pd.DataFrame, cfg: TreeRankerConfig) -> pd.DataFrame:
    if cfg.max_rows_per_fold <= 0 or len(df) <= cfg.max_rows_per_fold:
        return df
    rng = np.random.default_rng(cfg.random_state)
    keep = rng.choice(df.index.to_numpy(), size=cfg.max_rows_per_fold, replace=False)
    return df.loc[np.sort(keep)].copy()


def _fit_lightgbm(train_df: pd.DataFrame, feature_cols: list[str], cfg: TreeRankerConfig) -> Any:
    from lightgbm import LGBMRanker

    ordered = train_df.sort_values([DATE_COL, ID_COL]).copy()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=cfg.lgb_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
        importance_type="gain",
        verbose=-1,
    )
    model.fit(
        ordered[feature_cols],
        _make_relevance(ordered, cfg.top_k, cfg.relevance_mode),
        group=_date_groups(ordered),
    )
    return model


def _fit_xgboost(train_df: pd.DataFrame, feature_cols: list[str], cfg: TreeRankerConfig) -> Any:
    from xgboost import XGBRanker

    ordered = train_df.sort_values([DATE_COL, ID_COL]).copy()
    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        n_estimators=cfg.xgb_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=6 if cfg.max_depth <= 0 else cfg.max_depth,
        min_child_weight=max(1, cfg.min_child_samples // 3),
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=max(cfg.reg_lambda, 0.8),
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(
        ordered[feature_cols],
        _make_relevance(ordered, cfg.top_k, cfg.relevance_mode),
        group=_date_groups(ordered),
        verbose=False,
    )
    return model


def _fit_catboost(train_df: pd.DataFrame, feature_cols: list[str], cfg: TreeRankerConfig) -> Any:
    from catboost import CatBoostRanker, Pool

    ordered = train_df.sort_values([DATE_COL, ID_COL]).copy()
    train_pool = Pool(
        ordered[feature_cols],
        label=_make_relevance(ordered, cfg.top_k, cfg.relevance_mode),
        group_id=ordered[DATE_COL].astype(str),
    )
    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=5",
        iterations=cfg.cat_estimators,
        learning_rate=cfg.learning_rate,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=cfg.random_state,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    return model


MODEL_BUILDERS = {
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "catboost": _fit_catboost,
}


def fit_available_models(train_df: pd.DataFrame, feature_cols: list[str], cfg: TreeRankerConfig) -> dict[str, Any]:
    models: dict[str, Any] = {}
    train_df = _maybe_downsample(train_df, cfg)
    for name, builder in MODEL_BUILDERS.items():
        try:
            models[name] = builder(train_df, feature_cols, cfg)
            print(f"trained {name}")
        except ImportError as exc:
            print(f"skip {name}: dependency is not installed ({exc})")
        except Exception as exc:
            print(f"skip {name}: training failed ({exc})")
    if not models:
        raise RuntimeError("No tree ranker was trained. Install lightgbm, xgboost, or catboost.")
    return models


def score_with_models(df: pd.DataFrame, feature_cols: list[str], models: dict[str, Any]) -> pd.DataFrame:
    keep_cols = [DATE_COL, ID_COL]
    for col in [LABEL_COL, "risk_vol_20", "risk_vol_60", "行业"]:
        if col in df.columns:
            keep_cols.append(col)
    scored = df[keep_cols].copy()
    if "risk_vol_20" not in scored.columns:
        scored["risk_vol_20"] = 0.03
    if "risk_vol_60" not in scored.columns:
        scored["risk_vol_60"] = scored["risk_vol_20"]
    model_cols = []
    x = df[feature_cols]
    for name, model in models.items():
        col = f"score_{name}"
        scored[col] = np.asarray(model.predict(x), dtype=float)
        model_cols.append(col)

    ranked_scores = scored.groupby(DATE_COL, sort=False)[model_cols].rank(pct=True, method="first")
    scored["score"] = ranked_scores.mean(axis=1)
    return scored


def _choose_policy(oof: pd.DataFrame, cfg: TreeRankerConfig) -> tuple[str, dict[str, dict[str, float]]]:
    policies = ["risk_adjusted", "score_softmax", "equal"]
    metrics = {
        policy: evaluate_scored_frame(
            oof,
            top_k=cfg.top_k,
            policy=policy,
            temperature=cfg.weight_temperature,
        )
        for policy in policies
    }
    best = max(metrics, key=lambda p: metrics[p]["portfolio_return"])
    return best, metrics


def train_tree_ranker(
    raw_df: pd.DataFrame,
    feature_cfg: FeatureConfig | None = None,
    ranker_cfg: TreeRankerConfig | None = None,
) -> dict[str, Any]:
    feature_cfg = feature_cfg or FeatureConfig()
    ranker_cfg = ranker_cfg or TreeRankerConfig()
    output_dir = Path(ranker_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(ranker_cfg.progress_path) if ranker_cfg.progress_path else output_dir / "progress.json"

    print("building multi-window factors")
    _write_progress(progress_path, {"stage": "feature_build", "status": "running"})
    featured, feature_cols = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=True)
    trainable = filter_trainable_rows(featured, feature_cols, feature_cfg)
    dates = list(pd.DatetimeIndex(trainable[DATE_COL].unique()).sort_values())
    folds = make_rolling_folds(dates, ranker_cfg)
    if not folds:
        raise RuntimeError("Unable to create rolling validation folds.")

    print(f"trainable rows: {len(trainable)}, features: {len(feature_cols)}, folds: {len(folds)}")
    _write_progress(
        progress_path,
        {
            "stage": "rolling_validation",
            "status": "running",
            "trainable_rows": len(trainable),
            "feature_count": len(feature_cols),
            "fold_count": len(folds),
            "completed_folds": 0,
        },
    )
    oof_parts = []
    fold_summaries = []
    for fold_idx, (train_dates, val_dates) in enumerate(folds, start=1):
        fold_train = trainable[trainable[DATE_COL].isin(train_dates)].copy()
        fold_val = trainable[trainable[DATE_COL].isin(val_dates)].copy()
        print(
            f"fold {fold_idx}: train {min(train_dates).date()}..{max(train_dates).date()} "
            f"val {min(val_dates).date()}..{max(val_dates).date()}"
        )
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": fold_idx,
                "fold_count": len(folds),
                "completed_folds": fold_idx - 1,
                "train_start": str(min(train_dates).date()),
                "train_end": str(max(train_dates).date()),
                "val_start": str(min(val_dates).date()),
                "val_end": str(max(val_dates).date()),
                "train_rows": len(fold_train),
                "val_rows": len(fold_val),
            },
        )
        models = fit_available_models(fold_train, feature_cols, ranker_cfg)
        scored = score_with_models(fold_val, feature_cols, models)
        fold_metric = evaluate_scored_frame(scored, top_k=ranker_cfg.top_k, policy="risk_adjusted")
        fold_summaries.append({"fold": fold_idx, **fold_metric})
        oof_parts.append(scored)
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": fold_idx,
                "fold_count": len(folds),
                "completed_folds": fold_idx,
                "latest_fold_metric": fold_summaries[-1],
            },
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "oof_scores.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(output_dir / "rolling_validation.csv", index=False)
    best_policy, policy_metrics = _choose_policy(oof, ranker_cfg)
    print(f"best weight policy: {best_policy}")
    for policy, metric in policy_metrics.items():
        print(f"{policy}: return={metric['portfolio_return']:.6f}, final_score={metric['final_score']:.6f}")

    _write_progress(
        progress_path,
        {
            "stage": "final_train",
            "status": "running",
            "trainable_rows": len(trainable),
            "feature_count": len(feature_cols),
            "best_policy": best_policy,
            "policy_metrics": policy_metrics,
        },
    )
    final_models = fit_available_models(trainable, feature_cols, ranker_cfg)
    artifact = {
        "feature_config": asdict(feature_cfg),
        "ranker_config": asdict(ranker_cfg),
        "feature_cols": feature_cols,
        "models": final_models,
        "best_policy": best_policy,
        "policy_metrics": policy_metrics,
        "fold_summaries": fold_summaries,
        "train_dates": (str(min(dates).date()), str(max(dates).date())),
    }
    joblib.dump(artifact, output_dir / "tree_ranker.joblib")
    pd.DataFrame(policy_metrics).T.to_csv(output_dir / "weight_policy_metrics.csv")
    _write_progress(
        progress_path,
        {
            "stage": "finished",
            "status": "done",
            "trainable_rows": len(trainable),
            "feature_count": len(feature_cols),
            "best_policy": best_policy,
            "policy_metrics": policy_metrics,
            "artifact": str(output_dir / "tree_ranker.joblib"),
        },
    )
    return artifact
