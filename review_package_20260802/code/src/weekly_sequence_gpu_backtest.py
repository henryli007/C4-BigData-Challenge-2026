from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from advanced.features import (
    DATE_COL,
    ID_COL,
    LABEL_COL,
    FeatureConfig,
    add_market_context,
    build_feature_frame,
    enrich_with_metadata,
    filter_universe,
    load_price_data,
    load_universe_codes,
)
from train_current_lgb_submission import merge_extra_factor_dir, merge_extra_hot_data
from weekly_walkforward_backtest import _add_label_known_date, _valid_feature_rows, _weekly_cutoffs


class SequenceScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.12):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU sequence model walk-forward backtest.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output-dir", default="./model/weekly_sequence_gpu_recent8")
    parser.add_argument("--start-date", default="2026-04-20")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--max-weeks", type=int, default=8)
    parser.add_argument("--extra-hot-data", default="./model/extra_apr_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_apr_event_noboard")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-train-rows", type=int, default=70000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank = out.groupby(DATE_COL)[LABEL_COL].rank(method="first", ascending=False)
    out["rel"] = np.maximum(0.0, 11.0 - rank.to_numpy(dtype=float)) / 10.0
    out["target_ge10"] = (out[LABEL_COL] >= 0.10).astype(float)
    return out


def _pick_sequence_features(feature_cols: list[str]) -> list[str]:
    preferred = [
        "f_ret_1",
        "f_gap",
        "f_intraday",
        "f_high_low_pct",
        "f_close_to_high",
        "f_close_to_low",
        "f_upper_shadow",
        "f_lower_shadow",
        "f_log_amount",
        "f_turnover",
        "f_amplitude",
        "f_close_ret_3",
        "f_close_ret_5",
        "f_close_ret_10",
        "f_close_ret_20",
        "f_ret_std_5",
        "f_ret_std_10",
        "f_ret_std_20",
        "f_amount_rel_5",
        "f_amount_rel_20",
        "f_volume_rel_5",
        "f_volume_rel_20",
        "f_channel_pos_20",
        "f_mom_5_minus_20",
        "f_vol_ratio_5_20",
        "f_hit_up_limit",
        "f_close_near_up_limit",
        "f_close_to_up_limit",
        "f_ths_hot_listed",
        "f_ths_hot_rank_score",
        "f_ths_hot_listed_count_5",
        "f_ths_hot_rank_score_mean_20",
        "f_limit_list_up",
        "f_limit_list_up_count_20",
        "f_top_listed",
        "f_top_listed_count_20",
        "f_margin_fin_balance_chg_5",
        "f_margin_fin_balance_chg_20",
        "f_net_mf_amount_rel",
        "f_large_order_net_rel",
        "f_extra_large_order_net_rel",
        "f_mkt_ret_mean",
        "f_mkt_up_ratio",
        "f_beta_mkt_20",
        "risk_vol_20",
    ]
    return [c for c in preferred if c in feature_cols]


def _sequence_index(featured: pd.DataFrame, seq_features: list[str], seq_len: int) -> dict[tuple[pd.Timestamp, str], np.ndarray]:
    seqs: dict[tuple[pd.Timestamp, str], np.ndarray] = {}
    for code, g in featured.sort_values([ID_COL, DATE_COL]).groupby(ID_COL, sort=False):
        values = g[seq_features].to_numpy(np.float32)
        dates = pd.to_datetime(g[DATE_COL]).to_numpy()
        for i in range(seq_len - 1, len(g)):
            seqs[(pd.Timestamp(dates[i]), str(code).zfill(6))] = values[i - seq_len + 1 : i + 1]
    return seqs


def _make_arrays(rows: pd.DataFrame, seqs: dict[tuple[pd.Timestamp, str], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, rel, ge10, ret = [], [], [], []
    for row in rows.itertuples(index=False):
        key = (pd.Timestamp(getattr(row, DATE_COL)), str(getattr(row, ID_COL)).zfill(6))
        seq = seqs.get(key)
        if seq is None:
            continue
        xs.append(seq)
        rel.append(float(getattr(row, "rel")))
        ge10.append(float(getattr(row, "target_ge10")))
        ret.append(float(getattr(row, LABEL_COL)))
    return np.asarray(xs, np.float32), np.asarray(rel, np.float32), np.asarray(ge10, np.float32), np.asarray(ret, np.float32)


def _train_model(x: np.ndarray, rel: np.ndarray, ge10: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[SequenceScorer, StandardScaler]:
    rng = np.random.default_rng(args.seed)
    if args.max_train_rows and len(x) > args.max_train_rows:
        idx = rng.choice(np.arange(len(x)), size=args.max_train_rows, replace=False)
        x, rel, ge10 = x[idx], rel[idx], ge10[idx]
    scaler = StandardScaler()
    flat = x.reshape(-1, x.shape[-1])
    scaler.fit(flat)
    x = scaler.transform(flat).reshape(x.shape).astype(np.float32)
    model = SequenceScorer(x.shape[-1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    y_rel = torch.from_numpy(rel)
    y_ge10 = torch.from_numpy(ge10)
    x_t = torch.from_numpy(x)
    n = len(x)
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        model.train()
        losses = []
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_t[idx].to(device)
            rb = y_rel[idx].to(device)
            gb = y_ge10[idx].to(device)
            pred = model(xb)
            loss = F.mse_loss(torch.sigmoid(pred), rb) + 0.35 * F.binary_cross_entropy_with_logits(pred, gb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}")
    return model, scaler


def _predict(model: SequenceScorer, scaler: StandardScaler, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    x = scaler.transform(flat).reshape(x.shape).astype(np.float32)
    xt = torch.from_numpy(x)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores.append(model(xt[start : start + batch_size].to(device)).detach().cpu().numpy())
    return np.concatenate(scores)


def _eval(scored: pd.DataFrame) -> pd.DataFrame:
    templates = {
        "equal": np.ones(5) / 5,
        "rank1": np.array([0.999996, 0.000001, 0.000001, 0.000001, 0.000001]),
        "tpl80": np.array([0.8, 0.1, 0.05, 0.03, 0.02]),
    }
    rows = []
    for date, day in scored.groupby(DATE_COL, sort=True):
        top = day.sort_values("score", ascending=False).head(5)
        truth = day.sort_values(LABEL_COL, ascending=False)
        top10 = set(truth.head(10)[ID_COL])
        vals = top[LABEL_COL].to_numpy(float)
        row = {
            "date": pd.Timestamp(date).date().isoformat(),
            "selected": ",".join(top[ID_COL].astype(str).str.zfill(6).tolist()),
            "hit_true_top10": sum(x in top10 for x in top[ID_COL]),
        }
        for name, w in templates.items():
            row[name] = float(np.sum(vals * (w / w.sum())))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = enrich_with_metadata(load_price_data(args.data), Path(args.data).parent)
    raw = merge_extra_hot_data(raw, args.extra_hot_data)
    raw = merge_extra_factor_dir(raw, args.extra_factor_dir)
    raw = filter_universe(add_market_context(raw), load_universe_codes(args.target_universe))
    cfg = FeatureConfig(windows=(3, 5, 10, 20, 30, 60), min_history=max(60, args.seq_len), horizon=5, top_k=5)
    featured, feature_cols = build_feature_frame(raw, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    featured = _make_targets(featured)
    seq_features = _pick_sequence_features(feature_cols)
    print(f"sequence_features={len(seq_features)}")
    seqs = _sequence_index(featured, seq_features, args.seq_len)
    trainable = featured.dropna(subset=[LABEL_COL, "label_known_date"]).copy()
    pred_all = _valid_feature_rows(featured, feature_cols, cfg).dropna(subset=[LABEL_COL]).copy()
    cutoffs = [d for d in _weekly_cutoffs(pred_all[DATE_COL]) if pd.Timestamp(args.start_date) <= d <= pd.Timestamp(args.end_date)]
    if args.max_weeks:
        cutoffs = cutoffs[-args.max_weeks :]

    parts = []
    for i, cutoff in enumerate(cutoffs, 1):
        train_rows = trainable[trainable["label_known_date"] <= cutoff].copy()
        day = pred_all[pred_all[DATE_COL] == cutoff].copy()
        print(f"[{i}/{len(cutoffs)}] cutoff={cutoff.date()} train_rows={len(train_rows)} pred_rows={len(day)}")
        x_train, rel, ge10, _ = _make_arrays(train_rows, seqs)
        x_day, _, _, _ = _make_arrays(day, seqs)
        if len(x_train) == 0 or len(x_day) == 0:
            continue
        model, scaler = _train_model(x_train, rel, ge10, args, device)
        score = _predict(model, scaler, x_day, args.batch_size, device)
        scored_day = day[[DATE_COL, ID_COL, LABEL_COL]].copy()
        scored_day["score"] = score
        parts.append(scored_day)
    scored = pd.concat(parts, ignore_index=True)
    scored.to_csv(out_dir / "walkforward_scores.csv", index=False)
    per = _eval(scored)
    per.to_csv(out_dir / "per_date.csv", index=False)
    summary = []
    for col in ["equal", "rank1", "tpl80"]:
        s = per[col]
        summary.append({"policy": col, "avg": float(s.mean()), "hit10": int((s >= 0.10).sum()), "min": float(s.min()), "avg_hit_true_top10": float(per["hit_true_top10"].mean())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(per.to_string(index=False))


if __name__ == "__main__":
    main()
