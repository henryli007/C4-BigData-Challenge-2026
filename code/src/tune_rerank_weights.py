from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "日期"
ID_COL = "股票代码"
LABEL_COL = "label_return"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune continuation/failure/decay rerank weights on walk-forward scores.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-col", default="")
    parser.add_argument("--continuation-grid", default="0.05,0.08,0.10,0.12,0.16,0.20")
    parser.add_argument("--failure-grid", default="0.10,0.12,0.16,0.20,0.24,0.30")
    parser.add_argument("--decay-grid", default="0,0.005,0.01,0.015,0.02")
    parser.add_argument("--soft-filter-factor", type=float, default=0.98)
    return parser.parse_args()


def _parse_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _rank_pct(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return df.groupby(DATE_COL)[col].rank(pct=True, method="first", ascending=ascending).fillna(0.5)


def _risk_penalty(df: pd.DataFrame) -> pd.Series:
    cols = {
        "strength_decay_score": 0.34,
        "runup_reversal_score": 0.27,
        "event_proxy_risk": 0.18,
        "theme_lifecycle_risk": 0.11,
        "market_regime_score": 0.10,
    }
    out = pd.Series(0.0, index=df.index)
    for col, weight in cols.items():
        if col in df.columns:
            out += weight * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return out


def _soft_failure_filter(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    if "up_streak" in df.columns:
        flags |= pd.to_numeric(df["up_streak"], errors="coerce").fillna(0.0) >= 5
    if "bull_candle_streak" in df.columns:
        flags |= pd.to_numeric(df["bull_candle_streak"], errors="coerce").fillna(0.0) >= 5
    if "volume_price_divergence_count_5" in df.columns:
        flags |= pd.to_numeric(df["volume_price_divergence_count_5"], errors="coerce").fillna(0.0) >= 2
    if "hot_score_chg_3" in df.columns:
        flags |= pd.to_numeric(df["hot_score_chg_3"], errors="coerce").fillna(0.0) <= -0.10
    if "ret_5" in df.columns:
        ret5_rank = df.groupby(DATE_COL)["ret_5"].rank(pct=True, method="first")
        weak_close = pd.to_numeric(df.get("close_strength_chg_3", 0.0), errors="coerce").fillna(0.0) <= -0.20
        flags |= (ret5_rank >= 0.93) & weak_close
    return flags.fillna(False)


def _evaluate(df: pd.DataFrame, score_col: str = "tuned_score") -> dict[str, float]:
    rows = []
    for date, day in df.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, score_col]).copy()
        if len(day) < 10:
            continue
        ranked = day.sort_values(score_col, ascending=False)
        truth = day.sort_values(LABEL_COL, ascending=False)
        pred5 = ranked.head(5)
        pred10_ids = set(ranked.head(10)[ID_COL])
        pred20_ids = set(ranked.head(20)[ID_COL])
        true5_ids = set(truth.head(5)[ID_COL])
        true10_ids = set(truth.head(10)[ID_COL])
        rows.append(
            {
                "date": date,
                "rank1_return": float(pred5[LABEL_COL].iloc[0]),
                "top5_equal_return": float(pred5[LABEL_COL].mean()),
                "top5_hit_true_top5": len(set(pred5[ID_COL]) & true5_ids),
                "top5_hit_true_top10": len(set(pred5[ID_COL]) & true10_ids),
                "top10_precision_true_top10": len(set(ranked.head(10)[ID_COL]) & true10_ids) / 10.0,
                "cover_true_top10_in_pred20": len(pred20_ids & true10_ids),
                "cover_true_top5_in_pred10": len(pred10_ids & true5_ids),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    top5 = frame["top5_equal_return"].to_numpy(dtype=float)
    rank1 = frame["rank1_return"].to_numpy(dtype=float)
    return {
        "weeks": int(len(frame)),
        "top5_equal_return": float(np.mean(top5)),
        "top5_equal_hit10": int((top5 >= 0.10).sum()),
        "rank1_return": float(np.mean(rank1)),
        "rank1_hit10": int((rank1 >= 0.10).sum()),
        "worst_top5_return": float(np.min(top5)),
        "win_rate_top5_positive": float((top5 > 0).mean()),
        "avg_hit_true_top5": float(frame["top5_hit_true_top5"].mean()),
        "avg_hit_true_top10": float(frame["top5_hit_true_top10"].mean()),
        "top10_precision_true_top10": float(frame["top10_precision_true_top10"].mean()),
        "avg_cover_true_top10_in_pred20": float(frame["cover_true_top10_in_pred20"].mean()),
        "avg_cover_true_top5_in_pred10": float(frame["cover_true_top5_in_pred10"].mean()),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scores, dtype={ID_COL: str})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    base_col = args.base_col
    if not base_col:
        for candidate in ["score_before_continuation", "score_model_raw", "model_score", "score"]:
            if candidate in df.columns:
                base_col = candidate
                break
    if not base_col or base_col not in df.columns:
        raise ValueError("No base score column found. Pass --base-col.")
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df["base_rank_for_tune"] = _rank_pct(df, base_col)
    df["continuation_rank_for_tune"] = _rank_pct(df, "continuation_score") if "continuation_score" in df.columns else 0.5
    df["failure_rank_for_tune"] = _rank_pct(df, "failure_score") if "failure_score" in df.columns else 0.5
    df["risk_penalty_for_tune"] = _risk_penalty(df)
    df["soft_failure_filter"] = _soft_failure_filter(df)

    rows = []
    for continuation_weight in _parse_grid(args.continuation_grid):
        for failure_weight in _parse_grid(args.failure_grid):
            for decay_weight in _parse_grid(args.decay_grid):
                work = df.copy()
                work["tuned_score"] = (
                    work["base_rank_for_tune"]
                    + continuation_weight * work["continuation_rank_for_tune"]
                    - failure_weight * work["failure_rank_for_tune"]
                    - decay_weight * work["risk_penalty_for_tune"]
                )
                factor = float(np.clip(args.soft_filter_factor, 0.95, 1.0))
                work.loc[work["soft_failure_filter"], "tuned_score"] *= factor
                metrics = _evaluate(work)
                if metrics:
                    rows.append(
                        {
                            "base_col": base_col,
                            "continuation_weight": continuation_weight,
                            "failure_weight": failure_weight,
                            "decay_weight": decay_weight,
                            **metrics,
                        }
                    )
    out = pd.DataFrame(rows)
    out["selection_score"] = (
        out["top5_equal_return"]
        + 0.012 * out["top10_precision_true_top10"]
        + 0.010 * out["avg_hit_true_top10"]
        + 0.004 * out["rank1_hit10"]
        + 0.003 * out["top5_equal_hit10"]
        - 0.20 * out["worst_top5_return"].clip(upper=0).abs()
    )
    out = out.sort_values(["selection_score", "top5_equal_return", "worst_top5_return"], ascending=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.head(20).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
