from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "日期"
ID_COL = "股票代码"
LABEL_COL = "label_return"

TEMPLATES = {
    "equal_20": np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=float),
    "front_40": np.array([0.40, 0.20, 0.15, 0.15, 0.10], dtype=float),
    "front_60": np.array([0.60, 0.15, 0.10, 0.10, 0.05], dtype=float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission-time rerank diagnostics and fixed-weight simulation.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-col", default="")
    parser.add_argument("--continuation-weight", type=float, default=0.12)
    parser.add_argument("--failure-weight", type=float, default=0.16)
    parser.add_argument("--decay-weight", type=float, default=0.005)
    parser.add_argument("--soft-filter-factor", type=float, default=0.98)
    return parser.parse_args()


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


def _choose_base_col(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Missing requested base column: {requested}")
        return requested
    for col in ["score_before_continuation", "score_model_raw", "model_score", "score"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a base score column.")


def _apply_score(df: pd.DataFrame, args: argparse.Namespace, base_col: str) -> pd.DataFrame:
    work = df.copy()
    work["base_rank_for_final"] = _rank_pct(work, base_col)
    work["continuation_rank_for_final"] = _rank_pct(work, "continuation_score") if "continuation_score" in work.columns else 0.5
    work["failure_rank_for_final"] = _rank_pct(work, "failure_score") if "failure_score" in work.columns else 0.5
    work["decay_penalty_for_final"] = _risk_penalty(work)
    work["soft_failure_filter"] = _soft_failure_filter(work)
    work["final_score_tuned"] = (
        work["base_rank_for_final"]
        + float(args.continuation_weight) * work["continuation_rank_for_final"]
        - float(args.failure_weight) * work["failure_rank_for_final"]
        - float(args.decay_weight) * work["decay_penalty_for_final"]
    )
    factor = float(np.clip(args.soft_filter_factor, 0.95, 1.0))
    work.loc[work["soft_failure_filter"], "final_score_tuned"] *= factor
    return work


def _diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    positions = []
    for date, day in df.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, "final_score_tuned"]).copy()
        if len(day) < 20:
            continue
        ranked = day.sort_values("final_score_tuned", ascending=False).reset_index(drop=True)
        ranked["pred_position"] = np.arange(1, len(ranked) + 1)
        truth = day.sort_values(LABEL_COL, ascending=False).reset_index(drop=True)
        truth["future_rank"] = np.arange(1, len(truth) + 1)
        merged = ranked.merge(truth[[ID_COL, "future_rank"]], on=ID_COL, how="left")
        top5 = merged.head(5)
        true_top10_pos = merged[merged["future_rank"] <= 10]["pred_position"].tolist()
        rows.append(
            {
                "week": date,
                "pred_top1_future_rank": int(top5["future_rank"].iloc[0]),
                "pred_top1_return": float(top5[LABEL_COL].iloc[0]),
                "pred_top5_avg_future_rank": float(top5["future_rank"].mean()),
                "pred_top5_return": float(top5[LABEL_COL].mean()),
                "pred_top5_hit_true_top5": int((top5["future_rank"] <= 5).sum()),
                "pred_top5_hit_true_top10": int((top5["future_rank"] <= 10).sum()),
                "true_top10_in_pred5": int(sum(pos <= 5 for pos in true_top10_pos)),
                "true_top10_in_pred10": int(sum(pos <= 10 for pos in true_top10_pos)),
                "true_top10_in_pred20": int(sum(pos <= 20 for pos in true_top10_pos)),
                "true_top10_avg_pred_position": float(np.mean(true_top10_pos)) if true_top10_pos else np.nan,
            }
        )
        keep_cols = [
            ID_COL,
            "pred_position",
            "future_rank",
            "final_score_tuned",
            LABEL_COL,
            "base_rank_for_final",
            "continuation_score",
            "failure_score",
            "total_decay_risk",
            "soft_failure_filter",
        ]
        for col in keep_cols:
            if col not in merged.columns:
                merged[col] = np.nan
        positions.append(merged.head(20)[[DATE_COL, *keep_cols]].rename(columns={DATE_COL: "week"}))
    weekly = pd.DataFrame(rows)
    pos = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()
    return weekly, pos


def _simulate_weights(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, weights in TEMPLATES.items():
        returns = []
        rank1_returns = []
        for date, day in df.groupby(DATE_COL, sort=True):
            day = day.dropna(subset=[LABEL_COL, "final_score_tuned"]).sort_values("final_score_tuned", ascending=False).head(5)
            if len(day) < 5:
                continue
            returns.append(float(np.dot(weights, day[LABEL_COL].to_numpy(dtype=float))))
            rank1_returns.append(float(day[LABEL_COL].iloc[0]))
        if not returns:
            continue
        arr = np.array(returns, dtype=float)
        rank1 = np.array(rank1_returns, dtype=float)
        rows.append(
            {
                "strategy": name,
                "weights": ",".join(f"{x:.2f}" for x in weights),
                "weeks": int(len(arr)),
                "avg_return": float(arr.mean()),
                "hit10_weeks": int((arr >= 0.10).sum()),
                "win_rate_positive": float((arr > 0).mean()),
                "max_drawdown_week": float(arr.min()),
                "rank1_hit10": int((rank1 >= 0.10).sum()),
                "rank1_avg_return": float(rank1.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["hit10_weeks", "avg_return", "max_drawdown_week"], ascending=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scores, dtype={ID_COL: str})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    base_col = _choose_base_col(df, args.base_col)
    scored = _apply_score(df, args, base_col)
    weekly, positions = _diagnostics(scored)
    portfolio = _simulate_weights(scored)
    scored.to_csv(out_dir / "scores_with_final_tuned.csv", index=False)
    weekly.to_csv(out_dir / "candidate_rerank_diagnostics.csv", index=False)
    positions.to_csv(out_dir / "candidate_top20_positions.csv", index=False)
    portfolio.to_csv(out_dir / "weight_strategy_simulation.csv", index=False)
    summary = {
        "base_col": base_col,
        "continuation_weight": args.continuation_weight,
        "failure_weight": args.failure_weight,
        "decay_weight": args.decay_weight,
        "soft_filter_factor": args.soft_filter_factor,
        "best_weight_strategy": portfolio.iloc[0].to_dict() if not portfolio.empty else {},
        "diagnostic_summary": weekly.mean(numeric_only=True).to_dict() if not weekly.empty else {},
    }
    (out_dir / "final_rerank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
