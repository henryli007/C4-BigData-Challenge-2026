from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "日期"
ID_COL = "股票代码"
LABEL_COL = "label_return"

WEIGHTS = {
    "equal": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
    "balanced": np.array([0.25, 0.22, 0.20, 0.18, 0.15]),
    "tilt35": np.array([0.35, 0.22, 0.18, 0.15, 0.10]),
    "tilt40": np.array([0.40, 0.20, 0.15, 0.15, 0.10]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search transparent Top20 -> Top5 selection rules.")
    parser.add_argument("--validation-scores", required=True)
    parser.add_argument("--current-scores", default="")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output", default="./output/result_selection_layer.csv")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def choose_score_col(df: pd.DataFrame, requested: str = "") -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Missing score column: {requested}")
        return requested
    for col in ["final_score_tuned", "score", "score_before_failure", "score_before_continuation", "score_model_raw"]:
        if col in df.columns:
            return col
    raise ValueError("No usable score column found.")


def as_rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    return s.rank(pct=True, method="first", ascending=ascending).fillna(0.5)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def prepare_candidates(raw: pd.DataFrame, score_col: str, top_n: int, with_label: bool) -> pd.DataFrame:
    rows = []
    for date, day in raw.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[score_col]).copy()
        if with_label:
            day = day.dropna(subset=[LABEL_COL]).copy()
            day["future_rank_full"] = day[LABEL_COL].rank(method="first", ascending=False)
        top = day.sort_values(score_col, ascending=False).head(top_n).copy()
        top["candidate_rank"] = np.arange(1, len(top) + 1)
        top["base_z"] = zscore(top[score_col])
        top["runup_risk"] = 0.0
        for col, weight in [("ret_5", 0.45), ("ret_10", 0.35), ("ret_20", 0.20)]:
            if col in top.columns:
                top["runup_risk"] += weight * as_rank_pct(top[col], ascending=True)
        if "upper_shadow_3" in top.columns:
            top["upper_risk"] = as_rank_pct(top["upper_shadow_3"], ascending=True)
        else:
            top["upper_risk"] = 0.5
        decay_cols = [c for c in ["total_decay_risk", "strength_decay_score", "runup_reversal_score"] if c in top.columns]
        top["decay_risk_norm"] = 0.0
        for col in decay_cols:
            top["decay_risk_norm"] += as_rank_pct(top[col], ascending=True) / max(1, len(decay_cols))
        if "failure_score" in top.columns:
            top["failure_norm"] = as_rank_pct(top["failure_score"], ascending=True)
        else:
            top["failure_norm"] = 0.5
        if "continuation_score" in top.columns:
            top["continuation_norm"] = as_rank_pct(top["continuation_score"], ascending=True)
        else:
            top["continuation_norm"] = 0.5
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def select_top5(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return day.sort_values(score_col, ascending=False).head(5)


def select_bucket(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ordered = day.sort_values(score_col, ascending=False).copy()
    parts = [
        ordered.iloc[0:5].head(2),
        ordered.iloc[5:10].head(2),
        ordered.iloc[10:20].head(1),
    ]
    return pd.concat(parts, ignore_index=False).head(5)


def select_rule(day: pd.DataFrame, score_col: str, params: dict[str, float]) -> pd.DataFrame:
    score = day["base_z"].copy()
    score -= params["runup"] * day["runup_risk"]
    score -= params["upper"] * day["upper_risk"]
    score -= params["decay"] * day["decay_risk_norm"]
    score -= params["failure"] * day["failure_norm"]
    score += params["continuation"] * day["continuation_norm"]
    if "market_regime_score" in day.columns:
        score -= params["market"] * as_rank_pct(day["market_regime_score"], ascending=True)
    tmp = day.copy()
    tmp["_selection_score"] = score
    return tmp.sort_values("_selection_score", ascending=False).head(5)


def select_alternating(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ordered = day.sort_values(score_col, ascending=False).copy()
    selected = []
    remaining = ordered.copy()
    modes = ["runup_risk", "upper_risk", "runup_risk", "upper_risk", "runup_risk"]
    for mode in modes:
        if remaining.empty:
            break
        pool = remaining.head(8).copy()
        score = zscore(pool[score_col]) - 0.85 * as_rank_pct(pool[mode], ascending=True)
        pick_idx = score.sort_values(ascending=False).index[0]
        selected.append(remaining.loc[pick_idx])
        remaining = remaining.drop(index=pick_idx)
    return pd.DataFrame(selected).head(5)


def select_mmr(day: pd.DataFrame, score_col: str, lam: float) -> pd.DataFrame:
    ordered = day.sort_values(score_col, ascending=False).copy()
    ret_cols = [c for c in ["ret_1", "ret_2", "ret_5", "ret_10", "ret_20"] if c in ordered.columns]
    selected_idx: list[int] = []
    remaining = list(ordered.index)
    for _ in range(5):
        best_idx = None
        best_score = -1e18
        for idx in remaining:
            row = ordered.loc[idx]
            penalty = 0.0
            for chosen in selected_idx:
                chosen_row = ordered.loc[chosen]
                if "行业" in ordered.columns and row.get("行业") == chosen_row.get("行业"):
                    penalty += 0.45
                if ret_cols:
                    a = row[ret_cols].astype(float).to_numpy()
                    b = chosen_row[ret_cols].astype(float).to_numpy()
                    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
                    penalty += max(0.0, float(np.dot(a, b) / denom))
            score = float(row["base_z"]) - lam * penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)
    return ordered.loc[selected_idx].head(5)


def evaluate(prepared: pd.DataFrame, strategy: str, selector) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    for date, day in prepared.groupby(DATE_COL, sort=True):
        selected = selector(day).copy()
        selected = selected.head(5)
        for weight_name, weights in WEIGHTS.items():
            weights = weights[: len(selected)]
            weights = weights / weights.sum()
            returns = selected[LABEL_COL].to_numpy(dtype=float)
            rows.append(
                {
                    "strategy": strategy,
                    "weight": weight_name,
                    "date": date,
                    "return_equal": float(returns.mean()),
                    "return_weighted": float(np.dot(weights, returns)),
                    "hit_true_top5": int((selected["future_rank_full"] <= 5).sum()),
                    "hit_true_top10": int((selected["future_rank_full"] <= 10).sum()),
                    "rank1_hit10": int(selected.iloc[0]["future_rank_full"] <= 10),
                    "selected": ",".join(selected[ID_COL].astype(str).str.zfill(6).tolist()),
                }
            )
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            details.append(
                {
                    "strategy": strategy,
                    "date": date,
                    "selected_rank": rank,
                    "stock_id": str(row[ID_COL]).zfill(6),
                    "future_return": float(row[LABEL_COL]),
                    "future_rank_full": float(row["future_rank_full"]),
                    "candidate_rank": int(row["candidate_rank"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(details)


def summarize(per_week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, weight), g in per_week.groupby(["strategy", "weight"], sort=False):
        wt = g["return_weighted"].to_numpy(dtype=float)
        rows.append(
            {
                "strategy": strategy,
                "weight": weight,
                "weeks": int(len(g)),
                "return_equal": float(g["return_equal"].mean()),
                "return_weighted": float(wt.mean()),
                "hit_true_top5": float(g["hit_true_top5"].mean()),
                "hit_true_top10": float(g["hit_true_top10"].mean()),
                "rank1_hit10": int(g["rank1_hit10"].sum()),
                "hit10_weeks": int((wt >= 0.10).sum()),
                "win_rate": float((wt > 0).mean()),
                "worst_week": float(wt.min()),
            }
        )
    out = pd.DataFrame(rows)
    out["robust_score"] = (
        1.15 * out["hit_true_top10"]
        + 0.75 * out["hit_true_top5"]
        + 0.07 * out["rank1_hit10"]
        + 2.2 * out["return_weighted"]
        + 0.30 * out["win_rate"]
        + 0.08 * out["hit10_weeks"]
        - 0.20 * out["worst_week"].clip(upper=0).abs()
    )
    return out.sort_values(["robust_score", "return_weighted", "hit_true_top10"], ascending=False)


def build_strategies(score_col: str) -> dict[str, object]:
    strategies = {
        "A_top5": lambda day: select_top5(day, score_col),
        "B_bucket_2_2_1": lambda day: select_bucket(day, score_col),
        "C_mmr_0p10": lambda day: select_mmr(day, score_col, 0.10),
        "C_mmr_0p20": lambda day: select_mmr(day, score_col, 0.20),
        "D_alt_low_runup_upper": lambda day: select_alternating(day, score_col),
    }
    for runup in [0.15, 0.30, 0.45, 0.60]:
        for upper in [0.00, 0.15, 0.30]:
            for decay in [0.00, 0.15, 0.30]:
                for failure in [0.00, 0.15, 0.30]:
                    for cont in [0.00, 0.10, 0.20]:
                        params = {
                            "runup": runup,
                            "upper": upper,
                            "decay": decay,
                            "failure": failure,
                            "continuation": cont,
                            "market": 0.05,
                        }
                        name = (
                            f"R_run{runup:.2f}_up{upper:.2f}_dec{decay:.2f}"
                            f"_fail{failure:.2f}_cont{cont:.2f}"
                        )
                        strategies[name] = lambda day, params=params: select_rule(day, score_col, params)
    return strategies


def write_current_selection(
    current_scores: str,
    output: str,
    score_col: str,
    top_n: int,
    strategy_name: str,
    weight_name: str,
) -> list[dict[str, object]]:
    current = pd.read_csv(current_scores, dtype={ID_COL: str})
    current[DATE_COL] = pd.to_datetime(current[DATE_COL])
    current_score = choose_score_col(current, score_col if score_col in current.columns else "")
    prepared = prepare_candidates(current[current[DATE_COL] == current[DATE_COL].max()].copy(), current_score, top_n, with_label=False)
    selector = build_strategies(current_score)[strategy_name]
    selected = selector(prepared).head(5).copy()
    weights = WEIGHTS[weight_name][: len(selected)]
    weights = weights / weights.sum()
    out = selected.rename(columns={ID_COL: "stock_id"})[["stock_id"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["weight"] = weights
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, float_format="%.6f")
    return out.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.validation_scores, dtype={ID_COL: str})
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])
    score_col = choose_score_col(raw, args.score_col)
    prepared = prepare_candidates(raw, score_col, args.top_n, with_label=True)
    strategies = build_strategies(score_col)

    per_week_frames = []
    detail_frames = []
    for name, selector in strategies.items():
        per_week, details = evaluate(prepared, name, selector)
        per_week_frames.append(per_week)
        detail_frames.append(details)
    per_week_all = pd.concat(per_week_frames, ignore_index=True)
    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary = summarize(per_week_all)

    per_week_all.to_csv(report_dir / "selection_per_week.csv", index=False)
    detail_all.to_csv(report_dir / "selection_details.csv", index=False)
    summary.to_csv(report_dir / "selection_summary.csv", index=False)
    best = summary.iloc[0].to_dict()
    selected = []
    if args.current_scores:
        selected = write_current_selection(
            args.current_scores,
            args.output,
            score_col,
            args.top_n,
            str(best["strategy"]),
            str(best["weight"]),
        )
    payload = {
        "score_col": score_col,
        "top_n": args.top_n,
        "best": best,
        "selected": selected,
    }
    (report_dir / "selection_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
