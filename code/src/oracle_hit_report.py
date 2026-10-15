from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from advanced.features import (
    DATE_COL,
    FeatureConfig,
    ID_COL,
    LABEL_COL,
    add_forward_labels,
    filter_universe,
    load_price_data,
    load_universe_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report oracle weekly opportunity and model TopK hit-rate for the THU-BDC stock task."
    )
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="./output/oracle_hit_report")
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=[
            "./model/weekly_walkforward_tushare_long_basic_20260420_20260724/walkforward_scores.csv",
            "./model/weekly_walkforward_tushare_basic_burst_20260420_20260724/walkforward_scores.csv",
        ],
        help="Optional walkforward score CSVs. If two are provided, default blend is 0.8 first + 0.2 second.",
    )
    parser.add_argument("--blend-weights", default="", help="Comma-separated weights matching --score-files.")
    return parser.parse_args()


def weekly_cutoffs(dates: pd.Series) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    periods = unique.dt.to_period("W-FRI")
    return [pd.Timestamp(x) for x in unique.groupby(periods).max().tolist()]


def summarize_oracle(labeled: pd.DataFrame, cutoffs: list[pd.Timestamp], top_k: int) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        day = labeled[labeled[DATE_COL] == cutoff].dropna(subset=[LABEL_COL]).copy()
        if len(day) < top_k:
            continue
        ranked = day.sort_values(LABEL_COL, ascending=False)
        top5 = ranked.head(5)
        top10 = ranked.head(10)
        rows.append(
            {
                "cutoff": cutoff.date().isoformat(),
                "stocks": len(day),
                "market_equal_return": day[LABEL_COL].mean(),
                "oracle_top5_equal_return": top5[LABEL_COL].mean(),
                "oracle_top10_equal_return": top10[LABEL_COL].mean(),
                "oracle_top5_min_return": top5[LABEL_COL].min(),
                "oracle_top10_min_return": top10[LABEL_COL].min(),
                "count_return_ge_10pct": int((day[LABEL_COL] >= 0.10).sum()),
                "count_return_ge_6pct": int((day[LABEL_COL] >= 0.06).sum()),
                "true_top5": ",".join(top5[ID_COL].astype(str).str.zfill(6).tolist()),
                "true_top10": ",".join(top10[ID_COL].astype(str).str.zfill(6).tolist()),
            }
        )
    return pd.DataFrame(rows)


def _load_score(path: str, idx: int) -> pd.DataFrame:
    score = pd.read_csv(path, dtype={ID_COL: str}, encoding="utf-8-sig")
    score.columns = [str(c).replace("\ufeff", "").strip() for c in score.columns]
    score[ID_COL] = score[ID_COL].astype(str).str.zfill(6)
    score[DATE_COL] = pd.to_datetime(score[DATE_COL])
    if "score_model_only" in score.columns:
        value_col = "score_model_only"
    elif "score" in score.columns:
        value_col = "score"
    else:
        raise ValueError(f"{path} has no score or score_model_only column")
    return score[[DATE_COL, ID_COL, value_col]].rename(columns={value_col: f"score_{idx}"})


def load_blended_scores(paths: list[str], weights_arg: str) -> pd.DataFrame:
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return pd.DataFrame()
    if weights_arg:
        weights = np.array([float(x) for x in weights_arg.split(",")], dtype=float)
        if len(weights) != len(paths):
            raise ValueError("--blend-weights length must match --score-files")
    elif len(paths) == 2:
        weights = np.array([0.8, 0.2], dtype=float)
    else:
        weights = np.ones(len(paths), dtype=float)
    weights = weights / weights.sum()

    merged = None
    for i, path in enumerate(paths):
        part = _load_score(path, i)
        merged = part if merged is None else merged.merge(part, on=[DATE_COL, ID_COL], how="inner")
    assert merged is not None
    cols = [f"score_{i}" for i in range(len(paths))]
    for col in cols:
        merged[f"{col}_pct"] = merged.groupby(DATE_COL)[col].rank(pct=True, method="first")
    merged["score"] = sum(weights[i] * merged[f"score_{i}_pct"] for i in range(len(paths)))
    return merged[[DATE_COL, ID_COL, "score"]]


def summarize_model_hits(labeled: pd.DataFrame, scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    joined = scores.merge(labeled[[DATE_COL, ID_COL, LABEL_COL]], on=[DATE_COL, ID_COL], how="inner")
    rows = []
    for cutoff, day in joined.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, "score"])
        if len(day) < top_k:
            continue
        pred = day.sort_values("score", ascending=False).head(top_k)
        truth = day.sort_values(LABEL_COL, ascending=False)
        true5 = set(truth.head(5)[ID_COL])
        true10 = set(truth.head(10)[ID_COL])
        true20 = set(truth.head(20)[ID_COL])
        pred_ids = pred[ID_COL].tolist()
        row = {
            "cutoff": pd.Timestamp(cutoff).date().isoformat(),
            "pred5_equal_return": pred[LABEL_COL].mean(),
            "hit_true_top5": sum(x in true5 for x in pred_ids),
            "hit_true_top10": sum(x in true10 for x in pred_ids),
            "hit_true_top20": sum(x in true20 for x in pred_ids),
            "pred_top5": ",".join(pred_ids),
            "true_top5": ",".join(truth.head(5)[ID_COL].astype(str).str.zfill(6).tolist()),
            "true_top10": ",".join(truth.head(10)[ID_COL].astype(str).str.zfill(6).tolist()),
        }
        for n in (10, 20, 30, 50):
            predn = set(day.sort_values("score", ascending=False).head(n)[ID_COL])
            row[f"hit_true_top5_in_pred{n}"] = sum(x in predn for x in true5)
            row[f"hit_true_top10_in_pred{n}"] = sum(x in predn for x in true10)
        rows.append(
            row
        )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def print_summary(name: str, frame: pd.DataFrame, ret_col: str) -> None:
    if frame.empty:
        print(f"{name}: empty")
        return
    print(f"\n{name}")
    print(f"weeks: {len(frame)}")
    print(f"avg {ret_col}: {pct(frame[ret_col].mean())}")
    print(f"median {ret_col}: {pct(frame[ret_col].median())}")
    print(f"min {ret_col}: {pct(frame[ret_col].min())}")
    print(f"weeks {ret_col} >= 10%: {int((frame[ret_col] >= 0.10).sum())}/{len(frame)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_price_data(args.data)
    raw = filter_universe(raw, load_universe_codes(args.target_universe))
    labeled = add_forward_labels(raw, FeatureConfig(horizon=5))
    labeled[DATE_COL] = pd.to_datetime(labeled[DATE_COL])
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date) if args.end_date else labeled[DATE_COL].max()
    cutoffs = [d for d in weekly_cutoffs(labeled[DATE_COL]) if start <= d <= end]

    oracle = summarize_oracle(labeled, cutoffs, args.top_k)
    oracle.to_csv(out_dir / "oracle_weekly.csv", index=False, encoding="utf-8-sig")
    print_summary("Oracle top5 equal", oracle, "oracle_top5_equal_return")
    print_summary("Oracle top10 equal", oracle, "oracle_top10_equal_return")
    if not oracle.empty:
        print(f"avg count >=10%: {oracle['count_return_ge_10pct'].mean():.2f}")
        print(f"median count >=10%: {oracle['count_return_ge_10pct'].median():.2f}")
        print(f"min count >=10%: {oracle['count_return_ge_10pct'].min()}")
        print(f"avg market equal return: {pct(oracle['market_equal_return'].mean())}")

    scores = load_blended_scores(args.score_files, args.blend_weights)
    hits = summarize_model_hits(labeled, scores, args.top_k)
    if not hits.empty:
        hits.to_csv(out_dir / "model_hits.csv", index=False, encoding="utf-8-sig")
        print_summary("Model pred5 equal", hits, "pred5_equal_return")
        print(f"avg hit true top5: {hits['hit_true_top5'].mean():.3f}")
        print(f"avg hit true top10: {hits['hit_true_top10'].mean():.3f}")
        print(f"avg hit true top20: {hits['hit_true_top20'].mean():.3f}")
        for n in (10, 20, 30, 50):
            print(f"avg true top5 covered by pred{n}: {hits[f'hit_true_top5_in_pred{n}'].mean():.3f}/5")
            print(f"avg true top10 covered by pred{n}: {hits[f'hit_true_top10_in_pred{n}'].mean():.3f}/10")
        print("\nWorst model weeks")
        cols = ["cutoff", "pred5_equal_return", "hit_true_top5", "hit_true_top10", "pred_top5", "true_top5"]
        print(hits.sort_values("pred5_equal_return").head(5)[cols].to_string(index=False))

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
