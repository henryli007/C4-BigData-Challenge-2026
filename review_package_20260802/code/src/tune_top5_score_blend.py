from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["cutoff", "日期", "股票代码"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune score blends against Top5 equal-weight weekly return.")
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Named score CSV in the form name=path. CSV must contain cutoff/日期/股票代码/label_return/score.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cover-k", type=int, default=10)
    parser.add_argument("--max-per-industry", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=200)
    return parser.parse_args()


def _parse_score_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--score must be name=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty score name in {value!r}")
    return name, Path(path)


def _load_score(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"股票代码": str}, parse_dates=["日期", "cutoff"])
    missing = [col for col in [*KEYS, "label_return", "score"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    keep = [*KEYS, "label_return", "score"]
    if "行业" in df.columns:
        keep.append("行业")
    out = df[keep].copy()
    out["股票代码"] = out["股票代码"].astype(str).str.zfill(6)
    out[f"score_{name}"] = out.groupby("cutoff")["score"].rank(pct=True, method="first")
    out = out.drop(columns=["score"])
    return out


def _weight_grid(n: int, step: float, min_weight: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--step must divide 1.0 exactly, for example 0.1, 0.05, 0.025")
    min_units = int(np.ceil(min_weight / step - 1e-12))
    weights: list[tuple[float, ...]] = []
    for vals in product(range(units + 1), repeat=n - 1):
        last = units - sum(vals)
        if last < 0:
            continue
        combo = (*vals, last)
        if min(combo) < min_units:
            continue
        weights.append(tuple(round(x * step, 10) for x in combo))
    return weights


def _select_top(group: pd.DataFrame, score_col: str, top_k: int, max_per_industry: int) -> pd.DataFrame:
    ranked = group.sort_values(score_col, ascending=False).copy()
    if max_per_industry <= 0 or "行业" not in ranked.columns:
        return ranked.head(top_k).copy()
    counts: dict[str, int] = {}
    keep = []
    for idx, row in ranked.iterrows():
        industry = str(row.get("行业", "UNK"))
        if counts.get(industry, 0) >= max_per_industry:
            continue
        keep.append(idx)
        counts[industry] = counts.get(industry, 0) + 1
        if len(keep) >= top_k:
            break
    selected = ranked.loc[keep].copy()
    if len(selected) < top_k:
        selected = pd.concat([selected, ranked.drop(index=selected.index).head(top_k - len(selected))], axis=0)
    return selected


def _evaluate(df: pd.DataFrame, score_cols: list[str], weights: tuple[float, ...], args: argparse.Namespace) -> dict[str, float]:
    score_col = "_blend_score"
    df[score_col] = 0.0
    for col, weight in zip(score_cols, weights, strict=True):
        df[score_col] += weight * df[col]

    returns = []
    hit_true_top5 = []
    hit_true_top10 = []
    cover_true_top10_pred10 = []
    mins = []
    for _, group in df.groupby("cutoff", sort=True):
        selected = _select_top(group, score_col, args.top_k, args.max_per_industry)
        returns.append(float(selected["label_return"].mean()))
        mins.append(float(selected["label_return"].min()))
        true_rank = group["label_return"].rank(method="first", ascending=False)
        true_top5 = set(group.loc[true_rank <= 5, "股票代码"])
        true_top10 = set(group.loc[true_rank <= 10, "股票代码"])
        pred_top5 = set(selected["股票代码"])
        pred_top10 = set(group.nlargest(args.cover_k, score_col)["股票代码"])
        hit_true_top5.append(len(pred_top5 & true_top5))
        hit_true_top10.append(len(pred_top5 & true_top10))
        cover_true_top10_pred10.append(len(pred_top10 & true_top10))

    ret = np.array(returns, dtype=float)
    out = {f"w_{col.removeprefix('score_')}": weight for col, weight in zip(score_cols, weights, strict=True)}
    out.update(
        {
            "top5_equal_return": float(ret.mean()),
            "median_return": float(np.median(ret)),
            "hit10_weeks": int((ret >= 0.10).sum()),
            "min_week_return": float(ret.min()),
            "avg_selected_min_return": float(np.mean(mins)),
            "avg_hit_true_top5": float(np.mean(hit_true_top5)),
            "avg_hit_true_top10": float(np.mean(hit_true_top10)),
            "avg_cover_true_top10_in_pred10": float(np.mean(cover_true_top10_pred10)),
            "weeks": int(len(ret)),
        }
    )
    return out


def main() -> None:
    args = parse_args()
    named_paths = [_parse_score_arg(item) for item in args.score]
    frames = []
    for i, (name, path) in enumerate(named_paths):
        frame = _load_score(name, path)
        if i > 0:
            drop = ["label_return"]
            if "行业" in frame.columns:
                drop.append("行业")
            frame = frame.drop(columns=drop)
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEYS, how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows after merging score files")

    score_cols = [f"score_{name}" for name, _ in named_paths]
    rows = []
    for weights in _weight_grid(len(score_cols), args.step, args.min_weight):
        rows.append(_evaluate(merged.copy(), score_cols, weights, args))
    result = pd.DataFrame(rows).sort_values(
        ["top5_equal_return", "hit10_weeks", "avg_hit_true_top10", "min_week_return"],
        ascending=[False, False, False, False],
    )
    if args.max_rows > 0:
        result = result.head(args.max_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(result.head(20).to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
