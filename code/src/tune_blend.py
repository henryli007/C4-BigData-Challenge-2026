from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from advanced.features import DATE_COL, ID_COL, LABEL_COL
from advanced.metrics import evaluate_scored_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ranker/portfolio blend on out-of-fold scores.")
    parser.add_argument("--ranker-oof", default="./model/advanced_ranker_conservative/oof_scores.csv")
    parser.add_argument("--portfolio-oof", default="./model/portfolio_direct/oof_scores.csv")
    parser.add_argument("--output", default="./model/blend_tuning.csv")
    parser.add_argument("--alphas", default="0,0.2,0.4,0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--max-weights", default="0.35,0.4,0.45,0.5,0")
    parser.add_argument("--max-per-industry", default="0,1,2")
    return parser.parse_args()


def _read_scores(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing score file: {p}")
    df = pd.read_csv(p, dtype={ID_COL: str})
    df[ID_COL] = df[ID_COL].astype(str).str.zfill(6)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    ranker = _read_scores(args.ranker_oof)
    portfolio = _read_scores(args.portfolio_oof)

    keep = [DATE_COL, ID_COL, "score", LABEL_COL, "risk_vol_20"]
    if "行业" in ranker.columns:
        keep.append("行业")
    left = ranker[keep].rename(columns={"score": "ranker_score"})
    right = portfolio[[DATE_COL, ID_COL, "score"]].rename(columns={"score": "portfolio_score"})
    merged = left.merge(right, on=[DATE_COL, ID_COL], how="inner")
    if merged.empty:
        raise ValueError("No overlapping date/stock rows between ranker and portfolio OOF scores.")

    by_date = merged.groupby(DATE_COL, sort=False)
    merged["ranker_pct"] = by_date["ranker_score"].rank(pct=True, method="first")
    merged["portfolio_pct"] = by_date["portfolio_score"].rank(pct=True, method="first")

    rows = []
    for alpha in _parse_floats(args.alphas):
        for max_weight in _parse_floats(args.max_weights):
            cap = None if max_weight <= 0 else max_weight
            for max_per_industry in _parse_ints(args.max_per_industry):
                scored = merged.copy()
                scored["score"] = alpha * scored["ranker_pct"] + (1.0 - alpha) * scored["portfolio_pct"]
                metrics = evaluate_scored_frame(
                    scored,
                    top_k=5,
                    policy="risk_adjusted",
                    temperature=0.02,
                    max_weight=cap,
                    max_per_industry=max_per_industry,
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "max_weight": max_weight,
                        "max_per_industry": max_per_industry,
                        **metrics,
                    }
                )

    result = pd.DataFrame(rows).sort_values(["final_score", "portfolio_return"], ascending=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.head(20).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
