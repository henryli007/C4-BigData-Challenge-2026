from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from advanced.features import DATE_COL, ID_COL, LABEL_COL
from advanced.metrics import select_portfolio


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy: str
    temperature: float
    max_per_industry: int


POLICIES = [
    PolicySpec("concentrated", "risk_adjusted", 0.005, 1),
    PolicySpec("balanced", "risk_adjusted", 0.02, 2),
    PolicySpec("diversified", "risk_adjusted", 0.05, 2),
    PolicySpec("equal", "equal", 0.02, 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune dynamic portfolio weight policy on OOF scores.")
    parser.add_argument("--oof", default="./model/ranker_hybrid_20260626/oof_scores.csv")
    parser.add_argument("--current-scored", default="./output/current_scored_top30_20260626.csv")
    parser.add_argument("--output", default="./model/ranker_hybrid_20260626/dynamic_policy_tuning.csv")
    parser.add_argument("--half-life-days", type=float, default=28.0)
    parser.add_argument("--min-confidence", default="0,0.005,0.01,0.015,0.02,0.03")
    parser.add_argument("--fallback", choices=[p.name for p in POLICIES], default="balanced")
    return parser.parse_args()


def _read_scores(path: str | Path, require_label: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={ID_COL: str})
    df[ID_COL] = df[ID_COL].astype(str).str.zfill(6)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if require_label and LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain {LABEL_COL}")
    if "risk_vol_20" not in df.columns:
        df["risk_vol_20"] = 0.03
    return df


def _parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _date_confidence(day: pd.DataFrame) -> dict[str, float]:
    ranked = day.sort_values("score", ascending=False)
    scores = ranked["score"].to_numpy(dtype=float)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else scores[0]
    top5 = scores[min(4, len(scores) - 1)]
    risk = float(ranked.iloc[0].get("risk_vol_20", 0.03))
    return {
        "top1_score": float(top1),
        "gap_1_2": float(top1 - top2),
        "gap_1_5": float(top1 - top5),
        "top1_risk_vol_20": risk,
    }


def _score_day(day: pd.DataFrame, spec: PolicySpec) -> tuple[float, dict[str, float]]:
    pred = select_portfolio(
        day,
        top_k=5,
        policy=spec.policy,
        temperature=spec.temperature,
        max_per_industry=spec.max_per_industry,
    )
    merged = pred.merge(day[[ID_COL, LABEL_COL]], on=ID_COL, how="left")
    score = float((merged["weight"] * merged[LABEL_COL]).sum())
    return score, {
        "max_weight": float(pred["weight"].max()),
        "selected": ",".join(pred[ID_COL].astype(str).str.zfill(6).tolist()),
    }


def _recency_weight(dates: pd.Series, half_life_days: float) -> np.ndarray:
    max_date = dates.max()
    age = (max_date - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age / max(half_life_days, 1e-6))


def main() -> None:
    args = parse_args()
    oof = _read_scores(args.oof, require_label=True)
    thresholds = _parse_thresholds(args.min_confidence)
    specs = {p.name: p for p in POLICIES}
    fallback = specs[args.fallback]

    per_date_rows = []
    for date, day in oof.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, "score"])
        if len(day) < 5:
            continue
        conf = _date_confidence(day)
        for spec in POLICIES:
            score, extra = _score_day(day, spec)
            per_date_rows.append({"date": date, "candidate": spec.name, "return": score, **conf, **extra})

    per_date = pd.DataFrame(per_date_rows)
    weights = _recency_weight(per_date["date"], args.half_life_days)
    per_date["recency_weight"] = weights

    rows = []
    for threshold in thresholds:
        chosen = []
        for date, group in per_date.groupby("date", sort=True):
            conf = float(group["gap_1_2"].iloc[0])
            pick = "concentrated" if conf >= threshold else fallback.name
            row = group[group["candidate"] == pick].iloc[0].to_dict()
            row["rule"] = f"gap_1_2>={threshold:g}?concentrated:{fallback.name}"
            chosen.append(row)
        chosen_df = pd.DataFrame(chosen)
        weighted_return = float(np.average(chosen_df["return"], weights=chosen_df["recency_weight"]))
        plain_return = float(chosen_df["return"].mean())
        rows.append(
            {
                "threshold": threshold,
                "fallback": fallback.name,
                "plain_return": plain_return,
                "recency_weighted_return": weighted_return,
                "avg_max_weight": float(chosen_df["max_weight"].mean()),
                "recent_5_return": float(chosen_df.tail(5)["return"].mean()),
                "recent_10_return": float(chosen_df.tail(10)["return"].mean()),
            }
        )

    # Include static policy baselines.
    for spec in POLICIES:
        group = per_date[per_date["candidate"] == spec.name].copy()
        rows.append(
            {
                "threshold": np.nan,
                "fallback": f"static_{spec.name}",
                "plain_return": float(group["return"].mean()),
                "recency_weighted_return": float(np.average(group["return"], weights=group["recency_weight"])),
                "avg_max_weight": float(group["max_weight"].mean()),
                "recent_5_return": float(group.tail(5)["return"].mean()),
                "recent_10_return": float(group.tail(10)["return"].mean()),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["recency_weighted_return", "recent_5_return", "plain_return"],
        ascending=False,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.head(20).to_string(index=False))
    print(f"wrote {out}")

    current_path = Path(args.current_scored)
    if current_path.exists():
        current = _read_scores(current_path, require_label=False)
        conf = _date_confidence(current)
        best = result.iloc[0]
        threshold = best["threshold"]
        if pd.isna(threshold):
            selected_name = str(best["fallback"]).replace("static_", "")
        else:
            selected_name = "concentrated" if conf["gap_1_2"] >= float(threshold) else fallback.name
        spec = specs[selected_name]
        pred = select_portfolio(
            current,
            top_k=5,
            policy=spec.policy,
            temperature=spec.temperature,
            max_per_industry=spec.max_per_industry,
        )
        pred = pred.rename(columns={ID_COL: "stock_id"})[["stock_id", "weight"]]
        pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)
        pred["weight"] = pred["weight"] / pred["weight"].sum()
        current_out = out.with_name(out.stem + "_current_result.csv")
        pred.to_csv(current_out, index=False)
        print("current confidence:", conf)
        print(f"selected policy: {selected_name}")
        print(pred.to_string(index=False))
        print(f"wrote {current_out}")


if __name__ == "__main__":
    main()
