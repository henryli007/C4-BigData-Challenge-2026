from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feature importance and optional correlation audit for current fusion artifact.")
    parser.add_argument("--artifact", default="./model/current_submission/fusion_submission.joblib")
    parser.add_argument("--output-dir", default="./model/current_submission/feature_audit")
    parser.add_argument("--top-n", type=int, default=120)
    return parser.parse_args()


def _feature_importance(model: object, feature_cols: list[str]) -> pd.DataFrame:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
    else:
        gain = getattr(model, "feature_importances_", np.zeros(len(feature_cols)))
        split = np.zeros(len(feature_cols))
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": np.asarray(gain, dtype=float),
            "importance_split": np.asarray(split, dtype=float),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False)


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.artifact)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for profile in bundle["profiles"]:
        name = str(profile["profile"])
        feature_cols = list(profile["feature_cols"])
        imp = _feature_importance(profile["model"], feature_cols)
        total_gain = imp["importance_gain"].sum()
        imp["gain_share"] = imp["importance_gain"] / total_gain if total_gain > 0 else 0.0
        imp["profile"] = name
        imp.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)
        all_rows.append(imp)
        print(f"\n{name} top {args.top_n}")
        print(imp.head(args.top_n).to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / "all_feature_importance.csv", index=False)
    summary = (
        combined.groupby("feature", as_index=False)
        .agg(
            profiles=("profile", "nunique"),
            total_gain=("importance_gain", "sum"),
            total_split=("importance_split", "sum"),
            avg_gain_share=("gain_share", "mean"),
        )
        .sort_values(["total_gain", "total_split"], ascending=False)
    )
    summary.to_csv(out_dir / "combined_feature_importance.csv", index=False)
    low = combined[combined["importance_gain"] <= 0].groupby("profile")["feature"].count().reset_index(name="zero_gain_features")
    low.to_csv(out_dir / "zero_gain_summary.csv", index=False)
    print(f"\nwrote {out_dir}")
    try:
        import shap  # noqa: F401
    except Exception as exc:
        print(f"shap unavailable; skipped SHAP export ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
