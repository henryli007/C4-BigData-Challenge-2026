from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from advanced.features import DATE_COL, ID_COL, FeatureConfig, build_feature_frame
from train_current_fusion_submission import _latest_candidates, _load_raw_cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feature importance and optional correlation audit for current fusion artifact.")
    parser.add_argument("--artifact", default="./model/current_submission/fusion_submission.joblib")
    parser.add_argument("--output-dir", default="./model/current_submission/feature_audit")
    parser.add_argument("--top-n", type=int, default=120)
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--extra-hot-data", default="./model/extra_apr_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_apr_event_data")
    parser.add_argument("--shap-sample-rows", type=int, default=120)
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
    try:
        import shap
    except Exception as exc:
        shap = None
        print(f"shap unavailable; skipped SHAP export ({type(exc).__name__}: {exc})")
    raw_cache: dict[tuple[bool, bool], pd.DataFrame] = {}
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
        if shap is not None and args.shap_sample_rows > 0:
            raw = _load_raw_cached(
                args,
                raw_cache,
                use_hot=bool(profile["use_hot"]),
                use_extra=bool(profile["use_extra"]),
            )
            cfg = FeatureConfig(**profile["feature_config"])
            featured, _ = build_feature_frame(raw, cfg=cfg, with_labels=False)
            day = _latest_candidates(featured, feature_cols, cfg).sort_values(DATE_COL)
            if len(day) > args.shap_sample_rows:
                day = day.sample(args.shap_sample_rows, random_state=2026)
            x = day[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            explainer = shap.TreeExplainer(profile["model"])
            values = explainer.shap_values(x)
            if isinstance(values, list):
                values = values[0]
            shap_imp = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "mean_abs_shap": np.abs(np.asarray(values, dtype=float)).mean(axis=0),
                    "profile": name,
                }
            ).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(out_dir / f"{name}_shap_importance.csv", index=False)
            print(f"\n{name} SHAP top {min(args.top_n, len(shap_imp))}")
            print(shap_imp.head(args.top_n).to_string(index=False))

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


if __name__ == "__main__":
    main()
