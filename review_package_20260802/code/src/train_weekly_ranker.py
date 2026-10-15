from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from advanced.features import (
    DATE_COL,
    FeatureConfig,
    ID_COL,
    add_market_context,
    build_feature_frame,
    enrich_with_metadata,
    filter_trainable_rows,
    filter_universe,
    load_price_data,
    load_universe_codes,
)
from advanced.metrics import evaluate_scored_frame
from advanced.tree_ranker import TreeRankerConfig, fit_available_models, score_with_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a weekly submission-style tree ranker.")
    parser.add_argument("--data", default="./data/hybrid_stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output", default="./model/weekly_ranker_20260626")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-weeks", type=int, default=4)
    parser.add_argument("--min-train-weeks", type=int, default=26)
    parser.add_argument("--ranker-profile", choices=["default", "conservative"], default="conservative")
    return parser.parse_args()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _weekly_dates(dates: pd.Series | list[pd.Timestamp]) -> list[pd.Timestamp]:
    s = pd.Series(pd.to_datetime(pd.Series(dates).drop_duplicates()).sort_values())
    if s.empty:
        return []
    periods = s.dt.to_period("W-FRI")
    return [pd.Timestamp(x) for x in s.groupby(periods).max().tolist()]


def _make_weekly_folds(
    dates: list[pd.Timestamp],
    n_folds: int,
    val_weeks: int,
    min_train_weeks: int,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    dates = list(pd.DatetimeIndex(pd.to_datetime(dates)).sort_values())
    folds = []
    first_val = max(min_train_weeks, len(dates) - n_folds * val_weeks)
    for start in range(first_val, len(dates), val_weeks):
        end = min(start + val_weeks, len(dates))
        train_dates = dates[:start]
        val_dates = dates[start:end]
        if len(train_dates) >= min_train_weeks and val_dates:
            folds.append((train_dates, val_dates))
    return folds[-n_folds:]


def _ranker_config(output_dir: str, profile: str, folds: int) -> TreeRankerConfig:
    conservative = profile == "conservative"
    return TreeRankerConfig(
        output_dir=output_dir,
        n_folds=folds,
        val_size=20,
        embargo=0,
        min_train_dates=20,
        lgb_estimators=900 if conservative else 700,
        xgb_estimators=700 if conservative else 550,
        cat_estimators=900 if conservative else 700,
        learning_rate=0.025 if conservative else 0.035,
        num_leaves=31 if conservative else 63,
        max_depth=6 if conservative else -1,
        min_child_samples=45 if conservative else 25,
        subsample=0.78 if conservative else 0.85,
        colsample_bytree=0.75 if conservative else 0.85,
        reg_alpha=0.15 if conservative else 0.05,
        reg_lambda=1.2 if conservative else 0.2,
        progress_path=str(Path(output_dir) / "progress.json"),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    data_dir = Path(args.data).parent
    full_df = enrich_with_metadata(load_price_data(args.data), data_dir)
    full_df = add_market_context(full_df)
    raw_df = filter_universe(full_df, load_universe_codes(args.target_universe))
    print(f"full rows: {len(full_df)}, target rows: {len(raw_df)}, target stocks: {raw_df[ID_COL].nunique()}")

    feature_cfg = FeatureConfig(windows=(3, 5, 10, 20, 30, 60, 120), min_history=60, horizon=5, top_k=5)
    ranker_cfg = _ranker_config(str(output_dir), args.ranker_profile, args.folds)

    _write_progress(progress_path, {"stage": "feature_build", "status": "running"})
    featured, feature_cols = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=True)
    trainable = filter_trainable_rows(featured, feature_cols, feature_cfg)
    weekly_dates = _weekly_dates(trainable[DATE_COL])
    weekly = trainable[trainable[DATE_COL].isin(weekly_dates)].copy()
    dates = list(pd.DatetimeIndex(weekly[DATE_COL].unique()).sort_values())
    folds = _make_weekly_folds(dates, args.folds, args.val_weeks, args.min_train_weeks)
    if not folds:
        raise RuntimeError("Unable to build weekly folds.")

    print(f"weekly trainable rows: {len(weekly)}, weekly dates: {len(dates)}, features: {len(feature_cols)}, folds: {len(folds)}")
    _write_progress(
        progress_path,
        {
            "stage": "rolling_validation",
            "status": "running",
            "weekly_rows": len(weekly),
            "weekly_dates": len(dates),
            "feature_count": len(feature_cols),
            "fold_count": len(folds),
        },
    )

    oof_parts = []
    fold_rows = []
    for i, (train_dates, val_dates) in enumerate(folds, start=1):
        train_df = weekly[weekly[DATE_COL].isin(train_dates)].copy()
        val_df = weekly[weekly[DATE_COL].isin(val_dates)].copy()
        print(
            f"fold {i}: train {min(train_dates).date()}..{max(train_dates).date()} "
            f"val {min(val_dates).date()}..{max(val_dates).date()}"
        )
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": i,
                "fold_count": len(folds),
                "completed_folds": i - 1,
                "train_weeks": len(train_dates),
                "val_weeks": len(val_dates),
            },
        )
        models = fit_available_models(train_df, feature_cols, ranker_cfg)
        scored = score_with_models(val_df, feature_cols, models)
        oof_parts.append(scored.assign(fold=i))
        metric = evaluate_scored_frame(scored, top_k=5, policy="risk_adjusted", temperature=0.02, max_per_industry=2)
        fold_rows.append({"fold": i, **metric})
        _write_progress(
            progress_path,
            {
                "stage": "rolling_validation",
                "status": "running",
                "current_fold": i,
                "fold_count": len(folds),
                "completed_folds": i,
                "latest_fold_metric": fold_rows[-1],
            },
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "oof_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "rolling_validation.csv", index=False)

    policy_grid = []
    for name, policy, temp, mpi in [
        ("concentrated", "risk_adjusted", 0.005, 1),
        ("semi_concentrated", "risk_adjusted", 0.01, 1),
        ("balanced", "risk_adjusted", 0.02, 2),
        ("diversified", "risk_adjusted", 0.05, 2),
        ("equal", "equal", 0.02, 2),
    ]:
        metric = evaluate_scored_frame(oof, top_k=5, policy=policy, temperature=temp, max_per_industry=mpi)
        policy_grid.append({"name": name, "policy": policy, "temperature": temp, "max_per_industry": mpi, **metric})
    policy_df = pd.DataFrame(policy_grid).sort_values(["final_score", "portfolio_return"], ascending=False)
    policy_df.to_csv(output_dir / "weight_policy_metrics.csv", index=False)
    best = policy_df.iloc[0].to_dict()
    print(policy_df.to_string(index=False))

    _write_progress(progress_path, {"stage": "final_train", "status": "running", "best_policy": best})
    final_models = fit_available_models(weekly, feature_cols, ranker_cfg)
    artifact = {
        "feature_config": asdict(feature_cfg),
        "ranker_config": asdict(ranker_cfg),
        "feature_cols": feature_cols,
        "models": final_models,
        "best_policy": best,
        "fold_summaries": fold_rows,
        "weekly_train_dates": (str(min(dates).date()), str(max(dates).date())),
    }
    joblib.dump(artifact, output_dir / "tree_ranker.joblib")
    _write_progress(
        progress_path,
        {
            "stage": "finished",
            "status": "done",
            "weekly_rows": len(weekly),
            "weekly_dates": len(dates),
            "feature_count": len(feature_cols),
            "best_policy": best,
            "artifact": str(output_dir / "tree_ranker.joblib"),
        },
    )


if __name__ == "__main__":
    main()
