from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advanced.features import (
    DATE_COL,
    FeatureConfig,
    ID_COL,
    LABEL_COL,
    add_market_context,
    build_feature_frame,
    enrich_with_metadata,
    filter_trainable_rows,
    filter_universe,
    load_price_data,
    load_universe_codes,
)
from train_current_lgb_submission import merge_extra_factor_dir, merge_extra_hot_data
from weekly_walkforward_backtest import _add_label_known_date, _valid_feature_rows, _weekly_cutoffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward candidate recall classifiers for top10/10pct stock selection.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output-dir", default="./model/weekly_candidate_classifier")
    parser.add_argument("--start-date", default="2026-04-20")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--max-weeks", type=int, default=14)
    parser.add_argument("--extra-hot-data", default="./model/extra_apr_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_apr_event_noboard")
    parser.add_argument("--train-lookback-days", type=int, default=0)
    parser.add_argument("--estimators", type=int, default=120)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--targets", default="top10,top20,ge10")
    parser.add_argument(
        "--classifier-models",
        default="lgb,xgb,extra,sgd",
        help="Comma-separated subset of lgb,xgb,extra,sgd. Scores are percentile-fused per target.",
    )
    return parser.parse_args()


def _make_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank = out.groupby(DATE_COL)[LABEL_COL].rank(method="first", ascending=False)
    out["target_top10"] = (rank <= 10).astype(int)
    out["target_top20"] = (rank <= 20).astype(int)
    out["target_ge10"] = (out[LABEL_COL] >= 0.10).astype(int)
    out["target_ge06"] = (out[LABEL_COL] >= 0.06).astype(int)
    return out


def _fit_lgb_classifier(train: pd.DataFrame, feature_cols: list[str], target: str, estimators: int, seed: int = 42) -> Any:
    from lightgbm import LGBMClassifier

    y = train[target].astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=45,
        subsample=0.82,
        colsample_bytree=0.78,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=min(neg / pos, 30.0),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train[feature_cols], y)
    return model


def _fit_xgb_classifier(train: pd.DataFrame, feature_cols: list[str], target: str, estimators: int, seed: int = 42) -> Any:
    from xgboost import XGBClassifier

    y = train[target].astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=estimators,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=8,
        subsample=0.82,
        colsample_bytree=0.72,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=min(neg / pos, 40.0),
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], y, verbose=False)
    return model


def _fit_extra_classifier(train: pd.DataFrame, feature_cols: list[str], target: str, estimators: int, seed: int = 42) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    y = train[target].astype(int)
    model = ExtraTreesClassifier(
        n_estimators=max(200, estimators),
        max_depth=8,
        min_samples_leaf=25,
        max_features=0.35,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], y)
    return model


def _fit_sgd_classifier(train: pd.DataFrame, feature_cols: list[str], target: str, estimators: int, seed: int = 42) -> Any:
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = train[target].astype(int)
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0007,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=max(600, estimators * 4),
            tol=1e-3,
            random_state=seed,
        ),
    )
    model.fit(train[feature_cols], y)
    return model


CLASSIFIER_BUILDERS = {
    "lgb": _fit_lgb_classifier,
    "xgb": _fit_xgb_classifier,
    "extra": _fit_extra_classifier,
    "sgd": _fit_sgd_classifier,
}


def _score_day(day: pd.DataFrame, feature_cols: list[str], models: dict[str, Any]) -> pd.DataFrame:
    scored = day[[DATE_COL, ID_COL, LABEL_COL]].copy()
    for name, model in models.items():
        scored[f"score_{name}"] = model.predict_proba(day[feature_cols])[:, 1]
    for col in [c for c in scored.columns if c.startswith("score_")]:
        scored[f"{col}_pct"] = scored[col].rank(pct=True, method="first")
    weights = {"top10": 0.40, "top20": 0.25, "ge10": 0.25, "ge06": 0.10}
    score = np.zeros(len(scored), dtype=float)
    total = 0.0
    for name, weight in weights.items():
        cols = [c for c in scored.columns if c.startswith(f"score_{name}_") and c.endswith("_pct")]
        if cols:
            target_score = scored[cols].mean(axis=1).to_numpy(dtype=float)
            scored[f"score_{name}_blend_pct"] = target_score
            score += weight * target_score
            total += weight
    scored["score"] = score / max(total, 1e-12)
    return scored


def _eval(scored: pd.DataFrame) -> pd.DataFrame:
    templates = {
        "equal": np.ones(5) / 5,
        "tpl60": np.array([0.6, 0.2, 0.15, 0.049999, 0.000001]),
        "tpl80": np.array([0.8, 0.1, 0.05, 0.03, 0.02]),
        "tpl90": np.array([0.9, 0.05, 0.03, 0.01, 0.01]),
    }
    rows = []
    for date, day in scored.groupby(DATE_COL, sort=True):
        top = day.sort_values("score", ascending=False).head(5)
        truth = day.sort_values(LABEL_COL, ascending=False)
        true10 = set(truth.head(10)[ID_COL])
        true5 = set(truth.head(5)[ID_COL])
        row = {
            "date": pd.Timestamp(date).date().isoformat(),
            "selected": ",".join(top[ID_COL].astype(str).str.zfill(6).tolist()),
            "hit_true_top5": sum(x in true5 for x in top[ID_COL]),
            "hit_true_top10": sum(x in true10 for x in top[ID_COL]),
        }
        vals = top[LABEL_COL].to_numpy(dtype=float)
        for name, weights in templates.items():
            weights = weights / weights.sum()
            row[name] = float(np.sum(vals * weights))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data).parent
    raw = enrich_with_metadata(load_price_data(args.data), data_dir)
    raw = merge_extra_hot_data(raw, args.extra_hot_data)
    raw = merge_extra_factor_dir(raw, args.extra_factor_dir)
    raw = add_market_context(raw)
    raw = filter_universe(raw, load_universe_codes(args.target_universe))
    cfg = FeatureConfig(windows=(3, 5, 10, 20, 30, 60, 120), min_history=60, horizon=5, top_k=5)
    featured, feature_cols = build_feature_frame(raw, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    trainable = _make_targets(filter_trainable_rows(featured, feature_cols, cfg))
    pred_all = _valid_feature_rows(featured, feature_cols, cfg).dropna(subset=[LABEL_COL])
    cutoffs = [d for d in _weekly_cutoffs(pred_all[DATE_COL]) if pd.Timestamp(args.start_date) <= d <= pd.Timestamp(args.end_date)]
    if args.max_weeks:
        cutoffs = cutoffs[-args.max_weeks :]
    parts = []
    summary = []
    target_names = [x.strip() for x in args.targets.split(",") if x.strip()]
    classifier_names = [x.strip() for x in args.classifier_models.split(",") if x.strip()]
    for i, cutoff in enumerate(cutoffs, start=1):
        train = trainable[trainable["label_known_date"] <= cutoff].copy()
        if args.train_lookback_days:
            train = train[train[DATE_COL] >= cutoff - pd.Timedelta(days=args.train_lookback_days)].copy()
        if args.max_train_rows and len(train) > args.max_train_rows:
            train = train.sample(args.max_train_rows, random_state=1000 + i).sort_values([DATE_COL, ID_COL])
        day = pred_all[pred_all[DATE_COL] == cutoff].copy()
        print(f"[{i}/{len(cutoffs)}] cutoff={cutoff.date()} train={len(train)} day={len(day)}")
        target_cols = {"top10": "target_top10", "top20": "target_top20", "ge10": "target_ge10", "ge06": "target_ge06"}
        models = {}
        for j, target_name in enumerate(target_names):
            for k, model_name in enumerate(classifier_names):
                if model_name not in CLASSIFIER_BUILDERS:
                    raise ValueError(f"Unknown classifier model {model_name}; choose from {sorted(CLASSIFIER_BUILDERS)}")
                key = f"{target_name}_{model_name}"
                models[key] = CLASSIFIER_BUILDERS[model_name](train, feature_cols, target_cols[target_name], args.estimators, 40 + j * 10 + k)
        scored = _score_day(day, feature_cols, models)
        parts.append(scored)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(out_dir / "walkforward_scores.csv", index=False)
    per = _eval(oof)
    per.to_csv(out_dir / "per_date.csv", index=False)
    for col in ["equal", "tpl60", "tpl80", "tpl90"]:
        s = per[col]
        summary.append(
            {
                "policy": col,
                "avg": float(s.mean()),
                "hit10": int((s >= 0.10).sum()),
                "min": float(s.min()),
                "avg_hit_true_top5": float(per["hit_true_top5"].mean()),
                "avg_hit_true_top10": float(per["hit_true_top10"].mean()),
            }
        )
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(per.to_string(index=False))
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
