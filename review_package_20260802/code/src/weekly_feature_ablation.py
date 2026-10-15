from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
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
from advanced.decay_risk import apply_decay_risk_adjustment, build_decay_risk_frame
from advanced.tree_ranker import TreeRankerConfig, score_with_models
from weekly_walkforward_backtest import (
    POLICY_GRID,
    _add_label_known_date,
    _fit_selected_models,
    _merge_extra_factor_dir,
    _merge_extra_hot_data,
    _policy_metrics,
    _rank_diagnostics,
    _score_rows_for_output,
    _valid_feature_rows,
    _weekly_cutoffs,
)


DEFAULT_SPECS = [
    ("all", "", ""),
    ("no_event", "", "ths_hot|limit_|top_|margin_"),
    ("no_cross_section", "", "_cs_(z|rank)$"),
    ("no_industry_relative", "", "_ind_(z|rank)$|f_industry"),
    ("no_valuation_size", "", "pe_ttm|pb|ps_ttm|total_mv|circ_mv|market_"),
    ("no_120_window", "", "_120"),
    ("only_mom_vol_liq", "close_ret|open_ret|ret_mean|ret_std|amount|volume|turnover|atr|vol_|mom_", ""),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple weekly feature-family ablations from one feature build.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output-dir", default="./model/feature_ablation")
    parser.add_argument("--start-date", default="2026-04-20")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--max-weeks", type=int, default=8)
    parser.add_argument("--min-train-weeks", type=int, default=52)
    parser.add_argument("--models", default="lightgbm")
    parser.add_argument("--profile", choices=["fast", "conservative"], default="fast")
    parser.add_argument("--relevance-mode", choices=["rank", "burst", "recall"], default="rank")
    parser.add_argument("--windows", default="3,5,10,20,30,60,120")
    parser.add_argument("--extra-hot-data", default="./model/extra_apr_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_apr_event_noboard")
    parser.add_argument("--event-state", action="store_true")
    parser.add_argument("--event-text", action="store_true")
    parser.add_argument("--decay-risk", action="store_true")
    parser.add_argument("--risk-penalty-weight", type=float, default=0.02)
    parser.add_argument("--train-lookback-days", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--train-half-life-days", type=float, default=0.0)
    parser.add_argument("--blend-weight", type=float, default=0.0)
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Ablation spec as name|keep_regex|drop_regex. May be repeated. Defaults to built-in specs.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_specs(values: list[str]) -> list[tuple[str, str, str]]:
    if not values:
        return DEFAULT_SPECS
    specs = []
    for value in values:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError("--spec must use name|keep_regex|drop_regex")
        specs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return specs


def _ranker_config(args: argparse.Namespace) -> TreeRankerConfig:
    conservative = args.profile == "conservative"
    return TreeRankerConfig(
        output_dir=args.output_dir,
        n_folds=1,
        val_size=5,
        embargo=0,
        min_train_dates=20,
        lgb_estimators=180 if not conservative else 700,
        xgb_estimators=160 if not conservative else 550,
        cat_estimators=180 if not conservative else 700,
        learning_rate=0.04 if not conservative else 0.025,
        num_leaves=31,
        max_depth=5 if not conservative else 6,
        min_child_samples=50 if not conservative else 45,
        subsample=0.80 if not conservative else 0.78,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=1.2,
        max_rows_per_fold=args.max_train_rows,
        relevance_mode=args.relevance_mode,
    )


def _select_cols(feature_cols: list[str], keep_regex: str, drop_regex: str) -> list[str]:
    selected = list(feature_cols)
    if keep_regex:
        keep = re.compile(keep_regex)
        selected = [c for c in selected if keep.search(c)]
    if drop_regex:
        drop = re.compile(drop_regex)
        selected = [c for c in selected if not drop.search(c)]
    if not selected:
        raise ValueError(f"No feature columns remain for keep={keep_regex!r} drop={drop_regex!r}")
    return selected


def _maybe_add_quality(scored: pd.DataFrame, pred_day: pd.DataFrame, blend_weight: float) -> pd.DataFrame:
    if blend_weight <= 0:
        return scored
    required = ["f_close_ret_10", "f_close_ret_20", "f_amount_rel_5", "f_ret_std_10"]
    if any(c not in pred_day.columns for c in required):
        return scored
    work = scored.merge(pred_day[[DATE_COL, ID_COL, *required]], on=[DATE_COL, ID_COL], how="left")
    work["model_pct"] = work.groupby(DATE_COL)["score"].rank(pct=True, method="first")
    quality_raw = (
        work.groupby(DATE_COL)["f_close_ret_10"].rank(pct=True, method="first")
        + 0.3 * work.groupby(DATE_COL)["f_close_ret_20"].rank(pct=True, method="first")
        + 0.2 * work.groupby(DATE_COL)["f_amount_rel_5"].rank(pct=True, method="first")
        - 0.2 * work.groupby(DATE_COL)["f_ret_std_10"].rank(pct=True, method="first")
    )
    work["quality_pct"] = quality_raw.groupby(work[DATE_COL]).rank(pct=True, method="first")
    w = float(np.clip(blend_weight, 0.0, 1.0))
    work["score_model_only"] = work["score"]
    work["score"] = (1.0 - w) * work["model_pct"] + w * work["quality_pct"]
    return work


def _hit_summary(scored: pd.DataFrame) -> dict[str, float]:
    rows = []
    for date, day in scored.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, "score"]).copy()
        if len(day) < 20:
            continue
        pred5 = day.sort_values("score", ascending=False).head(5)
        pred10_ids = set(day.sort_values("score", ascending=False).head(10)[ID_COL])
        pred20_ids = set(day.sort_values("score", ascending=False).head(20)[ID_COL])
        truth = day.sort_values(LABEL_COL, ascending=False)
        true5_ids = set(truth.head(5)[ID_COL])
        true10_ids = set(truth.head(10)[ID_COL])
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "rank1_return": float(pred5.iloc[0][LABEL_COL]),
                "top5_equal_return": float(pred5[LABEL_COL].mean()),
                "hit_true_top5": sum(x in true5_ids for x in pred5[ID_COL]),
                "hit_true_top10": sum(x in true10_ids for x in pred5[ID_COL]),
                "cover_true_top5_in_pred10": sum(x in pred10_ids for x in true5_ids),
                "cover_true_top10_in_pred20": sum(x in pred20_ids for x in true10_ids),
                "pred_top5": ",".join(pred5[ID_COL].astype(str).str.zfill(6).tolist()),
                "true_top5": ",".join(truth.head(5)[ID_COL].astype(str).str.zfill(6).tolist()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    return {
        "rank1_return": float(frame["rank1_return"].mean()),
        "rank1_hit10": int((frame["rank1_return"] >= 0.10).sum()),
        "top5_equal_return": float(frame["top5_equal_return"].mean()),
        "top5_equal_hit10": int((frame["top5_equal_return"] >= 0.10).sum()),
        "avg_hit_true_top5": float(frame["hit_true_top5"].mean()),
        "avg_hit_true_top10": float(frame["hit_true_top10"].mean()),
        "avg_cover_true_top5_in_pred10": float(frame["cover_true_top5_in_pred10"].mean()),
        "avg_cover_true_top10_in_pred20": float(frame["cover_true_top10_in_pred20"].mean()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    specs = _parse_specs(args.spec)

    data_dir = Path(args.data).parent
    full = enrich_with_metadata(load_price_data(args.data), data_dir)
    full = _merge_extra_hot_data(full, args.extra_hot_data)
    full = _merge_extra_factor_dir(full, args.extra_factor_dir)
    full = add_market_context(full)
    raw = filter_universe(full, load_universe_codes(args.target_universe))
    decay_risk = build_decay_risk_frame(raw) if args.decay_risk else None
    print(f"raw rows={len(raw)} stocks={raw[ID_COL].nunique()}")

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    cfg = FeatureConfig(
        windows=windows,
        min_history=min(60, max(windows)) if windows else 60,
        horizon=5,
        top_k=5,
        add_event_state=args.event_state,
        add_event_text=args.event_text,
    )
    featured, all_feature_cols = build_feature_frame(raw, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    weekly_dates = _weekly_cutoffs(featured[DATE_COL])
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date) if args.end_date else featured[DATE_COL].max()
    cutoffs = [d for d in weekly_dates if start <= d <= end]
    if args.max_weeks > 0:
        cutoffs = cutoffs[-args.max_weeks :]
    print(f"features={len(all_feature_cols)} cutoffs={len(cutoffs)}")

    ranker_cfg = _ranker_config(args)
    policy_specs = [
        {"name": name, "policy": policy, "temperature": temp, "max_per_industry": mpi}
        for name, policy, temp, mpi in POLICY_GRID
    ]
    summary_rows = []
    for spec_idx, (name, keep_regex, drop_regex) in enumerate(specs, start=1):
        feature_cols = _select_cols(all_feature_cols, keep_regex, drop_regex)
        print(f"\n[{spec_idx}/{len(specs)}] spec={name} features={len(feature_cols)} keep={keep_regex!r} drop={drop_regex!r}")
        trainable_all = filter_trainable_rows(featured, feature_cols, cfg)
        pred_all = _valid_feature_rows(featured, feature_cols, cfg)
        scored_parts = []
        cutoff_rows = []
        for i, cutoff in enumerate(cutoffs, start=1):
            known_train = trainable_all[trainable_all["label_known_date"] <= cutoff].copy()
            if args.train_lookback_days > 0:
                known_train = known_train[known_train[DATE_COL] >= cutoff - pd.Timedelta(days=args.train_lookback_days)].copy()
            train_weeks = len(_weekly_cutoffs(known_train[DATE_COL]))
            pred_day = pred_all[pred_all[DATE_COL] == cutoff].dropna(subset=[LABEL_COL]).copy()
            if train_weeks < args.min_train_weeks or len(pred_day) < 5:
                continue
            print(f"  [{i}/{len(cutoffs)}] cutoff={cutoff.date()} train_rows={len(known_train)} pred_rows={len(pred_day)}")
            models = _fit_selected_models(
                known_train,
                feature_cols,
                ranker_cfg,
                model_names,
                cutoff=cutoff,
                half_life_days=args.train_half_life_days,
            )
            scored = score_with_models(pred_day, feature_cols, models)
            extra_cols = [
                c
                for c in ["f_close_ret_10", "f_close_ret_20", "f_amount_rel_5", "f_ret_std_10", "行业"]
                if c in pred_day.columns and c not in scored.columns
            ]
            if extra_cols:
                scored = scored.merge(pred_day[[DATE_COL, ID_COL, *extra_cols]], on=[DATE_COL, ID_COL], how="left")
            scored = _maybe_add_quality(scored, pred_day, args.blend_weight)
            if decay_risk is not None:
                scored = apply_decay_risk_adjustment(
                    scored,
                    decay_risk,
                    score_col="score",
                    output_col="score",
                    penalty_weight=args.risk_penalty_weight,
                )
            scored["cutoff"] = cutoff
            scored_parts.append(scored)
            cutoff_rows.append({"cutoff": cutoff, "train_rows": len(known_train), "pred_rows": len(pred_day)})
        if not scored_parts:
            print(f"skip {name}: no scored cutoffs")
            continue
        scored_all = pd.concat(scored_parts, ignore_index=True)
        spec_dir = out_dir / name
        spec_dir.mkdir(parents=True, exist_ok=True)
        scored_all.to_csv(spec_dir / "walkforward_scores.csv", index=False)
        _rank_diagnostics(scored_all, "score").to_csv(spec_dir / "rank_diagnostics.csv", index=False)
        pd.DataFrame(cutoff_rows).to_csv(spec_dir / "cutoff_summary.csv", index=False)
        metrics = _policy_metrics(scored_all, policy_specs, "score")
        metrics.to_csv(spec_dir / "policy_metrics.csv", index=False)
        per_date = _score_rows_for_output(scored_all, policy_specs)
        per_date.to_csv(spec_dir / "per_date_portfolios.csv", index=False)
        hit_summary = _hit_summary(scored_all)
        best = metrics.iloc[0].to_dict()
        row = {
            "spec": name,
            "features": len(feature_cols),
            "keep_regex": keep_regex,
            "drop_regex": drop_regex,
            "cutoffs": len(scored_parts),
            "best_policy": best["name"],
            "best_policy_return": best["portfolio_return"],
            "best_policy_final_score": best["final_score"],
            **hit_summary,
        }
        summary_rows.append(row)
        _write_json(
            spec_dir / "metadata.json",
            {
                "args": vars(args),
                "feature_config": asdict(cfg),
                "ranker_config": asdict(ranker_cfg),
                "spec": row,
            },
        )
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(summary_rows).sort_values(["best_policy_return", "top5_equal_return"], ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    _write_json(out_dir / "metadata.json", {"args": vars(args), "specs": specs, "feature_count": len(all_feature_cols)})
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
