from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
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
    normalize_stock_code,
)
from advanced.metrics import evaluate_scored_frame, select_portfolio
from advanced.decay_risk import apply_decay_risk_adjustment, build_decay_risk_frame
from advanced.tree_ranker import MODEL_BUILDERS, TreeRankerConfig, score_with_models


POLICY_GRID = [
    ("conc_t0005_mpi1", "risk_adjusted", 0.005, 1),
    ("semi_t001_mpi1", "risk_adjusted", 0.01, 1),
    ("balanced_t002_mpi2", "risk_adjusted", 0.02, 2),
    ("div_t005_mpi2", "risk_adjusted", 0.05, 2),
    ("equal", "equal", 0.02, 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict weekly walk-forward backtest for submission-style CSI300 portfolios.")
    parser.add_argument("--data", default="./data/hybrid_stock_data.csv")
    parser.add_argument("--target-universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output-dir", default="./model/weekly_walkforward")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--min-train-weeks", type=int, default=52)
    parser.add_argument("--train-frequency", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--models", default="lightgbm", help="Comma-separated subset of lightgbm,xgboost,catboost.")
    parser.add_argument("--profile", choices=["fast", "conservative"], default="fast")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--train-lookback-days", type=int, default=0, help="Use only training rows within N calendar days before each cutoff; 0 means all history.")
    parser.add_argument("--train-half-life-days", type=float, default=0.0, help="Recency sample-weight half life for LightGBM training; 0 disables weighting.")
    parser.add_argument("--blend-weight", type=float, default=0.10, help="Weight for quality factor percentile.")
    parser.add_argument("--no-quality-blend", action="store_true")
    parser.add_argument("--max-weeks", type=int, default=0, help="Keep only the most recent N weekly cutoffs; 0 means all.")
    parser.add_argument("--windows", default="3,5,10,20,30,60,120", help="Comma-separated feature windows.")
    parser.add_argument("--relevance-mode", choices=["rank", "burst", "recall"], default="rank")
    parser.add_argument("--extra-hot-data", default="")
    parser.add_argument("--extra-factor-dir", default="")
    parser.add_argument("--event-state", action="store_true", help="Enable experimental event recency/decay features.")
    parser.add_argument("--event-text", action="store_true", help="Enable experimental ths_hot concept/reason keyword features.")
    parser.add_argument("--decay-risk", action="store_true", help="Apply strength-decay/high-reversal/event-proxy risk adjustment.")
    parser.add_argument("--risk-penalty-weight", type=float, default=0.02)
    parser.add_argument(
        "--keep-feature-regex",
        default="",
        help="Optional regex. Keep only feature columns matching it after feature generation.",
    )
    parser.add_argument(
        "--drop-feature-regex",
        default="",
        help="Optional regex. Drop feature columns matching it after feature generation.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_extra_hot_data(df: pd.DataFrame, path: str) -> pd.DataFrame:
    if not path:
        return df
    p = Path(path)
    if not p.exists():
        return df
    hot = pd.read_csv(p, dtype={ID_COL: str}, encoding="utf-8-sig")
    hot.columns = [str(c).replace("\ufeff", "").strip() for c in hot.columns]
    if DATE_COL not in hot.columns or ID_COL not in hot.columns:
        raise ValueError(f"{path} must contain {DATE_COL} and {ID_COL}")
    hot[DATE_COL] = pd.to_datetime(hot[DATE_COL])
    hot[ID_COL] = hot[ID_COL].map(normalize_stock_code)
    keep = [
        DATE_COL,
        ID_COL,
        "ths_hot_rank",
        "ths_hot_pct_change",
        "ths_hot_value",
        "ths_hot_concept",
        "ths_hot_reason",
        "rank_time",
    ]
    hot = hot[[c for c in keep if c in hot.columns]].drop_duplicates([DATE_COL, ID_COL], keep="last")
    out = df.merge(hot, on=[DATE_COL, ID_COL], how="left")
    print(f"merged ths_hot rows={len(hot)} hot_matches={out['ths_hot_rank'].notna().sum() if 'ths_hot_rank' in out.columns else 0}")
    return out


def _merge_extra_factor_dir(df: pd.DataFrame, directory: str) -> pd.DataFrame:
    out = df
    if not directory:
        return out
    base = Path(directory)
    if not base.exists():
        return out
    for name in [
        "limit_list_d.csv",
        "top_list.csv",
        "margin_detail.csv",
        "ths_board_stock_hot.csv",
        "event_semantic_factors.csv",
        "stk_factor.csv",
    ]:
        path = base / name
        if not path.exists():
            continue
        extra = pd.read_csv(path, dtype={ID_COL: str}, encoding="utf-8-sig")
        extra.columns = [str(c).replace("\ufeff", "").strip() for c in extra.columns]
        if DATE_COL not in extra.columns or ID_COL not in extra.columns:
            raise ValueError(f"{path} must contain {DATE_COL} and {ID_COL}")
        extra[DATE_COL] = pd.to_datetime(extra[DATE_COL])
        extra[ID_COL] = extra[ID_COL].map(normalize_stock_code)
        extra = extra.drop_duplicates([DATE_COL, ID_COL], keep="last")
        value_cols = [c for c in extra.columns if c not in {DATE_COL, ID_COL}]
        before_cols = set(out.columns)
        out = out.merge(extra[[DATE_COL, ID_COL, *value_cols]], on=[DATE_COL, ID_COL], how="left")
        added = [c for c in out.columns if c not in before_cols]
        matched = int(out[added[0]].notna().sum()) if added else 0
        print(f"merged {name} rows={len(extra)} added_cols={len(added)} first_col_matches={matched}")
    return out


def _weekly_cutoffs(dates: pd.Series) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if unique.empty:
        return []
    periods = unique.dt.to_period("W-FRI")
    return [pd.Timestamp(x) for x in unique.groupby(periods).max().tolist()]


def _add_label_known_date(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values([ID_COL, DATE_COL]).copy()
    work["label_known_date"] = work.groupby(ID_COL, sort=False)[DATE_COL].shift(-horizon)
    return work


def _valid_feature_rows(df: pd.DataFrame, feature_cols: list[str], cfg: FeatureConfig) -> pd.DataFrame:
    work = df.copy()
    work["_history_count"] = work.groupby(ID_COL).cumcount() + 1
    work = work[work["_history_count"] >= cfg.min_history].copy()
    finite = np.isfinite(work[feature_cols].to_numpy(dtype=np.float32)).all(axis=1)
    return work.loc[finite].drop(columns=["_history_count"]).reset_index(drop=True)


def _select_feature_cols(feature_cols: list[str], args: argparse.Namespace) -> list[str]:
    selected = list(feature_cols)
    if args.keep_feature_regex:
        keep = re.compile(args.keep_feature_regex)
        selected = [c for c in selected if keep.search(c)]
    if args.drop_feature_regex:
        drop = re.compile(args.drop_feature_regex)
        selected = [c for c in selected if not drop.search(c)]
    if not selected:
        raise ValueError("No feature columns remain after keep/drop regex filtering.")
    print(f"selected features: {len(selected)}/{len(feature_cols)}")
    return selected


def _ranker_config(output_dir: str, args: argparse.Namespace) -> TreeRankerConfig:
    conservative = args.profile == "conservative"
    return TreeRankerConfig(
        output_dir=output_dir,
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
        progress_path=str(Path(output_dir) / "progress.json"),
        relevance_mode=args.relevance_mode,
    )


def _maybe_downsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(df.index.to_numpy(), size=max_rows, replace=False)
    return df.loc[np.sort(keep)].copy()


def _fit_selected_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: TreeRankerConfig,
    model_names: list[str],
    cutoff: pd.Timestamp | None = None,
    half_life_days: float = 0.0,
) -> dict[str, Any]:
    train_df = _maybe_downsample(train_df, cfg.max_rows_per_fold, cfg.random_state)
    models: dict[str, Any] = {}
    for name in model_names:
        if name not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model {name}; choose from {sorted(MODEL_BUILDERS)}")
        try:
            if name == "lightgbm" and cutoff is not None and half_life_days > 0:
                models[name] = _fit_lightgbm_recency_weighted(train_df, feature_cols, cfg, cutoff, half_life_days)
            else:
                models[name] = MODEL_BUILDERS[name](train_df, feature_cols, cfg)
            print(f"trained {name}")
        except ImportError as exc:
            print(f"skip {name}: dependency is not installed ({exc})")
        except Exception as exc:
            print(f"skip {name}: training failed ({exc})")
    if not models:
        raise RuntimeError("No ranker model was trained.")
    return models


def _date_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby(DATE_COL, sort=True).size().astype(int).tolist()


def _make_relevance(df: pd.DataFrame, top_k: int, mode: str = "rank") -> pd.Series:
    def _day(s: pd.Series) -> pd.Series:
        rank = s.rank(method="first", ascending=False)
        if mode == "burst":
            rel = np.zeros(len(s), dtype=int)
            rel += np.where(rank.to_numpy() <= 1, 10, 0)
            rel += np.where((rank.to_numpy() > 1) & (rank.to_numpy() <= 3), 7, 0)
            rel += np.where((rank.to_numpy() > 3) & (rank.to_numpy() <= top_k), 5, 0)
            rel += np.where((rank.to_numpy() > top_k) & (rank.to_numpy() <= 10), 2, 0)
            values = s.to_numpy(dtype=float)
            rel += np.where(values >= 0.10, 4, 0)
            rel += np.where((values >= 0.06) & (values < 0.10), 2, 0)
            rel += np.where((values >= 0.03) & (values < 0.06), 1, 0)
            rel -= np.where(values < 0.0, 1, 0)
            rel = np.clip(rel, 0, None)
        elif mode == "recall":
            rel = np.zeros(len(s), dtype=int)
            rank_values = rank.to_numpy()
            values = s.to_numpy(dtype=float)
            rel += np.where(rank_values <= 1, 8, 0)
            rel += np.where((rank_values > 1) & (rank_values <= 3), 6, 0)
            rel += np.where((rank_values > 3) & (rank_values <= top_k), 5, 0)
            rel += np.where((rank_values > top_k) & (rank_values <= 10), 3, 0)
            rel += np.where((rank_values > 10) & (rank_values <= 20), 1, 0)
            rel += np.where(values >= 0.10, 4, 0)
            rel += np.where((values >= 0.06) & (values < 0.10), 2, 0)
            rel += np.where((values >= 0.03) & (values < 0.06), 1, 0)
            rel -= np.where(values < -0.03, 1, 0)
            rel = np.clip(rel, 0, None)
        else:
            rel = np.maximum(0, top_k + 1 - rank.to_numpy()).astype(int)
        return pd.Series(rel, index=s.index)

    return df.groupby(DATE_COL, group_keys=False)[LABEL_COL].apply(_day).astype(int)


def _fit_lightgbm_recency_weighted(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: TreeRankerConfig,
    cutoff: pd.Timestamp,
    half_life_days: float,
) -> Any:
    from lightgbm import LGBMRanker

    ordered = train_df.sort_values([DATE_COL, ID_COL]).copy()
    age_days = (pd.Timestamp(cutoff) - ordered[DATE_COL]).dt.days.clip(lower=0).to_numpy(dtype=float)
    sample_weight = np.power(0.5, age_days / max(float(half_life_days), 1e-6))
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=cfg.lgb_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
        importance_type="gain",
        verbose=-1,
    )
    model.fit(
        ordered[feature_cols],
        _make_relevance(ordered, cfg.top_k, cfg.relevance_mode),
        group=_date_groups(ordered),
        sample_weight=sample_weight,
    )
    return model


def _add_quality_blend(scored: pd.DataFrame, blend_weight: float) -> pd.DataFrame:
    work = scored.copy()
    required = ["f_close_ret_10", "f_close_ret_20", "f_amount_rel_5", "f_ret_std_10"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Quality blend needs missing feature columns: {missing}")
    work["model_pct"] = work.groupby(DATE_COL)["score"].rank(pct=True, method="first")
    work["quality_raw"] = (
        work.groupby(DATE_COL)["f_close_ret_10"].rank(pct=True, method="first")
        + 0.3 * work.groupby(DATE_COL)["f_close_ret_20"].rank(pct=True, method="first")
        + 0.2 * work.groupby(DATE_COL)["f_amount_rel_5"].rank(pct=True, method="first")
        - 0.2 * work.groupby(DATE_COL)["f_ret_std_10"].rank(pct=True, method="first")
    )
    work["quality_pct"] = work.groupby(DATE_COL)["quality_raw"].rank(pct=True, method="first")
    w = float(np.clip(blend_weight, 0.0, 1.0))
    work["score_model_only"] = work["score"]
    work["score"] = (1.0 - w) * work["model_pct"] + w * work["quality_pct"]
    return work


def _score_rows_for_output(scored: pd.DataFrame, policy_rows: list[dict[str, Any]]) -> pd.DataFrame:
    out_rows = []
    for date, day in scored.groupby(DATE_COL, sort=True):
        if len(day.dropna(subset=[LABEL_COL, "score"])) < 5:
            continue
        for spec in policy_rows:
            pred = select_portfolio(
                day,
                top_k=5,
                policy=spec["policy"],
                temperature=spec["temperature"],
                max_per_industry=spec["max_per_industry"],
            )
            merged = pred.merge(day[[ID_COL, LABEL_COL]], on=ID_COL, how="left")
            out_rows.append(
                {
                    "date": date,
                    "policy_name": spec["name"],
                    "portfolio_return": float((merged["weight"] * merged[LABEL_COL]).sum()),
                    "max_weight": float(pred["weight"].max()),
                    "selected": ",".join(pred[ID_COL].astype(str).str.zfill(6).tolist()),
                    "weights": ",".join(f"{x:.8f}" for x in pred["weight"].to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(out_rows)


def _rank_diagnostics(scored: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    rows = []
    top1_returns = []
    top5_returns = []
    fail_returns = []
    high_score_returns = []
    for date, day in scored.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, score_col]).copy()
        if len(day) < 10:
            continue
        ranked = day.sort_values(score_col, ascending=False)
        true_rank = day[LABEL_COL].rank(method="first", ascending=False)
        true_top5 = set(day.loc[true_rank <= 5, ID_COL])
        true_top10 = set(day.loc[true_rank <= 10, ID_COL])
        pred_top1 = ranked.head(1)
        pred_top5 = ranked.head(5)
        pred_top10 = ranked.head(10)
        top1_ret = float(pred_top1[LABEL_COL].iloc[0])
        top5_ret = float(pred_top5[LABEL_COL].mean())
        top1_returns.append(top1_ret)
        top5_returns.append(top5_ret)
        high_score_returns.extend(pred_top10[LABEL_COL].astype(float).tolist())
        if top1_ret < 0:
            fail_returns.append(top1_ret)
        rows.append(
            {
                "date": date,
                "top1_return": top1_ret,
                "top5_equal_return": top5_ret,
                "top1_in_true_top5": int(pred_top1[ID_COL].iloc[0] in true_top5),
                "top1_in_true_top10": int(pred_top1[ID_COL].iloc[0] in true_top10),
                "top5_hit_true_top5": len(set(pred_top5[ID_COL]) & true_top5),
                "top5_hit_true_top10": len(set(pred_top5[ID_COL]) & true_top10),
                "top10_precision_true_top10": len(set(pred_top10[ID_COL]) & true_top10) / 10.0,
                "rank1_failed": int(top1_ret < 0),
                "rank1_failure_drawdown": min(top1_ret, 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    diag = pd.DataFrame(rows)
    summary = {
        "date": "SUMMARY",
        "top1_return": float(np.mean(top1_returns)),
        "top5_equal_return": float(np.mean(top5_returns)),
        "top1_in_true_top5": float(diag["top1_in_true_top5"].mean()),
        "top1_in_true_top10": float(diag["top1_in_true_top10"].mean()),
        "top5_hit_true_top5": float(diag["top5_hit_true_top5"].mean()),
        "top5_hit_true_top10": float(diag["top5_hit_true_top10"].mean()),
        "top10_precision_true_top10": float(diag["top10_precision_true_top10"].mean()),
        "rank1_failed": int(diag["rank1_failed"].sum()),
        "rank1_failure_drawdown": float(np.min(fail_returns)) if fail_returns else 0.0,
        "high_score_return_p10": float(np.quantile(high_score_returns, 0.10)),
        "high_score_return_median": float(np.median(high_score_returns)),
        "high_score_return_p90": float(np.quantile(high_score_returns, 0.90)),
    }
    return pd.concat([diag, pd.DataFrame([summary])], ignore_index=True)


def _policy_metrics(scored: pd.DataFrame, policy_specs: list[dict[str, Any]], score_col: str) -> pd.DataFrame:
    work = scored.copy()
    work["score"] = work[score_col]
    rows = []
    for spec in policy_specs:
        metric = evaluate_scored_frame(
            work,
            top_k=5,
            policy=spec["policy"],
            temperature=spec["temperature"],
            max_per_industry=spec["max_per_industry"],
        )
        rows.append({**spec, **metric})
    return pd.DataFrame(rows).sort_values(["final_score", "portfolio_return"], ascending=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]

    data_dir = Path(args.data).parent
    full_df = enrich_with_metadata(load_price_data(args.data), data_dir)
    full_df = _merge_extra_hot_data(full_df, args.extra_hot_data)
    full_df = _merge_extra_factor_dir(full_df, args.extra_factor_dir)
    full_df = add_market_context(full_df)
    raw_df = filter_universe(full_df, load_universe_codes(args.target_universe))
    decay_risk = build_decay_risk_frame(raw_df) if args.decay_risk else None
    print(f"full rows: {len(full_df)}, target rows: {len(raw_df)}, target stocks: {raw_df[ID_COL].nunique()}")

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    min_history = min(60, max(windows)) if windows else 60
    feature_cfg = FeatureConfig(
        windows=windows,
        min_history=min_history,
        horizon=5,
        top_k=5,
        add_event_state=args.event_state,
        add_event_text=args.event_text,
    )
    ranker_cfg = _ranker_config(str(output_dir), args)
    _write_json(progress_path, {"stage": "feature_build", "status": "running", "args": vars(args)})

    featured, feature_cols = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=True)
    feature_cols = _select_feature_cols(feature_cols, args)
    featured = _add_label_known_date(featured, feature_cfg.horizon)
    trainable_all = filter_trainable_rows(featured, feature_cols, feature_cfg)
    pred_all = _valid_feature_rows(featured, feature_cols, feature_cfg)
    weekly_dates = _weekly_cutoffs(pred_all[DATE_COL])
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date) if args.end_date else pred_all[DATE_COL].max()
    cutoffs = [d for d in weekly_dates if start_date <= d <= end_date]
    cutoffs = [d for d in cutoffs if d <= trainable_all[DATE_COL].max()]
    if args.max_weeks > 0:
        cutoffs = cutoffs[-args.max_weeks :]
    if not cutoffs:
        raise RuntimeError("No weekly cutoffs available for the requested date range.")

    all_weekly_train_dates = _weekly_cutoffs(trainable_all[DATE_COL])
    policy_specs = [
        {"name": name, "policy": policy, "temperature": temp, "max_per_industry": mpi}
        for name, policy, temp, mpi in POLICY_GRID
    ]
    print(
        f"feature rows: {len(featured)}, trainable rows: {len(trainable_all)}, "
        f"prediction rows: {len(pred_all)}, features: {len(feature_cols)}, cutoffs: {len(cutoffs)}"
    )

    scored_parts = []
    cutoff_rows = []
    for i, cutoff in enumerate(cutoffs, start=1):
        known_train = trainable_all[trainable_all["label_known_date"] <= cutoff].copy()
        if args.train_lookback_days > 0:
            min_train_date = cutoff - pd.Timedelta(days=args.train_lookback_days)
            known_train = known_train[known_train[DATE_COL] >= min_train_date].copy()
        if args.train_frequency == "weekly":
            eligible_weekly = [d for d in all_weekly_train_dates if d in set(known_train[DATE_COL])]
            known_train = known_train[known_train[DATE_COL].isin(eligible_weekly)].copy()
            train_week_count = len(eligible_weekly)
        else:
            train_week_count = len(_weekly_cutoffs(known_train[DATE_COL]))
        if train_week_count < args.min_train_weeks:
            continue
        pred_day = pred_all[pred_all[DATE_COL] == cutoff].copy()
        pred_day = pred_day.dropna(subset=[LABEL_COL])
        if len(pred_day) < 5:
            continue
        print(
            f"[{i}/{len(cutoffs)}] cutoff {cutoff.date()} "
            f"train_rows={len(known_train)} train_weeks={train_week_count} pred_rows={len(pred_day)}"
        )
        _write_json(
            progress_path,
            {
                "stage": "walk_forward",
                "status": "running",
                "current_cutoff": str(cutoff.date()),
                "completed_cutoffs": len(scored_parts),
                "total_cutoffs": len(cutoffs),
                "train_rows": len(known_train),
                "train_weeks": train_week_count,
            },
        )
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
        scored = scored.merge(pred_day[[DATE_COL, ID_COL, *extra_cols]], on=[DATE_COL, ID_COL], how="left")
        if not args.no_quality_blend:
            scored = _add_quality_blend(scored, args.blend_weight)
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
        cutoff_rows.append(
            {
                "cutoff": cutoff,
                "train_rows": len(known_train),
                "train_start": str(known_train[DATE_COL].min().date()),
                "last_label_date": str(known_train[DATE_COL].max().date()),
                "max_label_known_date": str(known_train["label_known_date"].max().date()),
                "pred_rows": len(pred_day),
            }
        )

    if not scored_parts:
        raise RuntimeError("No cutoffs were evaluated. Lower --min-train-weeks or adjust date range.")

    oof = pd.concat(scored_parts, ignore_index=True)
    oof.to_csv(output_dir / "walkforward_scores.csv", index=False)
    _rank_diagnostics(oof, "score").to_csv(output_dir / "rank_diagnostics.csv", index=False)
    pd.DataFrame(cutoff_rows).to_csv(output_dir / "cutoff_summary.csv", index=False)

    metrics = _policy_metrics(oof, policy_specs, "score")
    metrics.to_csv(output_dir / "policy_metrics.csv", index=False)
    metric_tables = [metrics.assign(score_type="primary")]
    if "score_model_only" in oof.columns:
        model_only_metrics = _policy_metrics(oof, policy_specs, "score_model_only")
        model_only_metrics.to_csv(output_dir / "policy_metrics_model_only.csv", index=False)
        metric_tables.append(model_only_metrics.assign(score_type="model_only"))
    pd.concat(metric_tables, ignore_index=True).to_csv(output_dir / "policy_metrics_all.csv", index=False)
    per_date = _score_rows_for_output(oof, policy_specs)
    per_date.to_csv(output_dir / "per_date_portfolios.csv", index=False)

    best = metrics.iloc[0].to_dict()
    metadata = {
        "feature_config": asdict(feature_cfg),
        "ranker_config": asdict(ranker_cfg),
        "feature_cols": feature_cols,
        "feature_filter": {
            "keep_feature_regex": args.keep_feature_regex,
            "drop_feature_regex": args.drop_feature_regex,
        },
        "models": model_names,
        "train_frequency": args.train_frequency,
        "quality_blend": not args.no_quality_blend,
        "blend_weight": args.blend_weight,
        "cutoff_count": len(cutoff_rows),
        "first_cutoff": str(pd.to_datetime(oof[DATE_COL]).min().date()),
        "last_cutoff": str(pd.to_datetime(oof[DATE_COL]).max().date()),
        "best_policy": best,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(progress_path, {"stage": "finished", "status": "done", **metadata})
    print(metrics.to_string(index=False))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
