from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from advanced.features import (
    AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    ID_COL,
    LOW_COL,
    OPEN_COL,
    PCT_COL,
    FeatureConfig,
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
from advanced.decay_risk import apply_decay_risk_adjustment, build_decay_risk_frame, confidence_weight_template
from advanced.tree_ranker import TreeRankerConfig, _fit_lightgbm
from train_current_lgb_submission import merge_extra_hot_data, maybe_load_universe, resolve_data_path


PROFILES = [
    ("base_rank", "rank", 0.60, False, False, False),
    ("event_text_rank", "rank", 0.20, True, True, True),
    ("event_board_rank", "rank", 0.20, True, True, False),
]
DEFAULT_TEMPLATE = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/predict multi-source fusion submission.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data", default="./data/stock_data")
    parser.add_argument("--target-universe", default="./data/hs300_stock_list.csv")
    parser.add_argument("--model-output", default="./model/current_submission")
    parser.add_argument("--artifact", default="")
    parser.add_argument("--output", default="./output/result.csv")
    parser.add_argument("--scored-output", default="./model/current_submission/scored_top50.csv")
    parser.add_argument("--extra-hot-data", default="./model/extra_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_data")
    parser.add_argument(
        "--weight-template",
        default="0.20,0.20,0.20,0.20,0.20",
        help="Comma-separated top-5 weights. Default follows the best recent no-event rank ablation.",
    )
    parser.add_argument("--max-per-industry", type=int, default=2)
    parser.add_argument(
        "--weight-policy",
        choices=["dynamic", "template"],
        default="dynamic",
        help="dynamic uses confidence gap and decay risk; template uses --weight-template.",
    )
    parser.add_argument("--risk-penalty-weight", type=float, default=0.005)
    parser.add_argument("--stage1-candidates", type=int, default=50)
    parser.add_argument("--stage1-recall-weight", type=float, default=0.18)
    parser.add_argument("--disable-stage1-recall", action="store_true")
    parser.add_argument("--failure-weight", type=float, default=0.10)
    parser.add_argument("--disable-failure-model", action="store_true")
    parser.add_argument("--continuation-weight", type=float, default=0.05)
    parser.add_argument("--disable-continuation-model", action="store_true")
    parser.add_argument("--soft-failure-factor", type=float, default=0.98)
    parser.add_argument(
        "--industry-relative",
        action="store_true",
        help="Enable _ind_z/_ind_rank features. Default keeps industry lifecycle only in risk layer.",
    )
    return parser.parse_args()


FAILURE_RISK_COLS = [
    "ret_2",
    "ret_5",
    "ret_10",
    "ret_20",
    "near_high_20",
    "new_high_fail_count_5",
    "amount_rel_20",
    "volume_price_divergence",
    "volume_price_divergence_count_5",
    "upper_shadow_3",
    "close_strength",
    "close_strength_chg_3",
    "below_ma5",
    "below_ma10",
    "up_streak",
    "bull_candle_streak",
    "high_volume_down",
    "high_volume_down_count_5",
    "industry_ret_2",
    "industry_return_5",
    "industry_return_10",
    "industry_up_ratio",
    "industry_strong_ratio",
    "industry_strength_change_5",
    "industry_diffusion_change_5",
    "industry_up_ratio_change_3",
    "hot_score_chg_3",
    "strength_decay_score",
    "runup_reversal_score",
    "event_proxy_risk",
    "theme_lifecycle_risk",
    "market_regime_score",
    "total_decay_risk",
]


def _parse_template(value: str) -> np.ndarray:
    template = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=float)
    if len(template) != 5:
        raise ValueError("--weight-template must contain exactly 5 weights")
    total = template.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("--weight-template must sum to a positive finite value")
    return template / total


def _add_label_known_date(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values([ID_COL, DATE_COL]).copy()
    work["label_known_date"] = work.groupby(ID_COL, sort=False)[DATE_COL].shift(-horizon)
    return work


def _latest_candidates(featured: pd.DataFrame, feature_cols: list[str], cfg: FeatureConfig) -> pd.DataFrame:
    history_count = featured.groupby(ID_COL).cumcount() + 1
    latest_date = featured[DATE_COL].max()
    day = featured[(featured[DATE_COL] == latest_date) & (history_count >= cfg.min_history)].copy()
    day[feature_cols] = day[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(day) < 5:
        raise ValueError(f"Fewer than 5 candidates on {latest_date.date()}: {len(day)}")
    return day


def _load_raw(args: argparse.Namespace, use_hot: bool, use_extra: bool) -> pd.DataFrame:
    data_path = resolve_data_path(args.data)
    data_dir = Path(data_path).parent
    full_df = enrich_with_metadata(load_price_data(data_path), data_dir)
    if use_hot:
        full_df = merge_extra_hot_data(full_df, args.extra_hot_data)
    if use_extra:
        full_df = _merge_event_factor_dir(full_df, args.extra_factor_dir)
    full_df = add_market_context(full_df)
    return filter_universe(full_df, maybe_load_universe(args.target_universe))


def _load_raw_cached(
    args: argparse.Namespace,
    cache: dict[tuple[bool, bool], pd.DataFrame],
    use_hot: bool,
    use_extra: bool,
) -> pd.DataFrame:
    key = (use_hot, use_extra)
    if key not in cache:
        cache[key] = _load_raw(args, use_hot=use_hot, use_extra=use_extra)
    return cache[key]


def _merge_event_factor_dir(df: pd.DataFrame, directory: str) -> pd.DataFrame:
    out = df
    base = Path(directory)
    if not base.exists():
        return out
    for name in [
        "limit_list_d.csv",
        "top_list.csv",
        "margin_detail.csv",
        "ths_board_stock_hot.csv",
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


def _risk_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in raw_df.sort_values([ID_COL, DATE_COL]).groupby(ID_COL, sort=False):
        close = g[CLOSE_COL].astype(float)
        high = g[HIGH_COL].astype(float)
        low = g[LOW_COL].astype(float)
        open_ = g[OPEN_COL].astype(float)
        amount = g[AMOUNT_COL].astype(float)
        ret = close.pct_change()
        high_20 = high.rolling(20, min_periods=5).max()
        amount_ma_20 = amount.rolling(20, min_periods=5).mean()
        vol_10 = ret.rolling(10, min_periods=5).std()
        vol_20 = ret.rolling(20, min_periods=5).std()
        day_range = (high - low).clip(lower=1e-12)
        m = pd.DataFrame({DATE_COL: g[DATE_COL], ID_COL: code})
        m["last_pct"] = g[PCT_COL].astype(float) if PCT_COL in g.columns else ret * 100.0
        m["last_intraday_ret"] = close / open_.clip(lower=1e-12) - 1.0
        m["last_close_pos"] = (close - low) / day_range
        m["last_upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_.clip(lower=1e-12)
        m["ret_10"] = close / close.shift(10) - 1.0
        m["ret_20"] = close / close.shift(20) - 1.0
        m["drawdown_20"] = close / high_20.clip(lower=1e-12) - 1.0
        m["amount_rel_20"] = amount / amount_ma_20.clip(lower=1e-12)
        m["vol_10"] = vol_10
        m["vol_ratio_10_20"] = vol_10 / vol_20.clip(lower=1e-12)
        rows.append(m)
    risk = pd.concat(rows, ignore_index=True)
    risk["risk_count"] = 0
    flags = [
        risk["ret_10"] >= 0.25,
        risk["ret_20"] >= 0.40,
        risk["amount_rel_20"] >= 1.8,
        risk["vol_10"] >= 0.05,
        risk["vol_ratio_10_20"] >= 1.35,
        risk["last_pct"] <= -5.0,
        risk["last_intraday_ret"] <= -0.04,
        risk["last_close_pos"] <= 0.25,
        risk["last_upper_shadow"] >= 0.04,
        (risk["drawdown_20"] <= -0.12) & (risk["ret_20"] > 0),
    ]
    for flag in flags:
        risk["risk_count"] += flag.fillna(False).astype(int)
    return risk[[DATE_COL, ID_COL, "risk_count", "ret_20", "last_close_pos"]]


def _train_one(
    raw_df: pd.DataFrame,
    profile: str,
    relevance: str,
    use_event_text: bool = False,
    add_industry_relative: bool = False,
) -> dict[str, object]:
    cfg = FeatureConfig(
        windows=(3, 5, 10, 20, 30, 60, 120),
        min_history=60,
        horizon=5,
        top_k=5,
        add_event_text=use_event_text,
        add_industry_relative=add_industry_relative,
    )
    featured, feature_cols = build_feature_frame(raw_df, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    latest_date = featured[DATE_COL].max()
    trainable = filter_trainable_rows(featured, feature_cols, cfg)
    trainable = trainable[trainable["label_known_date"] <= latest_date].copy()
    ranker_cfg = TreeRankerConfig(
        output_dir="",
        lgb_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=5,
        min_child_samples=50,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=1.2,
        relevance_mode=relevance,
    )
    print(f"training {profile}: rows={len(trainable)} features={len(feature_cols)} latest={latest_date.date()}")
    model = _fit_lightgbm(trainable, feature_cols, ranker_cfg)
    return {
        "profile": profile,
        "feature_config": asdict(cfg),
        "feature_cols": feature_cols,
        "ranker_config": asdict(ranker_cfg),
        "model": model,
    }


def _make_stage1_recall_label(df: pd.DataFrame) -> pd.Series:
    future_rank = df.groupby(DATE_COL, group_keys=False)[LABEL_COL].rank(method="first", ascending=False)
    return ((future_rank <= 20) | (df[LABEL_COL] >= 0.06)).astype(int)


def _train_stage1_recall(raw_df: pd.DataFrame, add_industry_relative: bool = False) -> dict[str, object]:
    from lightgbm import LGBMClassifier

    cfg = FeatureConfig(
        windows=(3, 5, 10, 20, 30, 60, 120),
        min_history=60,
        horizon=5,
        top_k=5,
        add_event_text=True,
        add_industry_relative=add_industry_relative,
    )
    featured, feature_cols = build_feature_frame(raw_df, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    latest_date = featured[DATE_COL].max()
    trainable = filter_trainable_rows(featured, feature_cols, cfg)
    trainable = trainable[trainable["label_known_date"] <= latest_date].copy()
    trainable["stage1_target"] = _make_stage1_recall_label(trainable)
    pos_rate = float(trainable["stage1_target"].mean())
    scale_pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=45,
        subsample=0.82,
        colsample_bytree=0.72,
        reg_alpha=0.20,
        reg_lambda=1.5,
        random_state=2026,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        importance_type="gain",
        verbose=-1,
    )
    print(
        f"training stage1_recall: rows={len(trainable)} features={len(feature_cols)} "
        f"pos_rate={pos_rate:.4f} latest={latest_date.date()}"
    )
    model.fit(trainable[feature_cols], trainable["stage1_target"])
    return {
        "profile": "stage1_recall",
        "feature_config": asdict(cfg),
        "feature_cols": feature_cols,
        "model": model,
        "positive_definition": "future_rank<=20 or label_return>=0.06",
        "positive_rate": pos_rate,
    }


def _merge_failure_features(df: pd.DataFrame, risk: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in FAILURE_RISK_COLS if c in risk.columns]
    work = df.merge(risk[[DATE_COL, ID_COL, *available]], on=[DATE_COL, ID_COL], how="left")
    work[available] = work[available].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return work, available


def _train_failure_model(raw_df: pd.DataFrame, add_industry_relative: bool = False) -> dict[str, object] | None:
    from lightgbm import LGBMClassifier

    cfg = FeatureConfig(
        windows=(3, 5, 10, 20, 30, 60, 120),
        min_history=60,
        horizon=5,
        top_k=5,
        add_event_text=True,
        add_industry_relative=add_industry_relative,
    )
    featured, feature_cols = build_feature_frame(raw_df, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    latest_date = featured[DATE_COL].max()
    trainable = filter_trainable_rows(featured, feature_cols, cfg)
    trainable = trainable[trainable["label_known_date"] <= latest_date].copy()
    risk = build_decay_risk_frame(raw_df)
    work, risk_cols = _merge_failure_features(trainable, risk)
    work = work.dropna(subset=[LABEL_COL]).copy()
    if len(work) < 500 or not risk_cols:
        print("skip failure_model: insufficient training rows or risk columns")
        return None

    grouped = work.groupby(DATE_COL, sort=False)
    ret5_rank = grouped["ret_5"].rank(pct=True, method="first")
    ret10_rank = grouped["ret_10"].rank(pct=True, method="first")
    ret20_rank = grouped["ret_20"].rank(pct=True, method="first")
    amount_rank = grouped["amount_rel_20"].rank(pct=True, method="first")
    hot_fading = work.get("hot_score_chg_3", pd.Series(0.0, index=work.index)) < -0.06
    past_strong = (
        (ret5_rank >= 0.72)
        | (ret10_rank >= 0.72)
        | (ret20_rank >= 0.76)
        | (amount_rank >= 0.80)
        | (work["total_decay_risk"] >= 0.35)
        | hot_fading
    )
    market_return = grouped[LABEL_COL].transform("mean")
    future_rank = grouped[LABEL_COL].rank(method="first", ascending=False)
    work = work[past_strong.fillna(False)].copy()
    if len(work) < 300:
        print("skip failure_model: insufficient strong-history rows")
        return None
    work["failure_target"] = (
        (work[LABEL_COL] < market_return.loc[work.index])
        | (work[LABEL_COL] < 0.0)
        | ((future_rank.loc[work.index] > 60) & (work[LABEL_COL] < 0.03))
    ).astype(int)
    if work["failure_target"].nunique() < 2:
        print("skip failure_model: target has one class")
        return None

    model_features = list(dict.fromkeys(feature_cols + risk_cols))
    pos_rate = float(work["failure_target"].mean())
    scale_pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=4,
        min_child_samples=40,
        subsample=0.82,
        colsample_bytree=0.70,
        reg_alpha=0.30,
        reg_lambda=2.0,
        random_state=2026,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        importance_type="gain",
        verbose=-1,
    )
    print(
        f"training failure_model: rows={len(work)} features={len(model_features)} "
        f"pos_rate={pos_rate:.4f} latest={latest_date.date()}"
    )
    model.fit(work[model_features], work["failure_target"])
    return {
        "profile": "failure_model",
        "feature_config": asdict(cfg),
        "feature_cols": feature_cols,
        "risk_cols": risk_cols,
        "model_features": model_features,
        "model": model,
        "positive_definition": "past strong/active and future below market, negative, or weak rank",
        "positive_rate": pos_rate,
    }


def _past_strong_mask(work: pd.DataFrame) -> pd.Series:
    grouped = work.groupby(DATE_COL, sort=False)
    ret5_rank = grouped["ret_5"].rank(pct=True, method="first")
    ret10_rank = grouped["ret_10"].rank(pct=True, method="first")
    ret20_rank = grouped["ret_20"].rank(pct=True, method="first")
    amount_rank = grouped["amount_rel_20"].rank(pct=True, method="first")
    hot_fading = work.get("hot_score_chg_3", pd.Series(0.0, index=work.index)) < -0.06
    return (
        (ret5_rank >= 0.72)
        | (ret10_rank >= 0.72)
        | (ret20_rank >= 0.76)
        | (amount_rank >= 0.80)
        | (work["total_decay_risk"] >= 0.35)
        | hot_fading
    ).fillna(False)


def _train_continuation_model(raw_df: pd.DataFrame, add_industry_relative: bool = False) -> dict[str, object] | None:
    from lightgbm import LGBMClassifier

    cfg = FeatureConfig(
        windows=(3, 5, 10, 20, 30, 60, 120),
        min_history=60,
        horizon=5,
        top_k=5,
        add_event_text=True,
        add_industry_relative=add_industry_relative,
    )
    featured, feature_cols = build_feature_frame(raw_df, cfg=cfg, with_labels=True)
    featured = _add_label_known_date(featured, cfg.horizon)
    latest_date = featured[DATE_COL].max()
    trainable = filter_trainable_rows(featured, feature_cols, cfg)
    trainable = trainable[trainable["label_known_date"] <= latest_date].copy()
    risk = build_decay_risk_frame(raw_df)
    work, risk_cols = _merge_failure_features(trainable, risk)
    work = work.dropna(subset=[LABEL_COL]).copy()
    if len(work) < 500 or not risk_cols:
        print("skip continuation_model: insufficient training rows or risk columns")
        return None
    work = work[_past_strong_mask(work)].copy()
    if len(work) < 300:
        print("skip continuation_model: insufficient strong-history rows")
        return None
    future_rank = work.groupby(DATE_COL, sort=False)[LABEL_COL].rank(method="first", ascending=False)
    work["continuation_target"] = ((future_rank <= 10) | (work[LABEL_COL] >= 0.06)).astype(int)
    if work["continuation_target"].nunique() < 2:
        print("skip continuation_model: target has one class")
        return None
    model_features = list(dict.fromkeys(feature_cols + risk_cols))
    pos_rate = float(work["continuation_target"].mean())
    scale_pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=4,
        min_child_samples=40,
        subsample=0.82,
        colsample_bytree=0.70,
        reg_alpha=0.25,
        reg_lambda=1.8,
        random_state=2027,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        importance_type="gain",
        verbose=-1,
    )
    print(
        f"training continuation_model: rows={len(work)} features={len(model_features)} "
        f"pos_rate={pos_rate:.4f} latest={latest_date.date()}"
    )
    model.fit(work[model_features], work["continuation_target"])
    return {
        "profile": "continuation_model",
        "feature_config": asdict(cfg),
        "feature_cols": feature_cols,
        "risk_cols": risk_cols,
        "model_features": model_features,
        "model": model,
        "positive_definition": "past strong/active and future rank<=10 or return>=6%",
        "positive_rate": pos_rate,
    }


def train(args: argparse.Namespace) -> None:
    artifacts = []
    raw_cache: dict[tuple[bool, bool], pd.DataFrame] = {}
    for name, relevance, weight, use_hot, use_extra, use_event_text in PROFILES:
        raw = _load_raw_cached(args, raw_cache, use_hot=use_hot, use_extra=use_extra)
        artifacts.append(
            {
                **_train_one(
                    raw,
                    name,
                    relevance,
                    use_event_text=use_event_text,
                    add_industry_relative=bool(args.industry_relative),
                ),
                "weight": weight,
                "use_hot": use_hot,
                "use_extra": use_extra,
                "use_event_text": use_event_text,
            }
        )
    stage1_recall = None
    if not args.disable_stage1_recall:
        recall_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        stage1_recall = _train_stage1_recall(recall_raw, add_industry_relative=bool(args.industry_relative))
    continuation_model = None
    if not args.disable_continuation_model:
        continuation_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        continuation_model = _train_continuation_model(
            continuation_raw,
            add_industry_relative=bool(args.industry_relative),
        )
    failure_model = None
    if not args.disable_failure_model:
        failure_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        failure_model = _train_failure_model(failure_raw, add_industry_relative=bool(args.industry_relative))
    out_dir = Path(args.model_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "fusion_submission.joblib"
    joblib.dump(
        {
            "profiles": artifacts,
            "template": _parse_template(args.weight_template).tolist(),
            "risk_filter": "close_strong",
            "max_per_industry": args.max_per_industry,
            "stage1_recall": stage1_recall,
            "stage1_recall_weight": float(args.stage1_recall_weight),
            "continuation_model": continuation_model,
            "continuation_weight": float(args.continuation_weight),
            "failure_model": failure_model,
            "failure_weight": float(args.failure_weight),
            "industry_relative": bool(args.industry_relative),
        },
        artifact_path,
    )
    predict(args, artifact_path=str(artifact_path))


def _score_profile(raw_df: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = list(artifact["feature_cols"])
    featured, _ = build_feature_frame(raw_df, cfg=cfg, with_labels=False)
    day = _latest_candidates(featured, feature_cols, cfg)
    raw_score = np.asarray(artifact["model"].predict(day[feature_cols]), dtype=float)
    return day[[DATE_COL, ID_COL]].assign(score=raw_score)


def _score_stage1_recall(raw_df: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = list(artifact["feature_cols"])
    featured, _ = build_feature_frame(raw_df, cfg=cfg, with_labels=False)
    day = _latest_candidates(featured, feature_cols, cfg)
    proba = np.asarray(artifact["model"].predict_proba(day[feature_cols])[:, 1], dtype=float)
    return day[[DATE_COL, ID_COL]].assign(stage1_recall_score=proba)


def _score_failure_model(raw_df: pd.DataFrame, artifact: dict[str, object], risk: pd.DataFrame) -> pd.DataFrame:
    cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = list(artifact["feature_cols"])
    model_features = list(artifact["model_features"])
    featured, _ = build_feature_frame(raw_df, cfg=cfg, with_labels=False)
    day = _latest_candidates(featured, feature_cols, cfg)
    work, _ = _merge_failure_features(day, risk)
    work[model_features] = work[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = np.asarray(artifact["model"].predict_proba(work[model_features])[:, 1], dtype=float)
    return work[[DATE_COL, ID_COL]].assign(failure_score=proba)


def _score_continuation_model(raw_df: pd.DataFrame, artifact: dict[str, object], risk: pd.DataFrame) -> pd.DataFrame:
    cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = list(artifact["feature_cols"])
    model_features = list(artifact["model_features"])
    featured, _ = build_feature_frame(raw_df, cfg=cfg, with_labels=False)
    day = _latest_candidates(featured, feature_cols, cfg)
    work, _ = _merge_failure_features(day, risk)
    work[model_features] = work[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = np.asarray(artifact["model"].predict_proba(work[model_features])[:, 1], dtype=float)
    return work[[DATE_COL, ID_COL]].assign(continuation_score=proba)


def _select_top(day: pd.DataFrame, top_k: int, max_per_industry: int) -> pd.DataFrame:
    candidates = day.sort_values("score", ascending=False).copy()
    if max_per_industry <= 0 or "行业" not in candidates.columns:
        return candidates.head(top_k).copy()

    counts: dict[str, int] = {}
    keep = []
    for idx, row in candidates.iterrows():
        industry = str(row.get("行业", "UNK"))
        if counts.get(industry, 0) >= max_per_industry:
            continue
        keep.append(idx)
        counts[industry] = counts.get(industry, 0) + 1
        if len(keep) >= top_k:
            break
    selected = candidates.loc[keep].copy()
    if len(selected) < top_k:
        selected = pd.concat([selected, candidates.drop(index=selected.index).head(top_k - len(selected))], axis=0)
    return selected


def _stage1_candidates(day: pd.DataFrame, candidate_count: int) -> pd.DataFrame:
    if candidate_count <= 0 or len(day) <= candidate_count:
        return day.copy()
    keep: set[object] = set(day.sort_values("model_score", ascending=False).head(candidate_count)[ID_COL])
    if "stage1_recall_score" in day.columns:
        keep.update(day.sort_values("stage1_recall_score", ascending=False).head(candidate_count)[ID_COL])
    return day[day[ID_COL].isin(keep)].copy()


def _apply_soft_failure_filter(day: pd.DataFrame, factor: float = 0.98) -> pd.DataFrame:
    work = day.copy()
    flags = pd.Series(False, index=work.index)
    if "up_streak" in work.columns:
        flags |= pd.to_numeric(work["up_streak"], errors="coerce").fillna(0.0) >= 5
    if "bull_candle_streak" in work.columns:
        flags |= pd.to_numeric(work["bull_candle_streak"], errors="coerce").fillna(0.0) >= 5
    if "volume_price_divergence_count_5" in work.columns:
        flags |= pd.to_numeric(work["volume_price_divergence_count_5"], errors="coerce").fillna(0.0) >= 2
    if "hot_score_chg_3" in work.columns:
        flags |= pd.to_numeric(work["hot_score_chg_3"], errors="coerce").fillna(0.0) <= -0.10
    if "ret_5" in work.columns:
        ret5_rank = work.groupby(DATE_COL)["ret_5"].rank(pct=True, method="first")
        weak_close = pd.to_numeric(work.get("close_strength_chg_3", 0.0), errors="coerce").fillna(0.0) <= -0.20
        flags |= (ret5_rank >= 0.93) & weak_close
    clipped = float(np.clip(factor, 0.95, 1.0))
    work["soft_failure_filter"] = flags.astype(int)
    work.loc[flags, "score"] *= clipped
    return work


def predict(args: argparse.Namespace, artifact_path: str | None = None) -> None:
    artifact_path = artifact_path or args.artifact or str(Path(args.model_output) / "fusion_submission.joblib")
    bundle = joblib.load(artifact_path)
    parts = []
    base_raw = None
    raw_cache: dict[tuple[bool, bool], pd.DataFrame] = {}
    for item in bundle["profiles"]:
        raw = _load_raw_cached(args, raw_cache, use_hot=bool(item["use_hot"]), use_extra=bool(item["use_extra"]))
        if base_raw is None:
            base_raw = raw
        score = _score_profile(raw, item).rename(columns={"score": f"score_{item['profile']}"})
        parts.append((score, float(item["weight"]), f"score_{item['profile']}"))
    merged = parts[0][0]
    for part, _, _ in parts[1:]:
        merged = merged.merge(part, on=[DATE_COL, ID_COL], how="inner")
    merged["score"] = 0.0
    for _, weight, col in parts:
        merged["score"] += weight * merged[col].rank(pct=True, method="first")
    merged["model_score"] = merged["score"]
    stage1_artifact = bundle.get("stage1_recall")
    if stage1_artifact is not None:
        recall_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        recall_score = _score_stage1_recall(recall_raw, stage1_artifact)
        merged = merged.merge(recall_score, on=[DATE_COL, ID_COL], how="left")
        recall_weight = float(getattr(args, "stage1_recall_weight", bundle.get("stage1_recall_weight", 0.18)))
    else:
        merged["stage1_recall_score"] = 0.0
        recall_weight = 0.0
    recall_weight = float(np.clip(recall_weight, 0.0, 0.6))
    model_rank = merged.groupby(DATE_COL)["model_score"].rank(pct=True, method="first")
    recall_rank = merged.groupby(DATE_COL)["stage1_recall_score"].rank(pct=True, method="first")
    merged["alpha_score"] = (1.0 - recall_weight) * model_rank + recall_weight * recall_rank
    assert base_raw is not None
    latest_date = merged[DATE_COL].max()
    risk_raw = raw_cache.get((True, True), base_raw)
    risk = build_decay_risk_frame(risk_raw)
    continuation_artifact = bundle.get("continuation_model")
    if continuation_artifact is not None:
        continuation_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        continuation_score = _score_continuation_model(continuation_raw, continuation_artifact, risk)
        merged = merged.merge(continuation_score, on=[DATE_COL, ID_COL], how="left")
        merged["continuation_score"] = merged["continuation_score"].fillna(0.0)
        base_rank = merged.groupby(DATE_COL)["alpha_score"].rank(pct=True, method="first")
        continuation_rank = merged.groupby(DATE_COL)["continuation_score"].rank(pct=True, method="first")
        continuation_weight = float(getattr(args, "continuation_weight", bundle.get("continuation_weight", 0.16)))
        merged["alpha_score_before_continuation"] = merged["alpha_score"]
        merged["alpha_score"] = base_rank + float(np.clip(continuation_weight, 0.0, 0.5)) * continuation_rank
    else:
        merged["continuation_score"] = 0.0
    merged = apply_decay_risk_adjustment(
        merged,
        risk,
        score_col="alpha_score",
        output_col="score",
        penalty_weight=float(getattr(args, "risk_penalty_weight", 0.22)),
    )
    failure_artifact = bundle.get("failure_model")
    if failure_artifact is not None:
        failure_raw = _load_raw_cached(args, raw_cache, use_hot=True, use_extra=True)
        failure_score = _score_failure_model(failure_raw, failure_artifact, risk)
        merged = merged.merge(failure_score, on=[DATE_COL, ID_COL], how="left")
        merged["failure_score"] = merged["failure_score"].fillna(0.0)
        base_rank = merged.groupby(DATE_COL)["score"].rank(pct=True, method="first")
        failure_rank = merged.groupby(DATE_COL)["failure_score"].rank(pct=True, method="first")
        failure_weight = float(getattr(args, "failure_weight", bundle.get("failure_weight", 0.14)))
        merged["score_before_failure"] = merged["score"]
        merged["score"] = base_rank - float(np.clip(failure_weight, 0.0, 0.5)) * failure_rank
    else:
        merged["failure_score"] = 0.0
    merged = _apply_soft_failure_filter(
        merged,
        factor=float(getattr(args, "soft_failure_factor", 0.98)),
    )
    day = merged[merged[DATE_COL] == latest_date].copy()
    day["stage1_rank"] = day["model_score"].rank(method="first", ascending=False)
    day["final_rank"] = day["score"].rank(method="first", ascending=False)
    risk_filter = bundle.get("risk_filter", "close_strong")
    if risk_filter == "close_strong":
        filtered = day[
            (day["close_strength"] > 0.25)
            & (day["ret_20"] > -0.25)
            & (day["total_decay_risk"] < 0.78)
        ].copy()
    else:
        filtered = day[(day["total_decay_risk"] < 0.65) & (day["ret_20"] > -0.25)].copy()
    if len(filtered) < 5:
        filtered = day
    stage1_n = int(getattr(args, "stage1_candidates", 50))
    filtered = _stage1_candidates(filtered, stage1_n)
    stage1_ids = set(filtered[ID_COL])
    filtered["stage1_selected"] = 1
    day["stage1_selected"] = day[ID_COL].isin(stage1_ids).astype(int)
    if "行业" not in filtered.columns and "行业" in base_raw.columns:
        industry = base_raw[base_raw[DATE_COL] == latest_date][[DATE_COL, ID_COL, "行业"]].drop_duplicates([DATE_COL, ID_COL])
        filtered = filtered.merge(industry, on=[DATE_COL, ID_COL], how="left")
        day = day.merge(industry, on=[DATE_COL, ID_COL], how="left")
    max_per_industry = int(bundle.get("max_per_industry", getattr(args, "max_per_industry", 2)))
    top = _select_top(filtered, top_k=5, max_per_industry=max_per_industry).copy()
    if getattr(args, "weight_policy", "dynamic") == "dynamic":
        template = confidence_weight_template(top)
    else:
        template = np.array(bundle.get("template", DEFAULT_TEMPLATE.tolist()), dtype=float)
    top["weight"] = template / template.sum()
    out = top.rename(columns={ID_COL: "stock_id"})[["stock_id", "weight"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, float_format="%.6f")
    Path(args.scored_output).parent.mkdir(parents=True, exist_ok=True)
    day.sort_values("score", ascending=False).head(50).to_csv(args.scored_output, index=False)
    print(out.to_string(index=False))
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "predict":
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
