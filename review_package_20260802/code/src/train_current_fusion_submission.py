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
    parser.add_argument("--risk-penalty-weight", type=float, default=0.02)
    return parser.parse_args()


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


def _train_one(raw_df: pd.DataFrame, profile: str, relevance: str, use_event_text: bool = False) -> dict[str, object]:
    cfg = FeatureConfig(
        windows=(3, 5, 10, 20, 30, 60, 120),
        min_history=60,
        horizon=5,
        top_k=5,
        add_event_text=use_event_text,
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


def train(args: argparse.Namespace) -> None:
    artifacts = []
    raw_cache: dict[tuple[bool, bool], pd.DataFrame] = {}
    for name, relevance, weight, use_hot, use_extra, use_event_text in PROFILES:
        raw = _load_raw_cached(args, raw_cache, use_hot=use_hot, use_extra=use_extra)
        artifacts.append(
            {
                **_train_one(raw, name, relevance, use_event_text=use_event_text),
                "weight": weight,
                "use_hot": use_hot,
                "use_extra": use_extra,
                "use_event_text": use_event_text,
            }
        )
    out_dir = Path(args.model_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "fusion_submission.joblib"
    joblib.dump(
        {
            "profiles": artifacts,
            "template": _parse_template(args.weight_template).tolist(),
            "risk_filter": "close_strong",
            "max_per_industry": args.max_per_industry,
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
    assert base_raw is not None
    latest_date = merged[DATE_COL].max()
    risk_raw = raw_cache.get((True, True), base_raw)
    risk = build_decay_risk_frame(risk_raw)
    merged = apply_decay_risk_adjustment(
        merged,
        risk,
        score_col="model_score",
        output_col="score",
        penalty_weight=float(getattr(args, "risk_penalty_weight", 0.22)),
    )
    day = merged[merged[DATE_COL] == latest_date].copy()
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
