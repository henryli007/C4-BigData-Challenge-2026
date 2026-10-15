from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

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
    normalize_stock_code,
)
from advanced.metrics import select_portfolio
from advanced.tree_ranker import TreeRankerConfig, _fit_lightgbm, score_with_models


def resolve_data_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    if p.suffix == "" and p.with_suffix(".csv").exists():
        return str(p.with_suffix(".csv"))
    raise FileNotFoundError(f"data file not found: {path}")


def maybe_load_universe(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return load_universe_codes(p)
    return None


def merge_extra_hot_data(df: pd.DataFrame, path: str) -> pd.DataFrame:
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
    before = len(df)
    out = df.merge(hot, on=[DATE_COL, ID_COL], how="left")
    print(f"merged ths_hot rows={len(hot)} into price rows={before}; hot_matches={out['ths_hot_rank'].notna().sum() if 'ths_hot_rank' in out.columns else 0}")
    return out


def merge_extra_factor_dir(df: pd.DataFrame, directory: str) -> pd.DataFrame:
    out = df
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train current fast LightGBM model and write submission.")
    parser.add_argument("--data", default="./data/stock_data")
    parser.add_argument("--target-universe", default="")
    parser.add_argument("--model-output", default="./model/current_submission")
    parser.add_argument("--output", default="./output/result.csv")
    parser.add_argument("--scored-output", default="./output/current_lgb_fullrows_scored_20260626.csv")
    parser.add_argument("--quality-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.0015)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--train-lookback-days", type=int, default=0)
    parser.add_argument("--windows", default="3,5,10,20,30,60,120")
    parser.add_argument("--relevance-mode", choices=["rank", "burst", "recall", "ensemble"], default="ensemble")
    parser.add_argument(
        "--blend-weights",
        default="0.8,0.2",
        help="Weights for rank,burst models when --relevance-mode ensemble.",
    )
    parser.add_argument(
        "--fixed-template",
        default="",
        help="Optional comma-separated top-5 weights assigned by model rank, e.g. 0.6,0.2,0.15,0.049999,0.000001.",
    )
    parser.add_argument("--extra-hot-data", default="./model/extra_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_data")
    return parser.parse_args()


def _add_label_known_date(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.sort_values([ID_COL, DATE_COL]).copy()
    work["label_known_date"] = work.groupby(ID_COL, sort=False)[DATE_COL].shift(-horizon)
    return work


def _latest_candidates(featured: pd.DataFrame, feature_cols: list[str], cfg: FeatureConfig) -> pd.DataFrame:
    history_count = featured.groupby(ID_COL).cumcount() + 1
    latest_date = featured[DATE_COL].max()
    day = featured[(featured[DATE_COL] == latest_date) & (history_count >= cfg.min_history)].copy()
    if len(day) < 5:
        raise ValueError(f"Fewer than 5 candidates on {latest_date.date()}: {len(day)}")
    day[feature_cols] = day[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return day


def _add_quality_score(scored: pd.DataFrame, day: pd.DataFrame, quality_weight: float) -> pd.DataFrame:
    extra = day[[DATE_COL, ID_COL, "f_close_ret_10", "f_close_ret_20", "f_amount_rel_5", "f_ret_std_10"]].copy()
    work = scored.merge(extra, on=[DATE_COL, ID_COL], how="left")
    work["model_pct"] = work["score"].rank(pct=True, method="first")
    quality_raw = (
        work["f_close_ret_10"].rank(pct=True, method="first")
        + 0.3 * work["f_close_ret_20"].rank(pct=True, method="first")
        + 0.2 * work["f_amount_rel_5"].rank(pct=True, method="first")
        - 0.2 * work["f_ret_std_10"].rank(pct=True, method="first")
    )
    work["quality_pct"] = quality_raw.rank(pct=True, method="first")
    w = float(np.clip(quality_weight, 0.0, 1.0))
    work["score_model_only"] = work["score"]
    work["score"] = (1.0 - w) * work["model_pct"] + w * work["quality_pct"]
    return work


def _score_ensemble(day: pd.DataFrame, feature_cols: list[str], models: dict[str, object], weights: list[float]) -> pd.DataFrame:
    keep_cols = [DATE_COL, ID_COL]
    for col in ["risk_vol_20", "risk_vol_60", "行业"]:
        if col in day.columns:
            keep_cols.append(col)
    scored = day[keep_cols].copy()
    total = np.zeros(len(day), dtype=float)
    norm = np.array(weights, dtype=float)
    norm = norm / norm.sum()
    for i, (name, model) in enumerate(models.items()):
        raw = np.asarray(model.predict(day[feature_cols]), dtype=float)
        pct = pd.Series(raw).rank(pct=True, method="first").to_numpy(dtype=float)
        scored[f"score_{name}"] = raw
        total += norm[i] * pct
    scored["score"] = total
    return scored


def _select_with_fixed_template(scored: pd.DataFrame, weights: str, max_per_industry: int) -> pd.DataFrame:
    template = np.array([float(x.strip()) for x in weights.split(",") if x.strip()], dtype=float)
    if len(template) != 5:
        raise ValueError("--fixed-template must contain exactly 5 weights")
    candidates = scored.sort_values("score", ascending=False).copy()
    if max_per_industry > 0 and "行业" in candidates.columns:
        counts: dict[str, int] = {}
        keep = []
        for idx, row in candidates.iterrows():
            industry = str(row.get("行业", "UNK"))
            if counts.get(industry, 0) >= max_per_industry:
                continue
            keep.append(idx)
            counts[industry] = counts.get(industry, 0) + 1
            if len(keep) == 5:
                break
        ranked = candidates.loc[keep].copy()
        if len(ranked) < 5:
            ranked = pd.concat([ranked, candidates.drop(index=ranked.index).head(5 - len(ranked))], axis=0)
    else:
        ranked = candidates.head(5).copy()
    ranked["weight"] = template / template.sum()
    return ranked[[ID_COL, "score", "weight"]]


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    data_dir = Path(data_path).parent
    full_df = enrich_with_metadata(load_price_data(data_path), data_dir)
    full_df = merge_extra_hot_data(full_df, args.extra_hot_data)
    full_df = merge_extra_factor_dir(full_df, args.extra_factor_dir)
    full_df = add_market_context(full_df)
    raw_df = filter_universe(full_df, maybe_load_universe(args.target_universe))

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    min_history = min(60, max(windows)) if windows else 60
    feature_cfg = FeatureConfig(windows=windows, min_history=min_history, horizon=5, top_k=5)
    base_ranker_cfg = TreeRankerConfig(
        output_dir=args.model_output,
        lgb_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=5,
        min_child_samples=50,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=1.2,
        max_rows_per_fold=0,
        relevance_mode="rank" if args.relevance_mode == "ensemble" else args.relevance_mode,
    )

    featured, feature_cols = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=True)
    featured = _add_label_known_date(featured, feature_cfg.horizon)
    latest_date = featured[DATE_COL].max()
    trainable = filter_trainable_rows(featured, feature_cols, feature_cfg)
    trainable = trainable[trainable["label_known_date"] <= latest_date].copy()
    if args.train_lookback_days > 0:
        trainable = trainable[trainable[DATE_COL] >= latest_date - pd.Timedelta(days=args.train_lookback_days)].copy()
    day = _latest_candidates(featured, feature_cols, feature_cfg)

    print(
        f"latest_date={latest_date.date()} train_rows={len(trainable)} "
        f"last_label_date={trainable[DATE_COL].max().date()} candidates={len(day)}"
    )
    if args.relevance_mode == "ensemble":
        weights = [float(x.strip()) for x in args.blend_weights.split(",") if x.strip()]
        if len(weights) != 2:
            raise ValueError("--blend-weights must contain 2 values for rank,burst")
        rank_cfg = base_ranker_cfg
        burst_cfg = TreeRankerConfig(**{**asdict(base_ranker_cfg), "relevance_mode": "burst"})
        rank_model = _fit_lightgbm(trainable, feature_cols, rank_cfg)
        burst_model = _fit_lightgbm(trainable, feature_cols, burst_cfg)
        models = {"rank_lightgbm": rank_model, "burst_lightgbm": burst_model}
        scored = _score_ensemble(day, feature_cols, models, weights)
        ranker_cfg = rank_cfg
    else:
        model = _fit_lightgbm(trainable, feature_cols, base_ranker_cfg)
        models = {"lightgbm": model}
        weights = []
        scored = score_with_models(day, feature_cols, models)
        ranker_cfg = base_ranker_cfg
    scored = _add_quality_score(scored, day, args.quality_weight)
    if "行业" in day.columns and "行业" not in scored.columns:
        scored = scored.merge(day[[DATE_COL, ID_COL, "行业"]], on=[DATE_COL, ID_COL], how="left")

    if args.fixed_template:
        portfolio = _select_with_fixed_template(scored, args.fixed_template, args.max_per_industry)
    else:
        portfolio = select_portfolio(
            scored,
            top_k=5,
            policy="risk_adjusted",
            temperature=args.temperature,
            max_per_industry=args.max_per_industry,
        )
    out = portfolio.rename(columns={ID_COL: "stock_id"})[["stock_id", "weight"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["weight"] = out["weight"] / out["weight"].sum()

    model_dir = Path(args.model_output)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_config": asdict(feature_cfg),
            "ranker_config": asdict(ranker_cfg),
            "feature_cols": feature_cols,
            "models": models,
            "blend_model_weights": weights,
            "policy": {
                "quality_weight": args.quality_weight,
                "policy": "risk_adjusted",
                "temperature": args.temperature,
                "max_per_industry": args.max_per_industry,
                "fixed_template": args.fixed_template,
            },
            "latest_date": str(latest_date.date()),
            "train_rows": len(trainable),
        },
        model_dir / "tree_ranker.joblib",
    )
    scored.sort_values("score", ascending=False).head(50).to_csv(args.scored_output, index=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
