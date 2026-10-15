from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from advanced.features import (
    DATE_COL,
    FeatureConfig,
    ID_COL,
    add_market_context,
    build_feature_frame,
    enrich_with_metadata,
    filter_universe,
    load_price_data,
    load_universe_codes,
    normalize_stock_code,
)
from advanced.metrics import select_portfolio
from advanced.portfolio_model import PortfolioScorer
from advanced.tree_ranker import score_with_models


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
    return df.merge(hot, on=[DATE_COL, ID_COL], how="left")


def merge_extra_factor_dir(df: pd.DataFrame, directory: str) -> pd.DataFrame:
    out = df
    base = Path(directory)
    if not base.exists():
        return out
    for name in ["limit_list_d.csv", "top_list.csv", "margin_detail.csv", "ths_board_stock_hot.csv", "stk_factor.csv"]:
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
        out = out.merge(extra[[DATE_COL, ID_COL, *value_cols]], on=[DATE_COL, ID_COL], how="left")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-5 stock portfolio.")
    parser.add_argument("--data", default="./data/stock_data")
    parser.add_argument("--target-universe", default="")
    parser.add_argument("--ranker-artifact", default="./model/current_submission/tree_ranker.joblib")
    parser.add_argument("--portfolio-artifact", default="./model/portfolio_direct/portfolio_artifact.joblib")
    parser.add_argument("--portfolio-model", default="./model/portfolio_direct/portfolio_model.pth")
    parser.add_argument("--output", default="./output/result.csv")
    parser.add_argument("--method", choices=["ranker", "portfolio", "blend"], default="ranker")
    parser.add_argument("--blend-alpha", type=float, default=0.6, help="Ranker weight in blended score.")
    parser.add_argument("--max-weight", type=float, default=0.45, help="Cap single-stock weight for portfolio/blend outputs.")
    parser.add_argument("--max-per-industry", type=int, default=2, help="Limit selected stocks per industry when industry metadata exists.")
    parser.add_argument("--extra-hot-data", default="./model/extra_data/ths_hot_stock.csv")
    parser.add_argument("--extra-factor-dir", default="./model/extra_data")
    return parser.parse_args()


def _latest_candidates(raw_df: pd.DataFrame, feature_cfg: FeatureConfig, feature_cols: list[str]) -> pd.DataFrame:
    featured, _ = build_feature_frame(raw_df, cfg=feature_cfg, with_labels=False)
    history_count = featured.groupby(ID_COL).cumcount() + 1
    latest_date = featured[DATE_COL].max()
    day = featured[(featured[DATE_COL] == latest_date) & (history_count >= feature_cfg.min_history)].copy()
    if len(day) < 5:
        raise ValueError(f"Fewer than 5 valid candidates on {latest_date.date()}: {len(day)}")
    for col in feature_cols:
        if col not in day.columns:
            day[col] = 0.0
    day[feature_cols] = day[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return day


def score_ranker_candidates(raw_df: pd.DataFrame, artifact_path: str) -> tuple[pd.DataFrame, dict]:
    artifact = joblib.load(artifact_path)
    feature_cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = artifact["feature_cols"]
    day = _latest_candidates(raw_df, feature_cfg, feature_cols)
    models = artifact["models"]
    blend_weights = artifact.get("blend_model_weights", [])
    if blend_weights:
        keep_cols = [DATE_COL, ID_COL]
        for col in ["risk_vol_20", "risk_vol_60", "行业"]:
            if col in day.columns:
                keep_cols.append(col)
        scored = day[keep_cols].copy()
        total = np.zeros(len(day), dtype=float)
        weights = np.asarray(blend_weights, dtype=float)
        weights = weights / weights.sum()
        for i, (name, model) in enumerate(models.items()):
            raw = np.asarray(model.predict(day[feature_cols]), dtype=float)
            scored[f"score_{name}"] = raw
            total += weights[i] * pd.Series(raw).rank(pct=True, method="first").to_numpy(dtype=float)
        scored["score"] = total
    else:
        scored = score_with_models(day, feature_cols, models)
    return scored, artifact


def predict_with_ranker(raw_df: pd.DataFrame, artifact_path: str, max_per_industry: int = 0) -> pd.DataFrame:
    scored, artifact = score_ranker_candidates(raw_df, artifact_path)
    policy_spec = artifact.get("best_policy", artifact.get("policy", "risk_adjusted"))
    if isinstance(policy_spec, dict):
        policy = policy_spec.get("policy", "risk_adjusted")
        temperature = policy_spec.get("temperature", artifact["ranker_config"].get("weight_temperature", 0.02))
        max_per_industry = int(policy_spec.get("max_per_industry", max_per_industry))
        fixed_template = policy_spec.get("fixed_template", "")
    else:
        policy = policy_spec
        temperature = artifact["ranker_config"].get("weight_temperature", 0.02)
        fixed_template = ""
    if fixed_template:
        template = np.array([float(x.strip()) for x in fixed_template.split(",") if x.strip()], dtype=float)
        if len(template) != 5:
            raise ValueError("artifact fixed_template must contain exactly 5 weights")
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
    return select_portfolio(
        scored,
        top_k=artifact["ranker_config"].get("top_k", 5),
        policy=policy,
        temperature=temperature,
        max_per_industry=max_per_industry,
    )


def score_portfolio_candidates(raw_df: pd.DataFrame, artifact_path: str, model_path: str) -> tuple[pd.DataFrame, dict]:
    artifact = joblib.load(artifact_path)
    feature_cfg = FeatureConfig(**artifact["feature_config"])
    feature_cols = artifact["feature_cols"]
    day = _latest_candidates(raw_df, feature_cfg, feature_cols)

    mean = artifact["mean"]
    std = artifact["std"]
    x = ((day[feature_cols].to_numpy(np.float32) - mean) / std).astype(np.float32)
    cfg = artifact["portfolio_config"]
    model = PortfolioScorer(len(feature_cols), cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
    keep_cols = [DATE_COL, ID_COL, "risk_vol_20"]
    if "行业" in day.columns:
        keep_cols.append("行业")
    scored = day[keep_cols].copy()
    scored["score"] = scores
    return scored, artifact


def predict_with_portfolio(
    raw_df: pd.DataFrame,
    artifact_path: str,
    model_path: str,
    max_weight: float | None = 0.45,
    max_per_industry: int = 0,
) -> pd.DataFrame:
    scored, artifact = score_portfolio_candidates(raw_df, artifact_path, model_path)
    cfg = artifact["portfolio_config"]
    return select_portfolio(
        scored,
        top_k=cfg.get("top_k", 5),
        policy="score_softmax",
        temperature=0.2,
        max_weight=max_weight,
        max_per_industry=max_per_industry,
    )


def predict_with_blend(
    raw_df: pd.DataFrame,
    ranker_artifact_path: str,
    portfolio_artifact_path: str,
    portfolio_model_path: str,
    alpha: float,
    max_weight: float | None = 0.45,
    max_per_industry: int = 0,
) -> pd.DataFrame:
    ranker_scored, ranker_artifact = score_ranker_candidates(raw_df, ranker_artifact_path)
    portfolio_scored, _ = score_portfolio_candidates(raw_df, portfolio_artifact_path, portfolio_model_path)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    keep_cols = [DATE_COL, ID_COL, "score", "risk_vol_20"]
    if "行业" in ranker_scored.columns:
        keep_cols.append("行业")
    ranker_scores = ranker_scored[keep_cols].rename(columns={"score": "ranker_score"})
    portfolio_scores = portfolio_scored[[ID_COL, "score"]].rename(columns={"score": "portfolio_score"})
    blended = ranker_scores.merge(portfolio_scores, on=ID_COL, how="inner")
    if len(blended) < 5:
        raise ValueError(f"Fewer than 5 common blended candidates: {len(blended)}")

    blended["ranker_pct"] = blended["ranker_score"].rank(pct=True, method="first")
    blended["portfolio_pct"] = blended["portfolio_score"].rank(pct=True, method="first")
    blended["score"] = alpha * blended["ranker_pct"] + (1.0 - alpha) * blended["portfolio_pct"]
    cfg = ranker_artifact["ranker_config"]
    return select_portfolio(
        blended,
        top_k=cfg.get("top_k", 5),
        policy=ranker_artifact.get("best_policy", "risk_adjusted"),
        temperature=cfg.get("weight_temperature", 0.02),
        max_weight=max_weight,
        max_per_industry=max_per_industry,
    )


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    data_dir = Path(data_path).parent
    full_df = enrich_with_metadata(load_price_data(data_path), data_dir)
    full_df = merge_extra_hot_data(full_df, args.extra_hot_data)
    full_df = merge_extra_factor_dir(full_df, args.extra_factor_dir)
    full_df = add_market_context(full_df)
    raw_df = filter_universe(full_df, maybe_load_universe(args.target_universe))
    if args.method == "portfolio":
        portfolio = predict_with_portfolio(
            raw_df,
            args.portfolio_artifact,
            args.portfolio_model,
            args.max_weight,
            args.max_per_industry,
        )
    elif args.method == "blend":
        portfolio = predict_with_blend(
            raw_df,
            args.ranker_artifact,
            args.portfolio_artifact,
            args.portfolio_model,
            args.blend_alpha,
            args.max_weight,
            args.max_per_industry,
        )
    else:
        portfolio = predict_with_ranker(raw_df, args.ranker_artifact, args.max_per_industry)

    out = portfolio.rename(columns={ID_COL: "stock_id"})[["stock_id", "weight"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    out["weight"] = out["weight"] / out["weight"].sum()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {output_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
