from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


DATE_COL = "日期"
ID_COL = "股票代码"
LABEL_COL = "label_return"
WEIGHTS = np.array([0.25, 0.22, 0.20, 0.18, 0.15], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Top20-to-Top5 selection without retraining.")
    parser.add_argument("--validation-scores", required=True)
    parser.add_argument("--current-scores", required=True)
    parser.add_argument("--output", default="./output/result.csv")
    parser.add_argument("--report-dir", default="./model/final_selection")
    parser.add_argument("--score-col", default="")
    return parser.parse_args()


def _score_col(df: pd.DataFrame, requested: str = "") -> str:
    if requested:
        return requested
    for col in ["final_score_tuned", "score", "score_before_failure", "score_model_raw"]:
        if col in df.columns:
            return col
    raise ValueError("No usable score column found.")


def _sorted_top20(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = day.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(20).copy()
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    return out


def select_a(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return _sorted_top20(day, score_col).head(5).copy()


def select_b(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    top20 = _sorted_top20(day, score_col)
    parts = [
        top20[(top20["candidate_rank"] >= 1) & (top20["candidate_rank"] <= 5)].head(2),
        top20[(top20["candidate_rank"] >= 6) & (top20["candidate_rank"] <= 10)].head(2),
        top20[(top20["candidate_rank"] >= 11) & (top20["candidate_rank"] <= 20)].head(1),
    ]
    picked = pd.concat(parts, axis=0)
    if len(picked) < 5:
        picked = pd.concat([picked, top20[~top20[ID_COL].isin(picked[ID_COL])].head(5 - len(picked))])
    return picked.head(5).copy()


def select_c(day: pd.DataFrame, score_col: str, lam: float) -> pd.DataFrame:
    top20 = _sorted_top20(day, score_col)
    if top20.empty:
        return top20
    score = top20[score_col].astype(float)
    norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
    top20 = top20.assign(_norm_score=norm)
    selected = []
    remaining = list(top20.index)
    while remaining and len(selected) < 5:
        best_idx = None
        best_value = -1e18
        for idx in remaining:
            row = top20.loc[idx]
            same_industry = 0.0
            if selected and "行业" in top20.columns:
                same_industry = sum(str(row.get("行业", "")) == str(top20.loc[j].get("行业", "")) for j in selected)
            penalty = lam * same_industry
            value = float(row["_norm_score"]) - penalty
            if value > best_value:
                best_value = value
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return top20.loc[selected].copy()


META_FEATURES = [
    "candidate_rank",
    "score",
    "score_model_raw",
    "score_model_raw_rank_hi",
    "stage1_recall_score",
    "stage1_recall_score_rank_hi",
    "continuation_score",
    "continuation_score_rank_hi",
    "failure_score",
    "failure_score_rank_lo",
    "total_decay_risk",
    "total_decay_risk_rank_lo",
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_5_rank_hi",
    "ret_10_rank_hi",
    "ret_20_rank_hi",
    "near_high_20",
    "amount_rel_20",
    "amount_rel_20_rank_hi",
    "volume_price_divergence_count_5",
    "upper_shadow_3",
    "close_strength",
    "close_strength_chg_3",
    "below_ma5",
    "below_ma10",
    "up_streak",
    "bull_candle_streak",
    "high_volume_down_count_5",
    "industry_ret_2",
    "industry_return_5",
    "industry_return_10",
    "industry_up_ratio",
    "industry_strong_ratio",
    "industry_strength_change_5",
    "industry_diffusion_change_5",
    "hot_score_chg_3",
    "strength_decay_score",
    "runup_reversal_score",
    "event_proxy_risk",
    "theme_lifecycle_risk",
    "market_regime_score",
    "soft_failure_filter",
]


def _prepare_meta_top20(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    top20 = _sorted_top20(day, score_col)
    for col in [
        "score",
        "score_model_raw",
        "stage1_recall_score",
        "continuation_score",
        "failure_score",
        "total_decay_risk",
        "ret_5",
        "ret_10",
        "ret_20",
        "amount_rel_20",
    ]:
        if col in top20.columns:
            top20[f"{col}_rank_hi"] = top20[col].rank(pct=True, method="first")
            top20[f"{col}_rank_lo"] = top20[col].rank(pct=True, method="first", ascending=False)
    return top20


def _add_full_future_rank(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if LABEL_COL in work.columns:
        work["future_rank_full"] = work.groupby(DATE_COL)[LABEL_COL].rank(method="first", ascending=False)
    return work


def _fit_meta_model(train: pd.DataFrame, score_col: str, model_name: str = "hgb10b") -> tuple[object, list[str]]:
    parts = []
    for _, day in train.groupby(DATE_COL, sort=True):
        top20 = _prepare_meta_top20(day, score_col)
        if "future_rank_full" in top20.columns:
            ranks = top20["future_rank_full"]
        else:
            full_ranks = day[LABEL_COL].rank(method="first", ascending=False)
            ranks = full_ranks.loc[top20.index]
        top20["target10"] = (ranks <= 10).astype(int)
        parts.append(top20)
    meta = pd.concat(parts, ignore_index=True)
    features = [c for c in META_FEATURES if c in meta.columns]
    configs = {
        "hgb10a": dict(max_iter=35, learning_rate=0.03, max_leaf_nodes=5, l2_regularization=10.0, random_state=1),
        "hgb10b": dict(max_iter=50, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=20.0, random_state=2),
    }
    cfg = configs.get(model_name, configs["hgb10b"])
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(**cfg),
    )
    model.fit(meta[features], meta["target10"])
    return model, features


def select_meta(day: pd.DataFrame, score_col: str, train: pd.DataFrame | None = None, model_name: str = "hgb10b") -> pd.DataFrame:
    if train is None:
        raise ValueError("meta selector needs a training frame")
    model, features = _fit_meta_model(train, score_col, model_name=model_name)
    top20 = _prepare_meta_top20(day, score_col)
    top20["meta_score"] = model.predict_proba(top20[features])[:, 1]
    return top20.sort_values("meta_score", ascending=False).head(5).copy()


def _future_ranks(day: pd.DataFrame) -> pd.Series:
    return day[LABEL_COL].rank(method="first", ascending=False)


def _eval_strategy(df: pd.DataFrame, score_col: str, name: str, selector) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    details = []
    for date, day in df.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, score_col]).copy()
        if len(day) < 20:
            continue
        if name in {"D_meta_hgb10a", "D_meta_hgb10b"}:
            train = df[df[DATE_COL] != date].copy()
            selected = select_meta(day, score_col, train=train, model_name=name.replace("D_meta_", "")).copy()
        else:
            selected = selector(day, score_col).copy()
        if len(selected) < 5:
            continue
        ranks = day["future_rank_full"] if "future_rank_full" in day.columns else _future_ranks(day)
        selected["future_rank"] = ranks.loc[selected.index].to_numpy()
        selected["strategy"] = name
        selected["week"] = date
        details.append(selected[[c for c in ["week", "strategy", ID_COL, score_col, LABEL_COL, "future_rank", "candidate_rank", "行业"] if c in selected.columns]])
        top20 = _sorted_top20(day, score_col)
        true10 = set(day.loc[ranks <= 10, ID_COL])
        rows.append(
            {
                "strategy": name,
                "week": date,
                "top5_equal_return": float(selected[LABEL_COL].mean()),
                "weighted_return": float(np.dot(WEIGHTS, selected[LABEL_COL].to_numpy(dtype=float))),
                "hit_true_top5": int((selected["future_rank"] <= 5).sum()),
                "hit_true_top10": int((selected["future_rank"] <= 10).sum()),
                "rank1_hit10": int(float(selected[LABEL_COL].iloc[0]) >= 0.10),
                "pred_top1_future_rank": int(selected["future_rank"].iloc[0]),
                "pred_top5_avg_future_rank": float(selected["future_rank"].mean()),
                "true_top10_in_pred20": int(len(set(top20[ID_COL]) & true10)),
            }
        )
    per_week = pd.DataFrame(rows)
    if per_week.empty:
        return {}, pd.DataFrame()
    summary = {
        "strategy": name,
        "weeks": int(len(per_week)),
        "top5_equal_return": float(per_week["top5_equal_return"].mean()),
        "weighted_return": float(per_week["weighted_return"].mean()),
        "hit_true_top5": float(per_week["hit_true_top5"].mean()),
        "hit_true_top10": float(per_week["hit_true_top10"].mean()),
        "rank1_hit10": int(per_week["rank1_hit10"].sum()),
        "win_rate": float((per_week["top5_equal_return"] > 0).mean()),
        "max_drawdown": float(per_week["top5_equal_return"].min()),
        "pred_top1_avg_future_rank": float(per_week["pred_top1_future_rank"].mean()),
        "pred_top5_avg_future_rank": float(per_week["pred_top5_avg_future_rank"].mean()),
        "true_top10_in_pred20": float(per_week["true_top10_in_pred20"].mean()),
    }
    return summary, pd.concat(details, ignore_index=True)


def _choose(summary: pd.DataFrame) -> str:
    work = summary.copy()
    work["selection_score"] = (
        work["hit_true_top10"]
        + 0.60 * work["hit_true_top5"]
        + 0.15 * work["rank1_hit10"]
        + 2.0 * work["top5_equal_return"]
        + 0.2 * work["win_rate"]
        - 0.2 * work["max_drawdown"].clip(upper=0).abs()
    )
    return str(work.sort_values(["selection_score", "top5_equal_return"], ascending=False).iloc[0]["strategy"])


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    valid = pd.read_csv(args.validation_scores, dtype={ID_COL: str})
    valid[DATE_COL] = pd.to_datetime(valid[DATE_COL])
    valid = _add_full_future_rank(valid)
    score_col = _score_col(valid, args.score_col)
    selectors = {
        "A_top5": select_a,
        "B_layered_2_2_1": select_b,
        "C_mmr_lam0p1": lambda day, col: select_c(day, col, 0.1),
        "C_mmr_lam0p2": lambda day, col: select_c(day, col, 0.2),
        "D_meta_hgb10a": select_a,
        "D_meta_hgb10b": select_a,
    }
    summaries = []
    detail_parts = []
    for name, selector in selectors.items():
        summary, detail = _eval_strategy(valid, score_col, name, selector)
        if summary:
            summaries.append(summary)
            detail_parts.append(detail)
    summary_df = pd.DataFrame(summaries)
    chosen = _choose(summary_df)
    summary_df.to_csv(report_dir / "selection_summary.csv", index=False)
    if detail_parts:
        pd.concat(detail_parts, ignore_index=True).to_csv(report_dir / "selection_details.csv", index=False)

    current = pd.read_csv(args.current_scores, dtype={ID_COL: str})
    current[DATE_COL] = pd.to_datetime(current[DATE_COL])
    current_score_col = _score_col(current, "score" if "score" in current.columns else "")
    current_day = current[current[DATE_COL] == current[DATE_COL].max()].copy()
    if chosen in {"D_meta_hgb10a", "D_meta_hgb10b"}:
        selected = select_meta(current_day, current_score_col, train=valid, model_name=chosen.replace("D_meta_", "")).copy()
    else:
        selected = selectors[chosen](current_day, current_score_col).copy()
    selected = selected.head(5)
    selected["weight"] = WEIGHTS[: len(selected)]
    selected["weight"] = selected["weight"] / selected["weight"].sum()
    out = selected.rename(columns={ID_COL: "stock_id"})[["stock_id", "weight"]]
    out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, float_format="%.6f")
    (report_dir / "chosen_strategy.json").write_text(
        json.dumps(
            {
                "chosen_strategy": chosen,
                "score_col_validation": score_col,
                "score_col_current": current_score_col,
                "weights": WEIGHTS.tolist(),
                "selected": out.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.sort_values(["hit_true_top10", "hit_true_top5", "top5_equal_return"], ascending=False).to_string(index=False))
    print(json.dumps({"chosen_strategy": chosen, "selected": out.to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
