from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATE_COL = "日期"
ID_COL = "股票代码"
LABEL_COL = "label_return"

TEMPLATES = {
    "equal": [0.20, 0.20, 0.20, 0.20, 0.20],
    "25_22_20_18_15": [0.25, 0.22, 0.20, 0.18, 0.15],
    "35_22_18_15_10": [0.35, 0.22, 0.18, 0.15, 0.10],
    "40_20_15_15_10": [0.40, 0.20, 0.15, 0.15, 0.10],
}

BASE_SIGNAL_COLS = [
    "score",
    "final_score_tuned",
    "score_before_failure",
    "score_before_continuation",
    "score_model_raw",
    "model_score",
    "stage1_recall_score",
    "continuation_score",
    "failure_score",
    "risk_penalty",
    "total_decay_risk",
    "strength_decay_score",
    "runup_reversal_score",
    "event_proxy_risk",
    "theme_lifecycle_risk",
    "market_regime_score",
    "ret_1",
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
    "soft_failure_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a Top20 meta selector using existing walk-forward scores.")
    parser.add_argument("--validation-scores", required=True)
    parser.add_argument("--current-scores", default="")
    parser.add_argument("--output", default="./output/result.csv")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--score-col", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--select-k", type=int, default=5)
    return parser.parse_args()


def _choose_score_col(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Missing score col: {requested}")
        return requested
    for col in ["final_score_tuned", "score", "score_before_failure", "score_model_raw", "model_score"]:
        if col in df.columns:
            return col
    raise ValueError("No score column found.")


def _prepare_topn(df: pd.DataFrame, score_col: str, top_n: int, with_label: bool) -> pd.DataFrame:
    rows = []
    for date, day in df.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[score_col]).copy()
        if with_label:
            day = day.dropna(subset=[LABEL_COL]).copy()
            day["future_rank_full"] = day[LABEL_COL].rank(method="first", ascending=False)
            day["target10"] = (day["future_rank_full"] <= 10).astype(int)
            day["target5"] = (day["future_rank_full"] <= 5).astype(int)
            day["meta_rel"] = (
                8 * (day["future_rank_full"] <= 1).astype(int)
                + 6 * day["future_rank_full"].between(2, 3).astype(int)
                + 5 * day["future_rank_full"].between(4, 5).astype(int)
                + 3 * day["future_rank_full"].between(6, 10).astype(int)
                + 3 * (day[LABEL_COL] >= 0.10).astype(int)
                + 1 * (day[LABEL_COL] >= 0.06).astype(int)
                - 1 * (day[LABEL_COL] < 0).astype(int)
            ).clip(lower=0)
        top = day.sort_values(score_col, ascending=False).head(top_n).copy()
        top["candidate_rank"] = np.arange(1, len(top) + 1)
        for col in BASE_SIGNAL_COLS:
            if col in top.columns and pd.api.types.is_numeric_dtype(top[col]):
                top[f"{col}_rank_hi"] = top[col].rank(pct=True, method="first")
                top[f"{col}_rank_lo"] = top[col].rank(pct=True, method="first", ascending=False)
                top[f"{col}_z"] = (top[col] - top[col].mean()) / (top[col].std() + 1e-9)
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    blocked = {DATE_COL, ID_COL, LABEL_COL, "cutoff", "行业", "行业_risk", "future_rank_full", "target10", "target5", "meta_rel"}
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def _models() -> dict[str, tuple[object, str]]:
    return {
        "hgb10a": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=35, learning_rate=0.03, max_leaf_nodes=5, l2_regularization=10.0, random_state=1),
            ),
            "target10",
        ),
        "hgb10b": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=50, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=20.0, random_state=2),
            ),
            "target10",
        ),
        "hgb5": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=35, learning_rate=0.03, max_leaf_nodes=5, l2_regularization=10.0, random_state=3),
            ),
            "target5",
        ),
        "logit10": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000)),
            "target10",
        ),
        "et10": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(n_estimators=500, max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=4, n_jobs=-1),
            ),
            "target10",
        ),
        "rf10": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=400, max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=5, n_jobs=-1),
            ),
            "target10",
        ),
        "lgb10": (
            LGBMClassifier(
                n_estimators=35,
                learning_rate=0.03,
                num_leaves=5,
                max_depth=3,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.65,
                reg_alpha=1.0,
                reg_lambda=20.0,
                class_weight="balanced",
                random_state=6,
                verbose=-1,
            ),
            "target10",
        ),
    }


def _score_model(model: object, train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    model.fit(train[features], train[target])
    return np.asarray(model.predict_proba(test[features])[:, 1], dtype=float)


def _score_ranker(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    ordered = train.sort_values([DATE_COL, "candidate_rank"]).copy()
    groups = ordered.groupby(DATE_COL, sort=True).size().tolist()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=35,
        learning_rate=0.03,
        num_leaves=5,
        max_depth=3,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.65,
        reg_alpha=1.0,
        reg_lambda=20.0,
        random_state=10,
        verbose=-1,
    )
    model.fit(ordered[features], ordered["meta_rel"], group=groups)
    return np.asarray(model.predict(test[features]), dtype=float)


def _evaluate_selector(name: str, prepared: pd.DataFrame, features: list[str], scorer: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    details = []
    for date in sorted(prepared[DATE_COL].unique()):
        train = prepared[prepared[DATE_COL] != date].copy()
        test = prepared[prepared[DATE_COL] == date].copy()
        test["selector_score"] = scorer(train, test)
        selected = test.sort_values("selector_score", ascending=False).head(5).copy()
        for template_name, weights in TEMPLATES.items():
            w = np.asarray(weights, dtype=float)
            rows.append(
                {
                    "model": name,
                    "template": template_name,
                    "date": date,
                    "return_equal": float(selected[LABEL_COL].mean()),
                    "return_weighted": float(np.dot(w, selected[LABEL_COL].to_numpy(dtype=float))),
                    "hit_true_top5": int((selected["future_rank_full"] <= 5).sum()),
                    "hit_true_top10": int((selected["future_rank_full"] <= 10).sum()),
                    "rank1_hit10": int(float(selected.iloc[0][LABEL_COL]) >= 0.10),
                    "selected": ",".join(selected[ID_COL].astype(str).str.zfill(6).tolist()),
                }
            )
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            details.append(
                {
                    "model": name,
                    "date": date,
                    "selected_rank": rank,
                    "stock_id": str(row[ID_COL]).zfill(6),
                    "selector_score": float(row["selector_score"]),
                    "future_return": float(row[LABEL_COL]),
                    "future_rank_full": float(row["future_rank_full"]),
                }
            )
    return rows, details


def _summarize(per_week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, template), g in per_week.groupby(["model", "template"], sort=False):
        eq = g["return_equal"].to_numpy(dtype=float)
        wt = g["return_weighted"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "template": template,
                "weeks": int(len(g)),
                "return_equal": float(eq.mean()),
                "return_weighted": float(wt.mean()),
                "hit_true_top5": float(g["hit_true_top5"].mean()),
                "hit_true_top10": float(g["hit_true_top10"].mean()),
                "rank1_hit10": int(g["rank1_hit10"].sum()),
                "hit10_weeks": int((wt >= 0.10).sum()),
                "win_rate": float((wt > 0).mean()),
                "max_drawdown": float(wt.min()),
            }
        )
    out = pd.DataFrame(rows)
    out["selection_score"] = (
        out["hit_true_top10"]
        + 0.8 * out["hit_true_top5"]
        + 0.08 * out["rank1_hit10"]
        + 2.0 * out["return_weighted"]
        + 0.35 * out["win_rate"]
        + 0.10 * out["hit10_weeks"]
        - 0.25 * out["max_drawdown"].clip(upper=0).abs()
    )
    return out.sort_values(["selection_score", "return_weighted", "hit_true_top10"], ascending=False)


def _fit_final_selector(prepared: pd.DataFrame, features: list[str], model_name: str) -> object:
    if model_name == "lgb_rank":
        return None
    model, target = _models()[model_name]
    model.fit(prepared[features], prepared[target])
    return model


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.validation_scores, dtype={ID_COL: str})
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])
    score_col = _choose_score_col(raw, args.score_col)
    prepared = _prepare_topn(raw, score_col, args.top_n, with_label=True)
    features = _feature_cols(prepared)

    per_week_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for name, (model, target) in _models().items():
        rows, details = _evaluate_selector(name, prepared, features, lambda tr, te, model=model, target=target: _score_model(model, tr, te, features, target))
        per_week_rows.extend(rows)
        detail_rows.extend(details)
    rows, details = _evaluate_selector("lgb_rank", prepared, features, lambda tr, te: _score_ranker(tr, te, features))
    per_week_rows.extend(rows)
    detail_rows.extend(details)

    per_week = pd.DataFrame(per_week_rows)
    summary = _summarize(per_week)
    per_week.to_csv(report_dir / "selector_per_week.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(report_dir / "selector_details.csv", index=False)
    summary.to_csv(report_dir / "selector_summary.csv", index=False)
    best = summary.iloc[0].to_dict()

    final_selected = []
    if args.current_scores:
        current = pd.read_csv(args.current_scores, dtype={ID_COL: str})
        current[DATE_COL] = pd.to_datetime(current[DATE_COL])
        current_score_col = _choose_score_col(current, "score" if "score" in current.columns else "")
        current_top = _prepare_topn(current[current[DATE_COL] == current[DATE_COL].max()].copy(), current_score_col, args.top_n, with_label=False)
        for col in features:
            if col not in current_top.columns:
                current_top[col] = np.nan
        if best["model"] == "lgb_rank":
            scores = _score_ranker(prepared, current_top, features)
        else:
            model = _fit_final_selector(prepared, features, str(best["model"]))
            scores = np.asarray(model.predict_proba(current_top[features])[:, 1], dtype=float)
        current_top["selector_score"] = scores
        selected = current_top.sort_values("selector_score", ascending=False).head(args.select_k).copy()
        weights = np.asarray(TEMPLATES[str(best["template"])], dtype=float)[: len(selected)]
        weights = weights / weights.sum()
        out = selected.rename(columns={ID_COL: "stock_id"})[["stock_id"]].copy()
        out["stock_id"] = out["stock_id"].astype(str).str.zfill(6)
        out["weight"] = weights
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False, float_format="%.6f")
        selected.to_csv(report_dir / "current_selected_detail.csv", index=False)
        final_selected = out.to_dict(orient="records")

    payload = {
        "score_col": score_col,
        "features": len(features),
        "best": best,
        "selected": final_selected,
    }
    (report_dir / "selection_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
