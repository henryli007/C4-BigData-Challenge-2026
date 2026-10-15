from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    ID_COL,
    LOW_COL,
    OPEN_COL,
    PCT_COL,
    THS_HOT_RANK_COL,
)


def _safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return a / (b + 1e-12)


def _rank_pct(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return df.groupby(DATE_COL)[col].rank(pct=True, method="average", ascending=ascending).fillna(0.5)


def _rolling_true_count(mask: pd.Series, window: int) -> pd.Series:
    return mask.astype(float).rolling(window, min_periods=1).sum()


def _consecutive_true(mask: pd.Series) -> pd.Series:
    arr = mask.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(arr), dtype=float)
    run = 0.0
    for i, value in enumerate(arr):
        run = run + 1.0 if value else 0.0
        out[i] = run
    return pd.Series(out, index=mask.index)


def build_decay_risk_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build market-trace risk scores for strong names whose upward logic may be fading.

    The scores are intentionally outside the ranker. They penalize names that look strong
    in longer windows but have recent price/action/event traces consistent with repricing.
    """
    rows = []
    sort_df = raw_df.sort_values([ID_COL, DATE_COL]).copy()
    for code, g in sort_df.groupby(ID_COL, sort=False):
        g = g.sort_values(DATE_COL)
        close = g[CLOSE_COL].astype(float)
        high = g[HIGH_COL].astype(float)
        low = g[LOW_COL].astype(float)
        open_ = g[OPEN_COL].astype(float)
        amount = g[AMOUNT_COL].astype(float).clip(lower=0.0)
        ret_1 = close.pct_change()
        day_range = (high - low).clip(lower=1e-12)
        close_strength = ((close - low) / day_range).clip(0.0, 1.0)
        upper_shadow = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), open_.abs()).clip(lower=0.0)
        amount_rel_20 = _safe_div(amount, amount.rolling(20, min_periods=5).mean())
        high_20 = high.rolling(20, min_periods=5).max()
        high_10 = high.rolling(10, min_periods=5).max()
        high_prev_20 = high_20.shift(1)
        ma5 = close.rolling(5, min_periods=3).mean()
        ma10 = close.rolling(10, min_periods=5).mean()
        m = pd.DataFrame({DATE_COL: g[DATE_COL], ID_COL: code})
        m["ret_1"] = ret_1
        m["ret_2"] = close / close.shift(2) - 1.0
        m["ret_5"] = close / close.shift(5) - 1.0
        m["ret_10"] = close / close.shift(10) - 1.0
        m["ret_20"] = close / close.shift(20) - 1.0
        m["momentum_slope_5_20"] = m["ret_5"] - m["ret_20"] * 0.25
        m["near_high_20"] = _safe_div(close, high_20) - 1.0
        m["near_high_10"] = _safe_div(close, high_10) - 1.0
        m["new_high_20"] = (high >= high_prev_20).astype(float)
        m["new_high_fail_count_5"] = _rolling_true_count((m["ret_5"] > 0.08) & (m["new_high_20"] <= 0), 5)
        m["amount_rel_20"] = amount_rel_20
        m["high_volume_down"] = ((ret_1 < 0.0) & (amount_rel_20 > 1.25)).astype(float)
        m["high_volume_down_count_5"] = _rolling_true_count(m["high_volume_down"] > 0, 5)
        m["volume_price_divergence"] = ((amount_rel_20 > 1.25) & (ret_1 < ret_1.rolling(5, min_periods=3).mean())).astype(float)
        m["volume_price_divergence_count_5"] = _rolling_true_count(m["volume_price_divergence"] > 0, 5)
        m["upper_shadow_3"] = upper_shadow.rolling(3, min_periods=1).mean()
        m["upper_shadow_5"] = upper_shadow.rolling(5, min_periods=1).mean()
        m["close_strength"] = close_strength
        m["close_strength_chg_3"] = close_strength - close_strength.shift(3)
        m["close_strength_mean_3"] = close_strength.rolling(3, min_periods=1).mean()
        m["below_ma5"] = (close < ma5).astype(float)
        m["below_ma10"] = (close < ma10).astype(float)
        m["up_streak"] = _consecutive_true(ret_1 > 0.0)
        m["bull_candle_streak"] = _consecutive_true(close > open_)
        if THS_HOT_RANK_COL in g.columns:
            hot_rank = g[THS_HOT_RANK_COL].astype(float)
            hot_score = (101.0 - hot_rank.fillna(999.0)).clip(lower=0.0) / 100.0
            m["hot_score"] = hot_score
            m["hot_score_chg_3"] = hot_score - hot_score.shift(3)
            m["hot_score_chg_5"] = hot_score - hot_score.shift(5)
        else:
            m["hot_score"] = 0.0
            m["hot_score_chg_3"] = 0.0
            m["hot_score_chg_5"] = 0.0
        rows.append(m)

    risk = pd.concat(rows, ignore_index=True)
    risk = risk.replace([np.inf, -np.inf], np.nan)

    market_base = sort_df[[DATE_COL, ID_COL, CLOSE_COL, AMOUNT_COL]].copy()
    market_base["stock_ret_1"] = market_base.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change()
    market_base["stock_ret_5"] = market_base.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change(5)
    market_base["stock_amount_chg_1"] = market_base.groupby(ID_COL, sort=False)[AMOUNT_COL].pct_change()
    market = (
        market_base.groupby(DATE_COL, sort=False)
        .agg(
            market_ret_1=("stock_ret_1", "mean"),
            market_return_5=("stock_ret_5", "mean"),
            market_up_ratio=("stock_ret_1", lambda s: float((s > 0).mean())),
            market_strong_ratio=("stock_ret_5", lambda s: float((s > 0.03).mean())),
            market_amount_chg_1=("stock_amount_chg_1", "mean"),
            market_vol_10=("stock_ret_1", lambda s: float(s.std(ddof=0))),
        )
        .reset_index()
        .sort_values(DATE_COL)
    )
    market["market_ret_2"] = market["market_ret_1"].rolling(2, min_periods=1).sum()
    market["market_diffusion_change_5"] = market["market_strong_ratio"].diff(5)
    market["market_up_ratio_change_3"] = market["market_up_ratio"].diff(3)
    risk = risk.merge(market, on=DATE_COL, how="left")
    risk["market_regime_score"] = (
        0.25 * (risk["market_ret_2"] < -0.012).astype(float)
        + 0.25 * (risk["market_up_ratio"] < 0.42).astype(float)
        + 0.18 * (risk["market_amount_chg_1"] < -0.12).astype(float)
        + 0.17 * (risk["market_diffusion_change_5"] < -0.12).astype(float)
        + 0.15 * ((risk["market_return_5"] < 0.0) & (risk["market_vol_10"] > risk["market_vol_10"].median())).astype(float)
    ).clip(0.0, 1.0)

    if "行业" in raw_df.columns:
        ind_base = sort_df[[DATE_COL, ID_COL, "行业", CLOSE_COL, AMOUNT_COL]].copy()
        ind_base["stock_ret_1"] = ind_base.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change()
        ind_base["stock_amount_chg_1"] = ind_base.groupby(ID_COL, sort=False)[AMOUNT_COL].pct_change()
        ind_base["stock_ret_5"] = ind_base.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change(5)
        ind_base["stock_ret_10"] = ind_base.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change(10)
        ind = (
            ind_base.groupby([DATE_COL, "行业"], sort=False)
            .agg(
                industry_ret_1=("stock_ret_1", "mean"),
                industry_return_5=("stock_ret_5", "mean"),
                industry_return_10=("stock_ret_10", "mean"),
                industry_up_ratio=("stock_ret_1", lambda s: float((s > 0).mean())),
                industry_strong_ratio=("stock_ret_5", lambda s: float((s > 0.03).mean())),
                industry_amount_chg_1=("stock_amount_chg_1", "mean"),
            )
            .reset_index()
            .sort_values(["行业", DATE_COL])
        )
        ind["industry_ret_2"] = ind.groupby("行业", sort=False)["industry_ret_1"].transform(lambda s: s.rolling(2, min_periods=1).sum())
        ind["industry_strength_change_5"] = ind.groupby("行业", sort=False)["industry_return_5"].diff(5)
        ind["industry_diffusion_change_5"] = ind.groupby("行业", sort=False)["industry_strong_ratio"].diff(5)
        ind["industry_up_ratio_change_3"] = ind.groupby("行业", sort=False)["industry_up_ratio"].diff(3)
        risk = risk.merge(ind_base[[DATE_COL, ID_COL, "行业"]].drop_duplicates([DATE_COL, ID_COL]), on=[DATE_COL, ID_COL], how="left")
        risk = risk.merge(ind, on=[DATE_COL, "行业"], how="left")
    else:
        risk["industry_ret_1"] = 0.0
        risk["industry_ret_2"] = 0.0
        risk["industry_return_5"] = 0.0
        risk["industry_return_10"] = 0.0
        risk["industry_up_ratio"] = 0.5
        risk["industry_strong_ratio"] = 0.0
        risk["industry_amount_chg_1"] = 0.0
        risk["industry_strength_change_5"] = 0.0
        risk["industry_diffusion_change_5"] = 0.0
        risk["industry_up_ratio_change_3"] = 0.0

    risk = risk.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    risk["ret20_rank"] = _rank_pct(risk, "ret_20")
    risk["ret10_rank"] = _rank_pct(risk, "ret_10")
    risk["ret5_rank"] = _rank_pct(risk, "ret_5")
    risk["amount_rel_rank"] = _rank_pct(risk, "amount_rel_20")
    risk["upper_shadow_rank"] = _rank_pct(risk, "upper_shadow_3")

    prior_strength = (
        0.45 * risk["ret20_rank"]
        + 0.35 * risk["ret10_rank"]
        + 0.20 * risk["ret5_rank"]
    ).clip(0.0, 1.0)
    recent_weak = (
        0.35 * (risk["ret_2"] < -0.015).astype(float)
        + 0.25 * (risk["ret_1"] < 0.0).astype(float)
        + 0.20 * (risk["momentum_slope_5_20"] < -0.03).astype(float)
        + 0.20 * (risk["close_strength_chg_3"] < -0.25).astype(float)
    )
    risk["strength_decay_score"] = (prior_strength * recent_weak).clip(0.0, 1.0)

    risk["runup_reversal_score"] = (
        0.28 * ((risk["ret10_rank"] > 0.88) | (risk["ret5_rank"] > 0.90)).astype(float)
        + 0.20 * (risk["up_streak"] >= 4).astype(float)
        + 0.14 * (risk["bull_candle_streak"] >= 4).astype(float)
        + 0.18 * ((risk["near_high_20"] < -0.035) & (risk["ret_20"] > 0.12)).astype(float)
        + 0.12 * ((risk["new_high_fail_count_5"] >= 2) & (risk["ret_10"] > 0.08)).astype(float)
        + 0.10 * (risk["upper_shadow_rank"] > 0.75).astype(float)
        + 0.10 * ((risk["below_ma5"] > 0) | (risk["below_ma10"] > 0)).astype(float)
    ).clip(0.0, 1.0)

    industry_weak = (
        0.25 * (risk["industry_ret_2"] < -0.015).astype(float)
        + 0.20 * (risk["industry_up_ratio"] < 0.38).astype(float)
        + 0.15 * (risk["industry_amount_chg_1"] < -0.15).astype(float)
        + 0.20 * ((risk["industry_return_10"] > 0.05) & (risk["industry_strength_change_5"] < -0.025)).astype(float)
        + 0.20 * (risk["industry_diffusion_change_5"] < -0.15).astype(float)
    )
    hot_fading = ((risk["hot_score"] > 0.2) & ((risk["hot_score_chg_3"] < -0.10) | (risk["hot_score_chg_5"] < -0.18))).astype(float)
    risk["theme_lifecycle_risk"] = (
        0.35 * ((risk["industry_return_10"] > 0.05) & (risk["industry_strength_change_5"] < -0.025)).astype(float)
        + 0.30 * (risk["industry_diffusion_change_5"] < -0.15).astype(float)
        + 0.20 * (risk["industry_up_ratio_change_3"] < -0.20).astype(float)
        + 0.15 * hot_fading
    ).clip(0.0, 1.0)
    risk["event_proxy_risk"] = (
        0.25 * risk["high_volume_down"].fillna(0.0)
        + 0.20 * (risk["high_volume_down_count_5"] >= 2).astype(float)
        + 0.20 * (risk["volume_price_divergence_count_5"] >= 2).astype(float)
        + 0.20 * industry_weak
        + 0.20 * hot_fading
    ).clip(0.0, 1.0)

    risk["total_decay_risk"] = (
        0.34 * risk["strength_decay_score"]
        + 0.27 * risk["runup_reversal_score"]
        + 0.18 * risk["event_proxy_risk"]
        + 0.11 * risk["theme_lifecycle_risk"]
        + 0.10 * risk["market_regime_score"]
    ).clip(0.0, 1.0)
    keep = [
        DATE_COL,
        ID_COL,
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
        "strength_decay_score",
        "runup_reversal_score",
        "event_proxy_risk",
        "theme_lifecycle_risk",
        "market_regime_score",
        "total_decay_risk",
    ]
    if "行业" in risk.columns:
        keep.append("行业")
    return risk[keep].fillna(0.0)


def apply_decay_risk_adjustment(
    scored: pd.DataFrame,
    risk: pd.DataFrame,
    score_col: str = "score",
    output_col: str = "score",
    penalty_weight: float = 0.22,
) -> pd.DataFrame:
    work = scored.merge(risk, on=[DATE_COL, ID_COL], how="left", suffixes=("", "_risk"))
    risk_cols = [
        "strength_decay_score",
        "runup_reversal_score",
        "event_proxy_risk",
        "theme_lifecycle_risk",
        "market_regime_score",
        "total_decay_risk",
    ]
    work[risk_cols] = work[risk_cols].fillna(0.0)
    model_pct = work.groupby(DATE_COL)[score_col].rank(pct=True, method="first")
    penalty = float(penalty_weight) * (
        0.34 * work["strength_decay_score"]
        + 0.27 * work["runup_reversal_score"]
        + 0.18 * work["event_proxy_risk"]
        + 0.11 * work["theme_lifecycle_risk"]
        + 0.10 * work["market_regime_score"]
    )
    work[f"{output_col}_model_raw"] = work[score_col]
    work["risk_penalty"] = penalty
    work[output_col] = model_pct - penalty
    return work


def confidence_weight_template(selected: pd.DataFrame) -> np.ndarray:
    scores = selected["score"].to_numpy(dtype=float)
    risks = selected.get("total_decay_risk", pd.Series(0.0, index=selected.index)).to_numpy(dtype=float)
    if len(scores) < 5:
        return np.ones(len(scores), dtype=float) / max(len(scores), 1)
    gap12 = float(scores[0] - scores[1])
    gap15 = float(scores[0] - scores[4])
    top_risk = float(np.nan_to_num(risks[0], nan=0.0))
    avg_risk = float(np.nanmean(np.nan_to_num(risks, nan=0.0)))

    if gap12 >= 0.08 and gap15 >= 0.16 and top_risk <= 0.25:
        template = np.array([0.60, 0.15, 0.10, 0.10, 0.05], dtype=float)
    elif gap12 >= 0.045 and top_risk <= 0.38 and avg_risk <= 0.45:
        template = np.array([0.40, 0.20, 0.15, 0.15, 0.10], dtype=float)
    elif gap12 >= 0.025 and top_risk <= 0.55:
        template = np.array([0.34, 0.22, 0.18, 0.14, 0.12], dtype=float)
    else:
        template = np.array([0.24, 0.22, 0.20, 0.18, 0.16], dtype=float)

    if top_risk > 0.55:
        template = np.array([0.24, 0.22, 0.20, 0.18, 0.16], dtype=float)
    return template / template.sum()
