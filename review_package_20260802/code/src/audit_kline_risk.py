from __future__ import annotations

import argparse
from pathlib import Path

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
    TURNOVER_COL,
    load_price_data,
    normalize_stock_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pure K-line risk for current stock candidates.")
    parser.add_argument("--data", default="./data/hybrid_stock_data.csv")
    parser.add_argument("--scored", default="./output/current_lgb_w60_q030_scored_20260626.csv")
    parser.add_argument("--submission", default="./output/result.csv")
    parser.add_argument("--output", default="./output/kline_risk_audit.csv")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def _latest_metrics(g: pd.DataFrame) -> dict[str, float | str]:
    g = g.sort_values(DATE_COL).copy()
    latest = g.iloc[-1]
    close = g[CLOSE_COL].astype(float)
    amount = g[AMOUNT_COL].astype(float)
    high = g[HIGH_COL].astype(float)
    low = g[LOW_COL].astype(float)
    open_ = g[OPEN_COL].astype(float)
    ret = close.pct_change()

    ma_amount_20 = amount.rolling(20, min_periods=5).mean()
    high_20 = high.rolling(20, min_periods=5).max()
    low_20 = low.rolling(20, min_periods=5).min()
    vol_10 = ret.rolling(10, min_periods=5).std()
    vol_20 = ret.rolling(20, min_periods=5).std()

    last_open = float(latest[OPEN_COL])
    last_close = float(latest[CLOSE_COL])
    last_high = float(latest[HIGH_COL])
    last_low = float(latest[LOW_COL])
    day_range = max(last_high - last_low, 1e-12)

    return {
        "latest_date": str(pd.Timestamp(latest[DATE_COL]).date()),
        "last_pct": float(latest.get(PCT_COL, np.nan)),
        "last_intraday_ret": last_close / max(last_open, 1e-12) - 1.0,
        "last_close_pos": (last_close - last_low) / day_range,
        "last_upper_shadow": (last_high - max(last_open, last_close)) / max(last_open, 1e-12),
        "last_lower_shadow": (min(last_open, last_close) - last_low) / max(last_open, 1e-12),
        "ret_3": close.iloc[-1] / close.iloc[-4] - 1.0 if len(close) >= 4 else np.nan,
        "ret_5": close.iloc[-1] / close.iloc[-6] - 1.0 if len(close) >= 6 else np.nan,
        "ret_10": close.iloc[-1] / close.iloc[-11] - 1.0 if len(close) >= 11 else np.nan,
        "ret_20": close.iloc[-1] / close.iloc[-21] - 1.0 if len(close) >= 21 else np.nan,
        "drawdown_20": close.iloc[-1] / max(float(high_20.iloc[-1]), 1e-12) - 1.0,
        "pos_20": (close.iloc[-1] - low_20.iloc[-1]) / max(high_20.iloc[-1] - low_20.iloc[-1], 1e-12),
        "amount_rel_20": amount.iloc[-1] / max(float(ma_amount_20.iloc[-1]), 1e-12),
        "vol_10": float(vol_10.iloc[-1]),
        "vol_ratio_10_20": float(vol_10.iloc[-1] / max(vol_20.iloc[-1], 1e-12)),
        "turnover": float(latest.get(TURNOVER_COL, np.nan)),
    }


def _risk_flags(row: pd.Series) -> tuple[int, list[str]]:
    flags: list[str] = []
    if row["ret_10"] >= 0.25:
        flags.append("10日涨幅过热")
    if row["ret_20"] >= 0.40:
        flags.append("20日涨幅过热")
    if row["amount_rel_20"] >= 1.8:
        flags.append("成交额显著放大")
    if row["vol_10"] >= 0.05:
        flags.append("短期波动高")
    if row["vol_ratio_10_20"] >= 1.35:
        flags.append("波动正在升高")
    if row["last_pct"] <= -5.0:
        flags.append("最新日大跌")
    if row["last_intraday_ret"] <= -0.04:
        flags.append("最新日开盘后走弱")
    if row["last_close_pos"] <= 0.25:
        flags.append("收盘接近日内低位")
    if row["last_upper_shadow"] >= 0.04:
        flags.append("上影线较长")
    if row["drawdown_20"] <= -0.12 and row["ret_20"] > 0:
        flags.append("强势股出现明显回撤")
    return len(flags), flags


def main() -> None:
    args = parse_args()
    prices = load_price_data(args.data)
    scored = pd.read_csv(args.scored, dtype={ID_COL: str})
    scored[ID_COL] = scored[ID_COL].map(normalize_stock_code)
    scored = scored.sort_values("score", ascending=False).head(args.top_n).copy()

    submission = pd.read_csv(args.submission, dtype={"stock_id": str})
    submission["stock_id"] = submission["stock_id"].map(normalize_stock_code)
    weights = dict(zip(submission["stock_id"], submission["weight"].astype(float)))

    metric_rows = []
    for code, g in prices[prices[ID_COL].isin(scored[ID_COL])].groupby(ID_COL):
        row = {ID_COL: code, **_latest_metrics(g)}
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    result = scored.merge(metrics, on=ID_COL, how="left")
    result["submission_weight"] = result[ID_COL].map(weights).fillna(0.0)
    result["score_rank"] = result["score"].rank(method="first", ascending=False).astype(int)

    risk_counts = []
    risk_text = []
    for _, row in result.iterrows():
        count, flags = _risk_flags(row)
        risk_counts.append(count)
        risk_text.append(";".join(flags))
    result["kline_risk_count"] = risk_counts
    result["kline_risk_flags"] = risk_text

    industry_col = "行业" if "行业" in result.columns else None
    if industry_col:
        top10 = result.nsmallest(10, "score_rank")
        industry_counts = top10[industry_col].value_counts()
        result["top10_industry_count"] = result[industry_col].map(industry_counts).fillna(0).astype(int)
        result.loc[result["top10_industry_count"] >= 3, "kline_risk_flags"] += ";Top10行业拥挤"
        result.loc[result["top10_industry_count"] >= 3, "kline_risk_count"] += 1

    ordered_cols = [
        "score_rank",
        ID_COL,
        "submission_weight",
        "score",
        "score_model_only",
        "quality_pct",
        "行业",
        "latest_date",
        "last_pct",
        "last_intraday_ret",
        "last_close_pos",
        "ret_5",
        "ret_10",
        "ret_20",
        "drawdown_20",
        "amount_rel_20",
        "vol_10",
        "vol_ratio_10_20",
        "kline_risk_count",
        "kline_risk_flags",
    ]
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    result = result.sort_values(["submission_weight", "score"], ascending=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result[ordered_cols].to_csv(out, index=False)
    print(result[ordered_cols].head(args.top_n).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
