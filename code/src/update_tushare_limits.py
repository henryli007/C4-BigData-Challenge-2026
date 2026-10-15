from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm


ID_COL = "股票代码"
DATE_COL = "日期"


def _normalize_code(code: object) -> str:
    raw = str(code).strip()
    if raw.startswith(("sh.", "sz.")) or raw.endswith((".SH", ".SZ")):
        raw = raw.split(".", 1)[-1] if raw.startswith(("sh.", "sz.")) else raw.split(".", 1)[0]
    return raw.zfill(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge TuShare daily up/down limit prices into stock_data.csv.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing environment variable {args.token_env}")

    data_path = Path(args.data)
    prices = pd.read_csv(data_path, dtype={ID_COL: str}, encoding="utf-8-sig", low_memory=False)
    prices[ID_COL] = prices[ID_COL].map(_normalize_code)
    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])

    universe = pd.read_csv(args.universe, dtype=str, encoding="utf-8-sig")
    if ID_COL in universe.columns:
        codes = set(universe[ID_COL].dropna().map(_normalize_code))
    elif "code" in universe.columns:
        codes = set(universe["code"].dropna().map(_normalize_code))
    else:
        codes = set(prices[ID_COL].unique())

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date) if args.end_date else prices[DATE_COL].max()
    trade_dates = (
        prices.loc[(prices[DATE_COL] >= start) & (prices[DATE_COL] <= end), DATE_COL]
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y%m%d")
        .tolist()
    )

    pro = ts.pro_api(token)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for trade_date in tqdm(trade_dates, desc="Tushare stk_limit"):
        try:
            df = pro.stk_limit(trade_date=trade_date)
        except Exception as exc:
            failed.append(f"{trade_date}: {exc}")
            time.sleep(max(args.sleep_seconds, 0.5))
            continue
        if df is not None and not df.empty:
            df = df[["trade_date", "ts_code", "up_limit", "down_limit"]].copy()
            df[ID_COL] = df["ts_code"].map(_normalize_code)
            df = df[df[ID_COL].isin(codes)]
            df[DATE_COL] = pd.to_datetime(df["trade_date"])
            df = df.rename(columns={"up_limit": "涨停价", "down_limit": "跌停价"})
            frames.append(df[[ID_COL, DATE_COL, "涨停价", "跌停价"]])
        time.sleep(args.sleep_seconds)

    if not frames:
        raise RuntimeError("No stk_limit rows were fetched.")

    limits = pd.concat(frames, ignore_index=True).drop_duplicates([ID_COL, DATE_COL], keep="last")
    for col in ["涨停价", "跌停价"]:
        limits[col] = pd.to_numeric(limits[col], errors="coerce")
    prices = prices.drop(columns=[c for c in ["涨停价", "跌停价"] if c in prices.columns])
    merged = prices.merge(limits, on=[ID_COL, DATE_COL], how="left")
    merged.to_csv(data_path, index=False, encoding="utf-8-sig")

    if failed:
        (data_path.parent / "stk_limit_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"wrote {data_path} rows={len(merged)} limit_rows={len(limits)} failed_days={len(failed)}")


if __name__ == "__main__":
    main()
