from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm


def normalize_code(value: object) -> str:
    return str(value).split(".", 1)[0].strip().zfill(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Tushare daily_basic fields into stock_data.csv.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="20260731")
    parser.add_argument("--sleep-seconds", type=float, default=0.03)
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Tushare token env var: {args.token_env}")

    ts.set_token(token)
    pro = ts.pro_api()
    universe = pd.read_csv(args.universe, dtype=str)
    wanted = set(universe["纯代码"].map(normalize_code))

    cal = pro.trade_cal(exchange="", start_date=args.start_date, end_date=args.end_date, fields="cal_date,is_open")
    trade_dates = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).sort_values().tolist()
    rows: list[pd.DataFrame] = []
    failed: list[dict[str, str]] = []
    fields = (
        "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe_ttm,pb,ps_ttm,total_mv,circ_mv"
    )
    for day in tqdm(trade_dates, desc="Tushare daily_basic", unit="day"):
        try:
            df = pro.daily_basic(trade_date=day, fields=fields)
            if df is None or df.empty:
                failed.append({"trade_date": day, "错误信息": "empty"})
                continue
            df["股票代码"] = df["ts_code"].map(normalize_code)
            df = df[df["股票代码"].isin(wanted)].copy()
            rows.append(df)
        except Exception as exc:
            failed.append({"trade_date": day, "错误信息": repr(exc)})
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not rows:
        raise RuntimeError("No daily_basic rows fetched.")
    basic = pd.concat(rows, ignore_index=True)
    basic["日期"] = pd.to_datetime(basic["trade_date"]).dt.strftime("%Y-%m-%d")
    basic = basic.rename(
        columns={
            "turnover_rate": "换手率",
            "turnover_rate_f": "自由流通换手率",
            "volume_ratio": "量比",
            "pe_ttm": "市盈率TTM",
            "pb": "市净率MRQ",
            "ps_ttm": "市销率TTM",
            "total_mv": "总市值",
            "circ_mv": "流通市值",
        }
    )
    keep = [
        "股票代码",
        "日期",
        "换手率",
        "自由流通换手率",
        "量比",
        "市盈率TTM",
        "市净率MRQ",
        "市销率TTM",
        "总市值",
        "流通市值",
    ]
    basic = basic[keep].drop_duplicates(["股票代码", "日期"], keep="last")

    data_path = Path(args.data)
    data = pd.read_csv(data_path, dtype={"股票代码": str}, low_memory=False)
    data["股票代码"] = data["股票代码"].map(normalize_code)
    data["日期"] = pd.to_datetime(data["日期"]).dt.strftime("%Y-%m-%d")
    drop_cols = [c for c in keep if c not in {"股票代码", "日期"} and c in data.columns]
    data = data.drop(columns=drop_cols)
    merged = data.merge(basic, on=["股票代码", "日期"], how="left")
    merged.to_csv(data_path, index=False, encoding="utf-8-sig")

    failed_path = data_path.parent / "failed_daily_basic_tushare.csv"
    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False, encoding="utf-8-sig")
    elif failed_path.exists():
        failed_path.unlink()
    print(
        f"wrote {data_path} rows={len(merged)} stocks={merged['股票代码'].nunique()} "
        f"daily_basic_rows={len(basic)} failed_days={len(failed)}"
    )


if __name__ == "__main__":
    main()
