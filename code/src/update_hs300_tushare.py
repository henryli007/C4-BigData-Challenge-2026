from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import tushare as ts
from tqdm import tqdm


ORDERED_COLS = [
    "股票代码",
    "股票名称",
    "市场",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "昨收",
    "成交量",
    "成交额",
    "换手率",
    "涨跌幅",
    "涨跌额",
    "振幅",
    "交易状态",
    "是否ST",
    "市盈率TTM",
    "市净率MRQ",
    "市销率TTM",
    "市现率TTM",
    "复权标记",
    "复权方式",
]


def normalize_code(value: object) -> str:
    code = str(value).strip()
    if "." in code:
        code = code.split(".", 1)[0]
    return code.zfill(6)


def ts_code_from_code(code: str) -> str:
    code = normalize_code(code)
    suffix = "SH" if code.startswith(("600", "601", "603", "605", "688", "689")) else "SZ"
    return f"{code}.{suffix}"


def market_from_ts_code(ts_code: str) -> str:
    return str(ts_code).split(".", 1)[1].upper()


def fetch_current_hs300_universe() -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol="000300")
    if df.empty:
        raise RuntimeError("AKShare returned empty CSI300 universe.")
    codes = df["成分券代码"].map(normalize_code)
    out = pd.DataFrame(
        {
            "updateDate": pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d"),
            "code": codes.map(lambda x: f"{ts_code_from_code(x).split('.', 1)[1].lower()}.{x}"),
            "股票名称": df["成分券名称"],
            "code_name": df["成分券名称"],
            "纯代码": codes,
            "市场": codes.map(lambda x: ts_code_from_code(x).split(".", 1)[1]),
            "ts_code": codes.map(ts_code_from_code),
        }
    )
    return out.sort_values(["市场", "纯代码"]).reset_index(drop=True)


def fetch_stock_basic(pro) -> pd.DataFrame:
    try:
        basic = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
    except Exception:
        return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry", "market", "list_date"])
    basic["symbol"] = basic["symbol"].map(normalize_code)
    return basic


def transform_bar(df: pd.DataFrame, ts_code: str, name: str, industry: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ORDERED_COLS)
    work = df.sort_values("trade_date").reset_index(drop=True).copy()
    code = normalize_code(ts_code)
    out = pd.DataFrame(
        {
            "股票代码": code,
            "股票名称": name,
            "市场": market_from_ts_code(ts_code),
            "日期": pd.to_datetime(work["trade_date"]).dt.strftime("%Y-%m-%d"),
            "开盘": pd.to_numeric(work["open"], errors="coerce"),
            "收盘": pd.to_numeric(work["close"], errors="coerce"),
            "最高": pd.to_numeric(work["high"], errors="coerce"),
            "最低": pd.to_numeric(work["low"], errors="coerce"),
            "昨收": pd.to_numeric(work["pre_close"], errors="coerce"),
            "成交量": pd.to_numeric(work["vol"], errors="coerce") * 100.0,
            "成交额": pd.to_numeric(work["amount"], errors="coerce") * 1000.0,
            "换手率": pd.NA,
            "涨跌幅": pd.to_numeric(work["pct_chg"], errors="coerce"),
            "涨跌额": pd.to_numeric(work["change"], errors="coerce"),
            "振幅": (
                (pd.to_numeric(work["high"], errors="coerce") - pd.to_numeric(work["low"], errors="coerce"))
                / pd.to_numeric(work["pre_close"], errors="coerce").abs()
                * 100.0
            ),
            "交易状态": "正常交易",
            "是否ST": 0,
            "市盈率TTM": pd.NA,
            "市净率MRQ": pd.NA,
            "市销率TTM": pd.NA,
            "市现率TTM": pd.NA,
            "复权标记": 1,
            "复权方式": "后复权",
        }
    )
    return out[ORDERED_COLS]


def write_metadata(path: Path, universe: pd.DataFrame, basic: pd.DataFrame) -> None:
    meta = universe[["纯代码", "股票名称", "市场", "ts_code"]].rename(columns={"纯代码": "股票代码"}).copy()
    if not basic.empty:
        keep = basic[["ts_code", "area", "industry", "market", "list_date"]].copy()
        meta = meta.merge(keep, on="ts_code", how="left")
    meta = meta.rename(columns={"area": "地区", "industry": "行业", "market": "板块", "list_date": "上市日期"})
    meta = meta.sort_values(["市场", "股票代码"])
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update CSI300 data with Tushare Pro daily HFQ bars.")
    parser.add_argument("--save-dir", default="./data_submit")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="20260731")
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Tushare token env var: {args.token_env}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    universe = fetch_current_hs300_universe()
    universe_path = save_dir / "hs300_universe.csv"
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")

    ts.set_token(token)
    pro = ts.pro_api()
    basic = fetch_stock_basic(pro)
    write_metadata(save_dir / "stock_metadata.csv", universe, basic)

    name_map = dict(zip(universe["ts_code"], universe["股票名称"]))
    rows: list[pd.DataFrame] = []
    failed: list[dict[str, str]] = []
    for ts_code in tqdm(universe["ts_code"].tolist(), desc="Tushare HS300", unit="stock"):
        try:
            bar = ts.pro_bar(ts_code=ts_code, adj="hfq", start_date=args.start_date, end_date=args.end_date)
            if bar is None or bar.empty:
                failed.append({"ts_code": ts_code, "错误信息": "empty"})
                continue
            rows.append(transform_bar(bar, ts_code, name_map.get(ts_code, "")))
        except Exception as exc:
            failed.append({"ts_code": ts_code, "错误信息": repr(exc)})
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not rows:
        raise RuntimeError("No Tushare bars fetched.")
    daily = pd.concat(rows, ignore_index=True)
    daily = daily.drop_duplicates(["股票代码", "日期"], keep="last").sort_values(["股票代码", "日期"])
    stock_path = save_dir / "stock_data.csv"
    daily.to_csv(stock_path, index=False, encoding="utf-8-sig")

    failed_path = save_dir / "failed_stocks_tushare.csv"
    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False, encoding="utf-8-sig")
    elif failed_path.exists():
        failed_path.unlink()
    print(
        f"wrote {stock_path} rows={len(daily)} stocks={daily['股票代码'].nunique()} "
        f"date={daily['日期'].min()}~{daily['日期'].max()} failed={len(failed)}"
    )
    print(f"wrote {universe_path} rows={len(universe)} update={universe['updateDate'].max()}")


if __name__ == "__main__":
    main()
