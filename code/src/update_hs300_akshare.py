from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path

import akshare as ak
import pandas as pd
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


@contextmanager
def maybe_proxy_disabled(enabled: bool) -> None:
    if not enabled:
        yield
        return
    keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
    backup = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def normalize_code(value: object) -> str:
    return str(value).strip().zfill(6)


def market_for_code(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    return "SZ"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ORDERED_COLS)
    df = pd.read_csv(path, dtype={"股票代码": str}, low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    df["股票代码"] = df["股票代码"].map(normalize_code)
    for col in ORDERED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ORDERED_COLS].copy()


def fetch_hs300_universe() -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol="000300")
    if df.empty:
        raise RuntimeError("AKShare returned empty CSI300 universe.")
    out = pd.DataFrame(
        {
            "updateDate": pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d"),
            "code": df["成分券代码"].map(lambda x: f"{market_for_code(normalize_code(x)).lower()}.{normalize_code(x)}"),
            "股票名称": df["成分券名称"],
            "code_name": df["成分券名称"],
            "纯代码": df["成分券代码"].map(normalize_code),
            "市场": df["成分券代码"].map(lambda x: market_for_code(normalize_code(x))),
        }
    )
    return out.sort_values(["市场", "纯代码"]).reset_index(drop=True)


def transform_hist(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ORDERED_COLS)
    out = raw.copy()
    if "date" in out.columns:
        out = out.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "close": "收盘",
                "high": "最高",
                "low": "最低",
                "volume": "成交量",
                "amount": "成交额",
                "turnover": "换手率",
            }
        )
    if "股票代码" not in out.columns:
        out["股票代码"] = code
    out["股票代码"] = out["股票代码"].map(normalize_code)
    out["股票名称"] = name
    out["市场"] = market_for_code(code)
    out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values("日期").reset_index(drop=True)
    out["昨收"] = out["收盘"].shift(1)
    out["涨跌幅"] = pd.to_numeric(out["收盘"], errors="coerce").pct_change() * 100.0
    out["涨跌额"] = pd.to_numeric(out["收盘"], errors="coerce") - pd.to_numeric(out["昨收"], errors="coerce")
    out["振幅"] = (
        (
            pd.to_numeric(out["最高"], errors="coerce")
            - pd.to_numeric(out["最低"], errors="coerce")
        )
        / pd.to_numeric(out["昨收"], errors="coerce").abs()
        * 100.0
    )
    out["换手率"] = pd.to_numeric(out["换手率"], errors="coerce") * 100.0
    out["交易状态"] = "正常交易"
    out["是否ST"] = 0
    out["市盈率TTM"] = pd.NA
    out["市净率MRQ"] = pd.NA
    out["市销率TTM"] = pd.NA
    out["市现率TTM"] = pd.NA
    out["复权标记"] = 1
    out["复权方式"] = "后复权"
    for col in ORDERED_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[ORDERED_COLS]


def fetch_hist_with_retries(code: str, start_date: str, end_date: str, timeout: float) -> pd.DataFrame:
    last_exc: Exception | None = None
    symbol = f"{market_for_code(code).lower()}{code}"
    for attempt in range(3):
        try:
            return ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq",
                timeout=timeout,
            )
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 + attempt * 0.5)
    if last_exc is None:
        raise RuntimeError("unknown AKShare history fetch failure")
    raise last_exc


def merge_rows(existing: pd.DataFrame, new_rows: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [existing, *[x for x in new_rows if not x.empty]]
    merged = pd.concat(parts, ignore_index=True)
    merged["股票代码"] = merged["股票代码"].map(normalize_code)
    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.dropna(subset=["日期"])
    merged = merged.drop_duplicates(["股票代码", "日期"], keep="last")
    return merged[ORDERED_COLS].sort_values(["股票代码", "日期"]).reset_index(drop=True)


def update_metadata(meta_path: Path, universe: pd.DataFrame) -> None:
    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype={"股票代码": str}, low_memory=False)
        meta.columns = [str(c).replace("\ufeff", "").strip() for c in meta.columns]
        meta["股票代码"] = meta["股票代码"].map(normalize_code)
    else:
        meta = pd.DataFrame(columns=["股票代码", "股票名称", "市场"])
    base = universe.rename(columns={"纯代码": "股票代码"})[["股票代码", "股票名称", "市场"]]
    merged = pd.concat([meta, base], ignore_index=True)
    merged["股票代码"] = merged["股票代码"].map(normalize_code)
    merged = merged.drop_duplicates("股票代码", keep="first").sort_values(["市场", "股票代码"])
    merged.to_csv(meta_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update current CSI300 daily data via AKShare.")
    parser.add_argument("--save-dir", default="./data_submit")
    parser.add_argument("--start-date", default="20260701", help="YYYYMMDD")
    parser.add_argument("--end-date", default="20260801", help="YYYYMMDD")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--disable-proxy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stock_path = save_dir / "stock_data.csv"
    universe_path = save_dir / "hs300_universe.csv"
    meta_path = save_dir / "stock_metadata.csv"

    with maybe_proxy_disabled(args.disable_proxy):
        print("fetching CSI300 universe...", flush=True)
        universe = fetch_hs300_universe()
        universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
        update_metadata(meta_path, universe)
        existing = load_csv(stock_path)
        existing_max = existing.groupby("股票代码")["日期"].max().to_dict() if not existing.empty else {}
        rows: list[pd.DataFrame] = []
        failed: list[dict[str, str]] = []
        print(f"fetching daily bars for {len(universe)} stocks...", flush=True)
        for item in tqdm(universe.itertuples(index=False), total=len(universe), desc="AKShare HS300"):
            code = normalize_code(item.纯代码)
            start = args.start_date
            max_date = existing_max.get(code)
            if max_date and str(max_date) >= "2026-07-01":
                next_day = pd.Timestamp(max_date) + pd.Timedelta(days=1)
                start = next_day.strftime("%Y%m%d")
            if start > args.end_date:
                continue
            try:
                raw = fetch_hist_with_retries(code, start, args.end_date, args.timeout)
                rows.append(transform_hist(raw, code, item.股票名称))
            except Exception as exc:
                failed.append({"股票代码": code, "股票名称": item.股票名称, "错误信息": repr(exc)})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    merged = merge_rows(existing, rows)
    merged.to_csv(stock_path, index=False, encoding="utf-8-sig")
    failed_path = save_dir / "failed_stocks_akshare.csv"
    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False, encoding="utf-8-sig")
    elif failed_path.exists():
        failed_path.unlink()
    print(
        f"wrote {stock_path} rows={len(merged)} stocks={merged['股票代码'].nunique()} "
        f"date={merged['日期'].min()}~{merged['日期'].max()} failed={len(failed)}"
    )
    print(f"wrote {universe_path} rows={len(universe)} update={universe['updateDate'].max()}")


if __name__ == "__main__":
    main()
