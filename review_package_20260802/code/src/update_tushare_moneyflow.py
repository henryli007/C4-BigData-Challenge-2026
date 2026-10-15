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


MONEYFLOW_RENAME = {
    "net_mf_amount": "资金净流入额",
    "buy_lg_amount": "大单买入额",
    "sell_lg_amount": "大单卖出额",
    "buy_elg_amount": "超大单买入额",
    "sell_elg_amount": "超大单卖出额",
    "buy_md_amount": "中单买入额",
    "sell_md_amount": "中单卖出额",
    "buy_sm_amount": "小单买入额",
    "sell_sm_amount": "小单卖出额",
}


def _to_ts_code(code: str) -> str:
    code = str(code).strip()
    if code.endswith((".SH", ".SZ")):
        return code
    pure = code.split(".", 1)[-1] if "." in code else code
    pure = pure.zfill(6)
    suffix = "SH" if pure.startswith(("5", "6", "9")) else "SZ"
    return f"{pure}.{suffix}"


def _normalize_code(code: object) -> str:
    raw = str(code).strip()
    if raw.startswith(("sh.", "sz.")):
        raw = raw.split(".", 1)[1]
    elif raw.endswith((".SH", ".SZ")):
        raw = raw.split(".", 1)[0]
    if "." in raw:
        raw = raw.split(".", 1)[0]
    return raw.zfill(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge TuShare moneyflow fields into stock_data.csv.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
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
        codes = universe[ID_COL].dropna().map(_normalize_code).unique().tolist()
    elif "code" in universe.columns:
        codes = universe["code"].dropna().map(_normalize_code).unique().tolist()
    else:
        codes = sorted(prices[ID_COL].unique().tolist())

    end_date = args.end_date or prices[DATE_COL].max().strftime("%Y%m%d")
    pro = ts.pro_api(token)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for code in tqdm(codes, desc="Tushare moneyflow"):
        try:
            df = pro.moneyflow(ts_code=_to_ts_code(code), start_date=args.start_date, end_date=end_date)
        except Exception as exc:
            failed.append(f"{code}: {exc}")
            time.sleep(max(args.sleep_seconds, 0.5))
            continue
        if df is not None and not df.empty:
            keep = ["ts_code", "trade_date", *MONEYFLOW_RENAME.keys()]
            df = df[[c for c in keep if c in df.columns]].copy()
            df[ID_COL] = df["ts_code"].map(_normalize_code)
            df[DATE_COL] = pd.to_datetime(df["trade_date"])
            df = df.rename(columns=MONEYFLOW_RENAME)
            frames.append(df[[ID_COL, DATE_COL, *MONEYFLOW_RENAME.values()]])
        time.sleep(args.sleep_seconds)

    if not frames:
        raise RuntimeError("No moneyflow rows were fetched.")

    money = pd.concat(frames, ignore_index=True)
    money = money.drop_duplicates([ID_COL, DATE_COL], keep="last")
    for col in MONEYFLOW_RENAME.values():
        if col in money.columns:
            money[col] = pd.to_numeric(money[col], errors="coerce") * 10000.0

    keep_existing = [c for c in MONEYFLOW_RENAME.values() if c in prices.columns]
    if keep_existing:
        prices = prices.drop(columns=keep_existing)
    merged = prices.merge(money, on=[ID_COL, DATE_COL], how="left")
    merged.to_csv(data_path, index=False, encoding="utf-8-sig")

    if failed:
        fail_path = data_path.parent / "moneyflow_failed.txt"
        fail_path.write_text("\n".join(failed), encoding="utf-8")
    print(
        f"wrote {data_path} rows={len(merged)} moneyflow_rows={len(money)} "
        f"stocks={money[ID_COL].nunique()} failed={len(failed)}"
    )


if __name__ == "__main__":
    main()
