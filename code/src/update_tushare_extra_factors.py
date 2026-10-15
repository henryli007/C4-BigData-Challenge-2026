from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts

from advanced.features import DATE_COL, ID_COL, load_price_data, load_universe_codes, normalize_stock_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch sparse event and technical TuShare factors.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output-dir", default="./model/extra_data")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--skip-stk-factor", action="store_true")
    parser.add_argument("--only-stk-factor", action="store_true")
    return parser.parse_args()


def plain_to_ts(code: str) -> str:
    code = normalize_stock_code(code)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


def ts_to_plain(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return normalize_stock_code(text)


def fetch_by_date(pro, dates: list[pd.Timestamp], api_name: str, sleep: float) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    parts = []
    failed = []
    api = getattr(pro, api_name)
    for i, date in enumerate(dates, start=1):
        trade_date = date.strftime("%Y%m%d")
        try:
            df = api(trade_date=trade_date)
            if df is not None and not df.empty:
                parts.append(df.copy())
        except Exception as exc:
            failed.append({"api": api_name, "trade_date": trade_date, "error": str(exc)})
        time.sleep(sleep)
        if i % 50 == 0:
            print(f"{api_name}: fetched {i}/{len(dates)}")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), failed


def main() -> None:
    args = parse_args()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_price_data(args.data)
    codes = sorted(load_universe_codes(args.universe) or set(raw[ID_COL].map(normalize_stock_code).unique()))
    dates = sorted(pd.to_datetime(raw[DATE_COL].drop_duplicates()))
    start = pd.Timestamp(args.start_date) if args.start_date else dates[0]
    end = pd.Timestamp(args.end_date) if args.end_date else dates[-1]
    dates = [d for d in dates if start <= d <= end]
    date_strings = {d.strftime("%Y%m%d") for d in dates}
    code_set = set(codes)

    pro = ts.pro_api(token)
    report: dict[str, object] = {"start_date": start.date().isoformat(), "end_date": end.date().isoformat()}
    failures: list[dict[str, str]] = []

    limit_list = pd.DataFrame()
    if not args.only_stk_factor:
        limit_list, failed = fetch_by_date(pro, dates, "limit_list_d", args.sleep)
        failures.extend(failed)
    if not limit_list.empty:
        limit_list[DATE_COL] = pd.to_datetime(limit_list["trade_date"])
        limit_list[ID_COL] = limit_list["ts_code"].map(ts_to_plain)
        limit_list = limit_list[limit_list[ID_COL].isin(code_set)].copy()
        limit_list = limit_list.rename(
            columns={
                "industry": "limit_industry",
                "pct_chg": "limit_pct_chg",
                "amount": "limit_amount_turnover",
                "float_mv": "limit_float_mv",
                "turnover_ratio": "limit_turnover_ratio",
                "fd_amount": "limit_seal_amount",
                "open_times": "limit_open_times",
                "limit_times": "limit_times",
                "limit": "limit_direction",
                "up_stat": "limit_up_stat",
            }
        )
        keep = [
            DATE_COL,
            ID_COL,
            "limit_pct_chg",
            "limit_amount_turnover",
            "limit_float_mv",
            "limit_turnover_ratio",
            "limit_seal_amount",
            "limit_open_times",
            "limit_times",
            "limit_direction",
            "limit_up_stat",
        ]
        limit_list[[c for c in keep if c in limit_list.columns]].to_csv(
            out_dir / "limit_list_d.csv", index=False, encoding="utf-8-sig"
        )
    report["limit_list_rows"] = int(len(limit_list))

    top_list = pd.DataFrame()
    if not args.only_stk_factor:
        top_list, failed = fetch_by_date(pro, dates, "top_list", args.sleep)
        failures.extend(failed)
    if not top_list.empty:
        top_list[DATE_COL] = pd.to_datetime(top_list["trade_date"])
        top_list[ID_COL] = top_list["ts_code"].map(ts_to_plain)
        top_list = top_list[top_list[ID_COL].isin(code_set)].copy()
        top_list = top_list.rename(
            columns={
                "pct_change": "top_pct_change",
                "turnover_rate": "top_turnover_rate",
                "amount": "top_amount",
                "l_sell": "top_l_sell",
                "l_buy": "top_l_buy",
                "l_amount": "top_l_amount",
                "net_amount": "top_net_amount",
                "net_rate": "top_net_rate",
                "amount_rate": "top_amount_rate",
                "float_values": "top_float_values",
                "reason": "top_reason",
            }
        )
        keep = [
            DATE_COL,
            ID_COL,
            "top_pct_change",
            "top_turnover_rate",
            "top_amount",
            "top_l_sell",
            "top_l_buy",
            "top_l_amount",
            "top_net_amount",
            "top_net_rate",
            "top_amount_rate",
            "top_float_values",
            "top_reason",
        ]
        top_list[[c for c in keep if c in top_list.columns]].drop_duplicates([DATE_COL, ID_COL], keep="last").to_csv(
            out_dir / "top_list.csv", index=False, encoding="utf-8-sig"
        )
    report["top_list_rows"] = int(len(top_list))

    margin_detail = pd.DataFrame()
    if not args.only_stk_factor:
        margin_detail, failed = fetch_by_date(pro, dates, "margin_detail", args.sleep)
        failures.extend(failed)
    if not margin_detail.empty:
        margin_detail[DATE_COL] = pd.to_datetime(margin_detail["trade_date"])
        margin_detail[ID_COL] = margin_detail["ts_code"].map(ts_to_plain)
        margin_detail = margin_detail[margin_detail[ID_COL].isin(code_set)].copy()
        margin_detail = margin_detail.rename(
            columns={
                "rzye": "margin_fin_balance",
                "rqye": "margin_sec_balance",
                "rzmre": "margin_fin_buy",
                "rqyl": "margin_sec_left",
                "rzche": "margin_fin_repay",
                "rqchl": "margin_sec_repay",
                "rqmcl": "margin_sec_sell",
                "rzrqye": "margin_total_balance",
            }
        )
        keep = [
            DATE_COL,
            ID_COL,
            "margin_fin_balance",
            "margin_sec_balance",
            "margin_fin_buy",
            "margin_sec_left",
            "margin_fin_repay",
            "margin_sec_repay",
            "margin_sec_sell",
            "margin_total_balance",
        ]
        margin_detail[[c for c in keep if c in margin_detail.columns]].drop_duplicates([DATE_COL, ID_COL], keep="last").to_csv(
            out_dir / "margin_detail.csv", index=False, encoding="utf-8-sig"
        )
    report["margin_detail_rows"] = int(len(margin_detail))

    factor_parts = []
    factor_failed = []
    if not args.skip_stk_factor:
        factor_path = out_dir / "stk_factor.csv"
        done_codes: set[str] = set()
        if factor_path.exists():
            old_factor = pd.read_csv(factor_path, dtype={ID_COL: str}, encoding="utf-8-sig")
            if ID_COL in old_factor.columns:
                done_codes = set(old_factor[ID_COL].map(normalize_stock_code).dropna().unique())
                factor_parts.append(old_factor)
        pending_codes = [code for code in codes if code not in done_codes]
        for i, code in enumerate(pending_codes, start=1):
            try:
                df = pro.stk_factor(ts_code=plain_to_ts(code), start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
                if df is not None and not df.empty:
                    factor_parts.append(df.copy())
            except Exception as exc:
                factor_failed.append({"api": "stk_factor", "ts_code": plain_to_ts(code), "error": str(exc)})
            time.sleep(args.sleep)
            if i % 25 == 0:
                print(f"stk_factor: fetched {i}/{len(pending_codes)} pending, existing_codes={len(done_codes)}")
    failures.extend(factor_failed)
    factor = pd.concat(factor_parts, ignore_index=True) if factor_parts else pd.DataFrame()
    if not factor.empty:
        factor = factor[factor["trade_date"].isin(date_strings)].copy()
        factor[DATE_COL] = pd.to_datetime(factor["trade_date"])
        factor[ID_COL] = factor["ts_code"].map(ts_to_plain)
        factor = factor[factor[ID_COL].isin(code_set)].copy()
        rename = {c: f"tsf_{c}" for c in ["macd_dif", "macd_dea", "macd", "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "cci"]}
        factor = factor.rename(columns=rename)
        keep = [DATE_COL, ID_COL, *rename.values()]
        factor[[c for c in keep if c in factor.columns]].drop_duplicates([DATE_COL, ID_COL], keep="last").to_csv(
            out_dir / "stk_factor.csv", index=False, encoding="utf-8-sig"
        )
    report["stk_factor_rows"] = int(len(factor))
    report["failed_count"] = len(failures)
    report["failed"] = failures[:100]
    (out_dir / "extra_factor_fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
