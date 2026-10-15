from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts

from advanced.features import DATE_COL, ID_COL, load_price_data, normalize_stock_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch TuShare ths_hot stock and board data for short-term theme features.")
    parser.add_argument("--data", default="./data_submit/stock_data.csv")
    parser.add_argument("--output-dir", default="./model/extra_data")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--sleep", type=float, default=0.12)
    return parser.parse_args()


def ts_code_to_plain(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return normalize_stock_code(text)


def main() -> None:
    args = parse_args()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_price_data(args.data)
    dates = sorted(pd.to_datetime(raw[DATE_COL].drop_duplicates()))
    start = pd.Timestamp(args.start_date) if args.start_date else dates[0]
    end = pd.Timestamp(args.end_date) if args.end_date else dates[-1]
    dates = [d for d in dates if start <= d <= end]

    pro = ts.pro_api(token)
    stock_parts: list[pd.DataFrame] = []
    board_parts: list[pd.DataFrame] = []
    failed: list[dict[str, str]] = []
    for i, date in enumerate(dates, start=1):
        trade_date = date.strftime("%Y%m%d")
        for market in ("热股", "概念板块", "行业板块"):
            try:
                df = pro.ths_hot(trade_date=trade_date, market=market, is_new="Y")
            except Exception as exc:
                failed.append({"trade_date": trade_date, "market": market, "error": str(exc)})
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df[DATE_COL] = pd.to_datetime(df["trade_date"])
            if market == "热股":
                stock_parts.append(df)
            else:
                board_parts.append(df)
            time.sleep(args.sleep)
        if i % 20 == 0:
            print(f"fetched {i}/{len(dates)} through {trade_date}; stock_parts={len(stock_parts)} board_parts={len(board_parts)}")

    stock = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame()
    if not stock.empty:
        stock[ID_COL] = stock["ts_code"].map(ts_code_to_plain)
        stock = stock.rename(
            columns={
                "rank": "ths_hot_rank",
                "pct_change": "ths_hot_pct_change",
                "hot": "ths_hot_value",
                "concept": "ths_hot_concept",
                "rank_reason": "ths_hot_reason",
            }
        )
        keep = [
            DATE_COL,
            ID_COL,
            "ths_hot_rank",
            "ths_hot_pct_change",
            "ths_hot_value",
            "ths_hot_concept",
            "ths_hot_reason",
            "rank_time",
        ]
        stock = stock[[c for c in keep if c in stock.columns]].drop_duplicates([DATE_COL, ID_COL], keep="last")
        stock.to_csv(out_dir / "ths_hot_stock.csv", index=False, encoding="utf-8-sig")

    board = pd.concat(board_parts, ignore_index=True) if board_parts else pd.DataFrame()
    if not board.empty:
        board = board.rename(
            columns={
                "data_type": "ths_board_type",
                "ts_name": "ths_board_name",
                "rank": "ths_board_rank",
                "pct_change": "ths_board_pct_change",
                "hot": "ths_board_hot",
            }
        )
        keep = [DATE_COL, "ths_board_type", "ts_code", "ths_board_name", "ths_board_rank", "ths_board_pct_change", "ths_board_hot", "rank_time"]
        board = board[[c for c in keep if c in board.columns]].drop_duplicates(
            [DATE_COL, "ths_board_type", "ths_board_name"], keep="last"
        )
        board.to_csv(out_dir / "ths_hot_board.csv", index=False, encoding="utf-8-sig")

    (out_dir / "ths_hot_fetch_report.json").write_text(
        json.dumps(
            {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "dates": len(dates),
                "stock_rows": int(len(stock)),
                "board_rows": int(len(board)),
                "failed": failed[:50],
                "failed_count": len(failed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"stock_rows={len(stock)} board_rows={len(board)} failed={len(failed)} wrote {out_dir}")


if __name__ == "__main__":
    main()
