from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts

from advanced.features import DATE_COL, ID_COL, load_universe_codes, normalize_stock_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map TuShare THS hot boards to stock-level daily features.")
    parser.add_argument("--board-hot", default="./model/extra_apr_data/ths_hot_board.csv")
    parser.add_argument("--universe", default="./data_submit/hs300_universe.csv")
    parser.add_argument("--output", default="./model/extra_apr_event_data/ths_board_stock_hot.csv")
    parser.add_argument("--member-cache", default="./model/extra_apr_data/ths_member_cache.csv")
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def ts_to_plain(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return normalize_stock_code(text)


def main() -> None:
    args = parse_args()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")

    board = pd.read_csv(args.board_hot, dtype=str, encoding="utf-8-sig")
    board.columns = [str(c).replace("\ufeff", "").strip() for c in board.columns]
    board[DATE_COL] = pd.to_datetime(board[DATE_COL])
    for col in ["ths_board_rank", "ths_board_pct_change", "ths_board_hot"]:
        board[col] = pd.to_numeric(board[col], errors="coerce")
    board_codes = sorted(board["ts_code"].dropna().unique())
    universe = load_universe_codes(args.universe) or set()

    cache_path = Path(args.member_cache)
    member_parts = []
    done = set()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype=str, encoding="utf-8-sig")
        cached.columns = [str(c).replace("\ufeff", "").strip() for c in cached.columns]
        if not cached.empty:
            member_parts.append(cached)
            done = set(cached["ts_code"].dropna().unique())
    pro = ts.pro_api(token)
    pending = [code for code in board_codes if code not in done]
    failed = []
    for i, code in enumerate(pending, start=1):
        try:
            df = pro.ths_member(ts_code=code)
            if df is not None and not df.empty:
                member_parts.append(df.copy())
        except Exception as exc:
            failed.append({"ts_code": code, "error": str(exc)})
        time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"ths_member fetched {i}/{len(pending)}")

    members = pd.concat(member_parts, ignore_index=True) if member_parts else pd.DataFrame(columns=["ts_code", "con_code", "con_name"])
    members = members.drop_duplicates(["ts_code", "con_code"], keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(cache_path, index=False, encoding="utf-8-sig")
    members[ID_COL] = members["con_code"].map(ts_to_plain)
    if universe:
        members = members[members[ID_COL].isin(universe)].copy()

    joined = board.merge(members[["ts_code", ID_COL]], on="ts_code", how="inner")
    joined["board_rank_score"] = (21.0 - joined["ths_board_rank"].clip(lower=1, upper=20)) / 20.0
    joined["board_hot_log"] = joined["ths_board_hot"].clip(lower=0).add(1.0).apply(lambda x: __import__("math").log(x))

    grouped = joined.groupby([DATE_COL, ID_COL], sort=False)
    out = grouped.agg(
        ths_board_hot_count=("ts_code", "nunique"),
        ths_board_best_rank_score=("board_rank_score", "max"),
        ths_board_rank_score_sum=("board_rank_score", "sum"),
        ths_board_hot_log_max=("board_hot_log", "max"),
        ths_board_hot_log_sum=("board_hot_log", "sum"),
        ths_board_pct_change_max=("ths_board_pct_change", "max"),
        ths_board_pct_change_mean=("ths_board_pct_change", "mean"),
    ).reset_index()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    report = {
        "board_codes": len(board_codes),
        "member_rows": len(members),
        "output_rows": len(out),
        "failed_count": len(failed),
        "failed": failed[:50],
    }
    (out_path.parent / "ths_board_stock_hot_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
