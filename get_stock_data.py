#!/usr/bin/env python3
"""
更完整的 A 股历史数据抓取脚本。

默认行为：
- 抓取全 A 股股票日线数据，而不是仅沪深300；
- 使用 BaoStock 稳定接口；
- 关闭代理环境，避免东财/外网代理干扰；
- 输出更丰富的日线字段和独立的股票元数据表；
- 支持断点续抓、增量更新、进度条。

产物：
- data/stock_data.csv: 日线主表
- data/stock_metadata.csv: 股票元数据表
- data/failed_stocks.csv: 失败代码列表
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import baostock as bs
import pandas as pd
from tqdm import tqdm


DEFAULT_DAILY_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "isST",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
]

ADJUSTFLAG_NAME = {
    "1": "后复权",
    "2": "前复权",
    "3": "不复权",
}

DAILY_RENAME_MAP = {
    "date": "日期",
    "code": "股票代码",
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "close": "收盘",
    "preclose": "昨收",
    "volume": "成交量",
    "amount": "成交额",
    "adjustflag": "复权标记",
    "turn": "换手率",
    "tradestatus": "交易状态",
    "pctChg": "涨跌幅",
    "isST": "是否ST",
    "peTTM": "市盈率TTM",
    "pbMRQ": "市净率MRQ",
    "psTTM": "市销率TTM",
    "pcfNcfTTM": "市现率TTM",
}

DAILY_NUMERIC_COLUMNS = [
    "开盘",
    "最高",
    "最低",
    "收盘",
    "昨收",
    "成交量",
    "成交额",
    "换手率",
    "涨跌幅",
    "市盈率TTM",
    "市净率MRQ",
    "市销率TTM",
    "市现率TTM",
]

DAILY_ORDERED_COLS = [
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

SCHEMA_CRITICAL_COLS = [
    "股票名称",
    "市场",
    "昨收",
    "交易状态",
    "是否ST",
    "市盈率TTM",
    "市净率MRQ",
    "市销率TTM",
    "市现率TTM",
    "复权标记",
    "复权方式",
]


@dataclass
class FetchStats:
    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    new_rows: int = 0


@dataclass
class StockFetchPlan:
    ranges: list[tuple[str, str]]
    reason: str


@dataclass
class StockProfile:
    stock_code: str
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None
    has_schema_gap: bool
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取更完整的 A 股历史数据")
    parser.add_argument("--start-date", default="2015-01-01", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="结束日期，格式 YYYY-MM-DD，默认今天",
    )
    parser.add_argument(
        "--latest-trade-day",
        action="store_true",
        help="自动将结束日期设置为最新交易日",
    )
    parser.add_argument(
        "--universe",
        choices=["all", "hs300", "zz500", "sz50"],
        default="all",
        help="股票池，默认全A股",
    )
    parser.add_argument(
        "--adjustflag",
        choices=["1", "2", "3"],
        default="1",
        help="复权方式: 1=后复权, 2=前复权, 3=不复权",
    )
    parser.add_argument("--save-dir", default="./data", help="输出目录")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="每只股票抓取后的暂停秒数，默认 0.05",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="每处理多少只股票落盘一次，默认 50",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="仅抓前 N 只股票，用于测试",
    )
    parser.add_argument(
        "--disable-proxy",
        action="store_true",
        default=True,
        help="抓数时临时关闭代理环境变量，默认开启",
    )
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="如果你明确要保留代理，则传这个参数覆盖 --disable-proxy",
    )
    parser.add_argument(
        "--skip-metadata-refresh",
        action="store_true",
        help="已有 stock_metadata.csv 时复用元数据，只更新股票池文件和日线数据。",
    )
    return parser.parse_args()


def _normalize_date(date_str: str, arg_name: str) -> str:
    ts = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"{arg_name} 日期格式无效: {date_str}")
    return ts.strftime("%Y-%m-%d")


def _a_share_prefixes() -> tuple[str, ...]:
    return (
        "sh.600",
        "sh.601",
        "sh.603",
        "sh.605",
        "sh.688",
        "sz.000",
        "sz.001",
        "sz.002",
        "sz.003",
        "sz.300",
        "sz.301",
        "bj.",
    )


def _normalize_bs_code(bs_code: str) -> str:
    return bs_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").zfill(6)


def _market_from_bs_code(bs_code: str) -> str:
    if bs_code.startswith("sh."):
        return "SH"
    if bs_code.startswith("sz."):
        return "SZ"
    if bs_code.startswith("bj."):
        return "BJ"
    return ""


@contextmanager
def temporary_proxy_disabled(enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return

    proxy_keys = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ]
    backup = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def login() -> None:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")


def logout() -> None:
    bs.logout()


def _query_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def fetch_universe(universe: str, end_date: str) -> pd.DataFrame:
    if universe == "hs300":
        rs = bs.query_hs300_stocks()
        df = _query_to_df(rs)
        if df.empty:
            raise RuntimeError("沪深300 成分股列表为空")
        df = df.rename(columns={"code_name": "股票名称"})
        df["code_name"] = df["股票名称"]
    elif universe == "zz500":
        rs = bs.query_zz500_stocks()
        df = _query_to_df(rs)
        if df.empty:
            raise RuntimeError("中证500 成分股列表为空")
        df = df.rename(columns={"code_name": "股票名称"})
        df["code_name"] = df["股票名称"]
    elif universe == "sz50":
        rs = bs.query_sz50_stocks()
        df = _query_to_df(rs)
        if df.empty:
            raise RuntimeError("上证50 成分股列表为空")
        df = df.rename(columns={"code_name": "股票名称"})
        df["code_name"] = df["股票名称"]
    else:
        df = pd.DataFrame()
        query_date = pd.to_datetime(end_date)
        for _ in range(10):
            day = query_date.strftime("%Y-%m-%d")
            rs = bs.query_all_stock(day=day)
            df = _query_to_df(rs)
            if not df.empty:
                if day != end_date:
                    print(f"全市场证券列表在 {end_date} 为空，已回退使用 {day}")
                break
            query_date -= pd.Timedelta(days=1)
        if df.empty:
            raise RuntimeError(f"全市场证券列表为空，已从 {end_date} 向前回退 10 天仍不可用")
        df = df[df["code"].str.startswith(_a_share_prefixes())].copy()
        df = df[df["tradeStatus"] == "1"].copy()
        df = df.rename(columns={"code_name": "股票名称"})
        df["code_name"] = df["股票名称"]

    df["纯代码"] = df["code"].map(_normalize_bs_code)
    df["市场"] = df["code"].map(_market_from_bs_code)
    df = df.sort_values(["市场", "纯代码"]).reset_index(drop=True)
    return df


def fetch_industry_snapshot(date: str) -> pd.DataFrame:
    rs = bs.query_stock_industry(date=date)
    df = _query_to_df(rs)
    if df.empty:
        return pd.DataFrame(columns=["code", "industry", "industryClassification", "updateDate"])
    df["纯代码"] = df["code"].map(_normalize_bs_code)
    return df


def fetch_latest_trade_day(anchor_end_date: str) -> str:
    start = (pd.to_datetime(anchor_end_date) - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start, end_date=anchor_end_date)
    df = _query_to_df(rs)
    if df.empty:
        raise RuntimeError("无法获取交易日历")
    trade_df = df[df["is_trading_day"] == "1"].copy()
    if trade_df.empty:
        raise RuntimeError("交易日历中没有可用交易日")
    return trade_df["calendar_date"].max()


def fetch_basic_info(bs_code: str) -> dict[str, str]:
    rs = bs.query_stock_basic(code=bs_code)
    df = _query_to_df(rs)
    if df.empty:
        return {
            "code": bs_code,
            "code_name": "",
            "ipoDate": "",
            "outDate": "",
            "type": "",
            "status": "",
        }
    return df.iloc[0].to_dict()


def fetch_history(bs_code: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        ",".join(DEFAULT_DAILY_FIELDS),
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    return _query_to_df(rs)


def transform_history(df: pd.DataFrame, stock_name: str, adjustflag: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out.rename(columns=DAILY_RENAME_MAP)
    out["股票名称"] = stock_name
    out["纯代码"] = out["股票代码"].map(_normalize_bs_code)
    out["市场"] = out["股票代码"].map(_market_from_bs_code)
    out["复权方式"] = ADJUSTFLAG_NAME.get(adjustflag, adjustflag)

    for col in DAILY_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).copy()
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")

    out["振幅"] = ((out["最高"] - out["最低"]) / out["昨收"].replace(0, pd.NA) * 100).astype(float)
    out["涨跌额"] = out["收盘"] - out["昨收"]
    out["交易状态"] = out["交易状态"].map({"1": "正常交易", "0": "停牌"}).fillna(out["交易状态"])
    out["是否ST"] = out["是否ST"].map({"1": 1, "0": 0}).fillna(out["是否ST"])
    out["股票代码"] = out["纯代码"]

    return out[DAILY_ORDERED_COLS].sort_values(["股票代码", "日期"]).reset_index(drop=True)


def merge_daily(existing_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return new_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    merged = merged.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    return merged


def ensure_daily_schema(existing_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if existing_df is None or existing_df.empty:
        return existing_df

    df = existing_df.copy()
    if "股票代码" in df.columns:
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    if "日期" in df.columns:
        dt = pd.to_datetime(df["日期"], errors="coerce")
        df["日期"] = dt.dt.strftime("%Y-%m-%d")

    for col in DAILY_ORDERED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in DAILY_NUMERIC_COLUMNS + ["涨跌额", "振幅"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[DAILY_ORDERED_COLS].sort_values(["股票代码", "日期"]).reset_index(drop=True)


def _stock_existing_slice(existing_df: pd.DataFrame | None, stock_code: str) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return pd.DataFrame(columns=DAILY_ORDERED_COLS)
    stock_df = existing_df[existing_df["股票代码"] == stock_code].copy()
    if stock_df.empty:
        return stock_df
    stock_df["日期"] = pd.to_datetime(stock_df["日期"], errors="coerce")
    stock_df = stock_df.dropna(subset=["日期"]).sort_values("日期")
    return stock_df


def build_stock_profiles(existing_df: pd.DataFrame | None) -> dict[str, StockProfile]:
    if existing_df is None or existing_df.empty:
        return {}

    df = existing_df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    critical_gap = df[SCHEMA_CRITICAL_COLS].isna().any(axis=1)
    grouped = df.groupby("股票代码", sort=False)

    profiles: dict[str, StockProfile] = {}
    for stock_code, group in grouped:
        valid_dates = group["日期"].dropna()
        profiles[stock_code] = StockProfile(
            stock_code=stock_code,
            min_date=valid_dates.min() if not valid_dates.empty else None,
            max_date=valid_dates.max() if not valid_dates.empty else None,
            has_schema_gap=bool(critical_gap.loc[group.index].any()),
            row_count=len(group),
        )
    return profiles


def build_stock_fetch_plan_from_profile(
    profile: StockProfile | None,
    start_date: str,
    end_date: str,
) -> StockFetchPlan:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if profile is None or profile.row_count == 0:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="new")

    if profile.min_date is None or profile.max_date is None:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="missing_window")

    if profile.has_schema_gap:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="schema_backfill")

    have_min = profile.min_date
    have_max = profile.max_date
    if have_max < start_ts or have_min > end_ts:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="missing_window")

    ranges: list[tuple[str, str]] = []
    if have_min > start_ts:
        early_end = have_min - pd.Timedelta(days=1)
        if early_end >= start_ts:
            ranges.append((start_ts.strftime("%Y-%m-%d"), early_end.strftime("%Y-%m-%d")))

    if have_max < end_ts:
        late_start = have_max + pd.Timedelta(days=1)
        if late_start <= end_ts:
            ranges.append((late_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))

    if not ranges:
        return StockFetchPlan(ranges=[], reason="complete")
    return StockFetchPlan(ranges=ranges, reason="incremental")


def build_stock_fetch_plan(
    stock_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> StockFetchPlan:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if stock_df.empty:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="new")

    in_range = stock_df[(stock_df["日期"] >= start_ts) & (stock_df["日期"] <= end_ts)].copy()
    if in_range.empty:
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="missing_window")

    # 如果旧数据缺少新增字段，则对该股票在目标窗口内整段重抓一次，补齐 schema。
    if in_range[SCHEMA_CRITICAL_COLS].isna().any().any():
        return StockFetchPlan(ranges=[(start_date, end_date)], reason="schema_backfill")

    ranges: list[tuple[str, str]] = []
    have_min = in_range["日期"].min()
    have_max = in_range["日期"].max()

    if have_min > start_ts:
        early_end = have_min - pd.Timedelta(days=1)
        if early_end >= start_ts:
            ranges.append((start_ts.strftime("%Y-%m-%d"), early_end.strftime("%Y-%m-%d")))

    if have_max < end_ts:
        late_start = have_max + pd.Timedelta(days=1)
        if late_start <= end_ts:
            ranges.append((late_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))

    if not ranges:
        return StockFetchPlan(ranges=[], reason="complete")
    return StockFetchPlan(ranges=ranges, reason="incremental")


def _plan_priority(reason: str) -> int:
    priority = {
        "schema_backfill": 0,
        "new": 1,
        "missing_window": 2,
        "incremental": 3,
        "complete": 9,
    }
    return priority.get(reason, 8)


def prioritize_universe(
    universe_df: pd.DataFrame,
    profiles: dict[str, StockProfile],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = universe_df.copy()
    work["plan_reason"] = work["纯代码"].map(
        lambda code: build_stock_fetch_plan_from_profile(profiles.get(code), start_date, end_date).reason
    )
    work["plan_priority"] = work["plan_reason"].map(_plan_priority)
    summary = work["plan_reason"].value_counts().to_dict()
    work = work.sort_values(["plan_priority", "市场", "纯代码"]).reset_index(drop=True)
    return work.drop(columns=["plan_priority"]), summary


def write_progress(
    progress_path: Path,
    stats: FetchStats,
    idx: int,
    stock_code: str,
    stock_name: str,
    reason: str,
    file_rows: int,
    file_stocks: int,
) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "processed": idx,
        "total": stats.total,
        "success": stats.success,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "new_rows": stats.new_rows,
        "current_stock_code": stock_code,
        "current_stock_name": stock_name,
        "current_reason": reason,
        "output_rows": file_rows,
        "output_stock_count": file_stocks,
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_metadata(
    universe_df: pd.DataFrame,
    end_date: str,
    checkpoint_every: int,
    sleep_seconds: float,
    output_path: Path,
) -> pd.DataFrame:
    existing_meta = maybe_load_csv(output_path)
    if existing_meta is not None and not existing_meta.empty:
        existing_meta["股票代码"] = existing_meta["股票代码"].astype(str).str.zfill(6)
    else:
        existing_meta = pd.DataFrame(
            columns=["股票代码", "股票名称", "市场", "bs_code", "上市日期", "退市日期", "证券类型", "上市状态", "行业", "行业分类", "行业更新日期"]
        )

    industry_df = fetch_industry_snapshot(end_date)
    industry_map = {}
    if not industry_df.empty:
        industry_map = (
            industry_df.sort_values(["纯代码", "updateDate"])
            .drop_duplicates(subset=["纯代码"], keep="last")
            .set_index("纯代码")[["industry", "industryClassification", "updateDate"]]
            .to_dict(orient="index")
        )

    existing_map = existing_meta.set_index("股票代码").to_dict(orient="index") if not existing_meta.empty else {}
    meta_rows: list[dict[str, str]] = []
    iterator = tqdm(
        universe_df.itertuples(index=False),
        total=len(universe_df),
        desc="抓取股票元数据",
        unit="stock",
    )
    for idx, row in enumerate(iterator, start=1):
        cached = existing_map.get(row.纯代码)
        if cached and str(cached.get("行业更新日期", "")) == industry_map.get(row.纯代码, {}).get("updateDate", ""):
            meta_rows.append({"股票代码": row.纯代码, **cached})
            if idx % checkpoint_every == 0:
                iterator.set_postfix(done=idx)
            continue

        basic = fetch_basic_info(row.code)
        industry = industry_map.get(row.纯代码, {})
        meta_rows.append(
            {
                "股票代码": row.纯代码,
                "股票名称": basic.get("code_name") or row.code_name or "",
                "市场": row.市场,
                "bs_code": row.code,
                "上市日期": basic.get("ipoDate", ""),
                "退市日期": basic.get("outDate", ""),
                "证券类型": basic.get("type", ""),
                "上市状态": basic.get("status", ""),
                "行业": industry.get("industry", ""),
                "行业分类": industry.get("industryClassification", ""),
                "行业更新日期": industry.get("updateDate", ""),
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if idx % checkpoint_every == 0:
            pd.DataFrame(meta_rows).sort_values(["市场", "股票代码"]).to_csv(output_path, index=False, encoding="utf-8-sig")
            iterator.set_postfix(done=idx)

    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.sort_values(["市场", "股票代码"]).reset_index(drop=True)
    meta_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return meta_df


def maybe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"股票代码": str}, low_memory=False)
    if "股票代码" in df.columns:
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    return df


def fetch_all_daily_data(
    universe_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    adjustflag: str,
    output_path: Path,
    progress_path: Path,
    checkpoint_every: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    existing_df = ensure_daily_schema(maybe_load_csv(output_path))
    profiles = build_stock_profiles(existing_df)
    universe_df, summary = prioritize_universe(universe_df, profiles, start_date, end_date)
    failed: list[tuple[str, str, str]] = []
    stats = FetchStats(total=len(universe_df))

    print("抓取优先级统计:", summary)

    iterator = tqdm(
        universe_df.itertuples(index=False),
        total=len(universe_df),
        desc="抓取日线数据",
        unit="stock",
    )
    for idx, row in enumerate(iterator, start=1):
        current_reason = getattr(row, "plan_reason", "")
        try:
            plan = build_stock_fetch_plan_from_profile(profiles.get(row.纯代码), start_date, end_date)
            current_reason = plan.reason

            if not plan.ranges:
                stats.skipped += 1
                iterator.set_postfix(success=stats.success, skipped=stats.skipped, failed=stats.failed, reason=plan.reason)
                if idx == 1 or idx % checkpoint_every == 0:
                    file_rows = len(existing_df) if existing_df is not None else 0
                    file_stocks = existing_df["股票代码"].nunique() if existing_df is not None and not existing_df.empty else 0
                    write_progress(progress_path, stats, idx, row.纯代码, row.code_name, plan.reason, file_rows, file_stocks)
                continue

            parts = []
            fetch_started_at = time.time()
            for fetch_start, fetch_end in plan.ranges:
                raw_df = fetch_history(row.code, fetch_start, fetch_end, adjustflag)
                if raw_df.empty:
                    continue
                parts.append(transform_history(raw_df, row.code_name, adjustflag))

            if not parts:
                stats.skipped += 1
                current_reason = f"{plan.reason}:empty"
                iterator.set_postfix(success=stats.success, skipped=stats.skipped, failed=stats.failed, reason=current_reason)
                if idx == 1 or idx % checkpoint_every == 0:
                    file_rows = len(existing_df) if existing_df is not None else 0
                    file_stocks = existing_df["股票代码"].nunique() if existing_df is not None and not existing_df.empty else 0
                    write_progress(progress_path, stats, idx, row.纯代码, row.code_name, current_reason, file_rows, file_stocks)
                continue

            daily_df = pd.concat(parts, ignore_index=True)
            existing_df = merge_daily(existing_df, daily_df)
            stats.success += 1
            stats.new_rows += len(daily_df)
            profiles[row.纯代码] = StockProfile(
                stock_code=row.纯代码,
                min_date=pd.to_datetime(existing_df.loc[existing_df["股票代码"] == row.纯代码, "日期"]).min(),
                max_date=pd.to_datetime(existing_df.loc[existing_df["股票代码"] == row.纯代码, "日期"]).max(),
                has_schema_gap=False,
                row_count=int((existing_df["股票代码"] == row.纯代码).sum()),
            )

            if idx % checkpoint_every == 0:
                existing_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                write_progress(
                    progress_path,
                    stats,
                    idx,
                    row.纯代码,
                    row.code_name,
                    plan.reason,
                    len(existing_df),
                    existing_df["股票代码"].nunique(),
                )

            elapsed = time.time() - fetch_started_at
            if elapsed >= 10:
                print(
                    f"慢请求告警: {row.纯代码} {row.code_name} | "
                    f"{plan.reason} | {elapsed:.1f}s | ranges={plan.ranges}"
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        except Exception as exc:
            failed.append((row.纯代码, row.code_name, str(exc)))
            stats.failed += 1
            current_reason = f"error:{type(exc).__name__}"

        iterator.set_postfix(success=stats.success, skipped=stats.skipped, failed=stats.failed, new_rows=stats.new_rows)
        if idx == 1 or idx % checkpoint_every == 0:
            file_rows = len(existing_df) if existing_df is not None else 0
            file_stocks = existing_df["股票代码"].nunique() if existing_df is not None and not existing_df.empty else 0
            write_progress(progress_path, stats, idx, row.纯代码, row.code_name, current_reason, file_rows, file_stocks)

    if existing_df is None:
        existing_df = pd.DataFrame()
    existing_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_progress(
        progress_path,
        stats,
        stats.total,
        "",
        "",
        "finished",
        len(existing_df),
        existing_df["股票代码"].nunique() if not existing_df.empty else 0,
    )
    return existing_df, failed


def main() -> None:
    args = parse_args()
    if args.keep_proxy:
        args.disable_proxy = False

    start_date = _normalize_date(args.start_date, "--start-date")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    daily_output_path = save_dir / "stock_data.csv"
    metadata_output_path = save_dir / "stock_metadata.csv"
    failed_output_path = save_dir / "failed_stocks.csv"
    progress_output_path = save_dir / "fetch_progress.json"
    universe_output_path = save_dir / f"{args.universe}_universe.csv"

    with temporary_proxy_disabled(args.disable_proxy):
        login()
        try:
            end_date = _normalize_date(args.end_date, "--end-date")
            if args.latest_trade_day:
                end_date = fetch_latest_trade_day(end_date)
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                raise ValueError(f"开始日期晚于结束日期: {start_date} > {end_date}")

            print(f"股票池: {args.universe}")
            print(f"时间范围: {start_date} ~ {end_date}")
            print(f"复权方式: {ADJUSTFLAG_NAME.get(args.adjustflag, args.adjustflag)}")
            print(f"输出目录: {save_dir.resolve()}")
            print(f"关闭代理: {args.disable_proxy}")
            print("=" * 72)

            universe_df = fetch_universe(args.universe, end_date)
            if args.max_stocks is not None:
                universe_df = universe_df.head(args.max_stocks).copy()
            universe_df.to_csv(universe_output_path, index=False, encoding="utf-8-sig")

            print(f"股票数量: {len(universe_df)}")
            if args.skip_metadata_refresh and metadata_output_path.exists():
                print(f"复用已有元数据: {metadata_output_path}")
                metadata_df = pd.read_csv(metadata_output_path, dtype={"股票代码": str})
            else:
                print("开始抓取元数据...")
                metadata_df = build_metadata(
                    universe_df=universe_df,
                    end_date=end_date,
                    checkpoint_every=args.checkpoint_every,
                    sleep_seconds=args.sleep_seconds,
                    output_path=metadata_output_path,
                )

            print("开始抓取日线主表...")
            daily_df, failed = fetch_all_daily_data(
                universe_df=universe_df,
                start_date=start_date,
                end_date=end_date,
                adjustflag=args.adjustflag,
                output_path=daily_output_path,
                progress_path=progress_output_path,
                checkpoint_every=args.checkpoint_every,
                sleep_seconds=args.sleep_seconds,
            )

            if failed:
                pd.DataFrame(failed, columns=["股票代码", "股票名称", "错误信息"]).to_csv(
                    failed_output_path,
                    index=False,
                    encoding="utf-8-sig",
                )
            elif failed_output_path.exists():
                failed_output_path.unlink()

        finally:
            logout()

    print("=" * 72)
    print("抓取完成")
    if metadata_output_path.exists():
        meta_df = pd.read_csv(metadata_output_path, dtype={"股票代码": str})
        print(f"元数据表: {metadata_output_path} | {len(meta_df)} 行")
    if daily_output_path.exists():
        daily_df = pd.read_csv(daily_output_path, dtype={"股票代码": str})
        print(
            f"日线主表: {daily_output_path} | {len(daily_df)} 行 | "
            f"{daily_df['股票代码'].nunique()} 只股票 | "
            f"{daily_df['日期'].min()} ~ {daily_df['日期'].max()}"
        )
    if universe_output_path.exists():
        print(f"股票池清单: {universe_output_path}")
    if failed_output_path.exists():
        failed_df = pd.read_csv(failed_output_path, dtype={"股票代码": str})
        print(f"失败列表: {failed_output_path} | {len(failed_df)} 行")


if __name__ == "__main__":
    main()
