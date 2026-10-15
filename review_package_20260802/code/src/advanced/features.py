from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "股票代码"
DATE_COL = "日期"
OPEN_COL = "开盘"
CLOSE_COL = "收盘"
HIGH_COL = "最高"
LOW_COL = "最低"
VOLUME_COL = "成交量"
AMOUNT_COL = "成交额"
TURNOVER_COL = "换手率"
AMPLITUDE_COL = "振幅"
PCT_COL = "涨跌幅"
LABEL_COL = "label_return"
FREE_TURNOVER_COL = "自由流通换手率"
VOLUME_RATIO_COL = "量比"
PE_TTM_COL = "市盈率TTM"
PB_COL = "市净率MRQ"
PS_TTM_COL = "市销率TTM"
TOTAL_MV_COL = "总市值"
CIRC_MV_COL = "流通市值"
UP_LIMIT_COL = "涨停价"
DOWN_LIMIT_COL = "跌停价"
NET_MF_AMOUNT_COL = "资金净流入额"
BUY_LG_AMOUNT_COL = "大单买入额"
SELL_LG_AMOUNT_COL = "大单卖出额"
BUY_ELG_AMOUNT_COL = "超大单买入额"
SELL_ELG_AMOUNT_COL = "超大单卖出额"
THS_HOT_RANK_COL = "ths_hot_rank"
THS_HOT_PCT_CHANGE_COL = "ths_hot_pct_change"
THS_HOT_VALUE_COL = "ths_hot_value"
LIMIT_DIRECTION_COL = "limit_direction"
TOP_NET_AMOUNT_COL = "top_net_amount"
MARGIN_FIN_BALANCE_COL = "margin_fin_balance"
MARGIN_TOTAL_BALANCE_COL = "margin_total_balance"
TSF_PREFIX = "tsf_"
BOARD_HOT_PREFIX = "ths_board_"


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (3, 5, 10, 20, 30, 60, 120)
    min_history: int = 60
    horizon: int = 5
    top_k: int = 5
    add_cross_sectional: bool = True
    add_industry_relative: bool = True
    add_industry_aggregate: bool = False
    add_event_state: bool = False
    add_event_text: bool = False


BASE_COLUMNS = {
    ID_COL,
    DATE_COL,
    OPEN_COL,
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    VOLUME_COL,
    AMOUNT_COL,
    TURNOVER_COL,
    AMPLITUDE_COL,
    PCT_COL,
    "股票名称",
    "市场",
    "昨收",
    "交易状态",
    "是否ST",
    "复权标记",
    "复权方式",
    FREE_TURNOVER_COL,
    VOLUME_RATIO_COL,
    PE_TTM_COL,
    PB_COL,
    PS_TTM_COL,
    TOTAL_MV_COL,
    CIRC_MV_COL,
    UP_LIMIT_COL,
    DOWN_LIMIT_COL,
    NET_MF_AMOUNT_COL,
    BUY_LG_AMOUNT_COL,
    SELL_LG_AMOUNT_COL,
    BUY_ELG_AMOUNT_COL,
    SELL_ELG_AMOUNT_COL,
    THS_HOT_RANK_COL,
    THS_HOT_PCT_CHANGE_COL,
    THS_HOT_VALUE_COL,
    "ths_hot_concept",
    "ths_hot_reason",
    "rank_time",
    LIMIT_DIRECTION_COL,
    "limit_pct_chg",
    "limit_amount_turnover",
    "limit_float_mv",
    "limit_turnover_ratio",
    "limit_seal_amount",
    "limit_open_times",
    "limit_times",
    "limit_up_stat",
    "top_pct_change",
    "top_turnover_rate",
    "top_amount",
    "top_l_sell",
    "top_l_buy",
    "top_l_amount",
    TOP_NET_AMOUNT_COL,
    "top_net_rate",
    "top_amount_rate",
    "top_float_values",
    "top_reason",
    MARGIN_FIN_BALANCE_COL,
    "margin_sec_balance",
    "margin_fin_buy",
    "margin_sec_left",
    "margin_fin_repay",
    "margin_sec_repay",
    "margin_sec_sell",
    MARGIN_TOTAL_BALANCE_COL,
    "ths_board_hot_count",
    "ths_board_best_rank_score",
    "ths_board_rank_score_sum",
    "ths_board_hot_log_max",
    "ths_board_hot_log_sum",
    "ths_board_pct_change_max",
    "ths_board_pct_change_mean",
}


def normalize_stock_code(value: object) -> str:
    code = str(value).strip()
    if "." in code and code.replace(".", "", 1).isdigit():
        code = code.split(".", 1)[0]
    if code.startswith(("sh.", "sz.")):
        code = code.split(".", 1)[1]
    return code.zfill(6)


def load_price_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={ID_COL: str}, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if ID_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"{path} must contain {ID_COL} and {DATE_COL}")

    df[ID_COL] = df[ID_COL].map(normalize_stock_code)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    for col in [OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, AMOUNT_COL, TURNOVER_COL, AMPLITUDE_COL, PCT_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        FREE_TURNOVER_COL,
        VOLUME_RATIO_COL,
        PE_TTM_COL,
        PB_COL,
        PS_TTM_COL,
        TOTAL_MV_COL,
        CIRC_MV_COL,
        UP_LIMIT_COL,
        DOWN_LIMIT_COL,
        NET_MF_AMOUNT_COL,
        BUY_LG_AMOUNT_COL,
        SELL_LG_AMOUNT_COL,
        BUY_ELG_AMOUNT_COL,
        SELL_ELG_AMOUNT_COL,
        THS_HOT_RANK_COL,
        THS_HOT_PCT_CHANGE_COL,
        THS_HOT_VALUE_COL,
        "limit_pct_chg",
        "limit_amount_turnover",
        "limit_float_mv",
        "limit_turnover_ratio",
        "limit_seal_amount",
        "limit_open_times",
        "limit_times",
        "top_pct_change",
        "top_turnover_rate",
        "top_amount",
        "top_l_sell",
        "top_l_buy",
        "top_l_amount",
        TOP_NET_AMOUNT_COL,
        "top_net_rate",
        "top_amount_rate",
        "top_float_values",
        MARGIN_FIN_BALANCE_COL,
        "margin_sec_balance",
        "margin_fin_buy",
        "margin_sec_left",
        "margin_fin_repay",
        "margin_sec_repay",
        "margin_sec_sell",
        MARGIN_TOTAL_BALANCE_COL,
        "ths_board_hot_count",
        "ths_board_best_rank_score",
        "ths_board_rank_score_sum",
        "ths_board_hot_log_max",
        "ths_board_hot_log_sum",
        "ths_board_pct_change_max",
        "ths_board_pct_change_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in df.columns if c.startswith(TSF_PREFIX)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, AMOUNT_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if TURNOVER_COL not in df.columns:
        df[TURNOVER_COL] = 0.0
    if AMPLITUDE_COL not in df.columns:
        df[AMPLITUDE_COL] = (df[HIGH_COL] - df[LOW_COL]) / (df[OPEN_COL].abs() + 1e-12) * 100.0
    if PCT_COL not in df.columns:
        df[PCT_COL] = df.groupby(ID_COL)[CLOSE_COL].pct_change() * 100.0

    df = df.sort_values([ID_COL, DATE_COL]).drop_duplicates([ID_COL, DATE_COL], keep="last")
    return df.reset_index(drop=True)


def load_universe_codes(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if ID_COL in df.columns:
        raw_codes = df[ID_COL]
    elif "code" in df.columns:
        raw_codes = df["code"]
    elif "纯代码" in df.columns:
        raw_codes = df["纯代码"]
    else:
        raise ValueError(f"Universe file {path} must contain 股票代码, code, or 纯代码")
    return {normalize_stock_code(code) for code in raw_codes.dropna()}


def filter_universe(df: pd.DataFrame, universe_codes: set[str] | None) -> pd.DataFrame:
    if not universe_codes:
        return df.copy()
    return df[df[ID_COL].map(normalize_stock_code).isin(universe_codes)].copy()


def add_market_context(price_df: pd.DataFrame) -> pd.DataFrame:
    """Use the full A-share panel to add daily market context before target filtering."""
    work = price_df.copy()
    work[ID_COL] = work[ID_COL].map(normalize_stock_code)
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    ret = work.groupby(ID_COL, sort=False)[CLOSE_COL].pct_change()
    work["_market_ret_1"] = ret
    by_date = work.groupby(DATE_COL, sort=False)
    context = pd.DataFrame(
        {
            DATE_COL: by_date.size().index,
            "f_mkt_ret_mean": by_date["_market_ret_1"].mean().to_numpy(),
            "f_mkt_ret_median": by_date["_market_ret_1"].median().to_numpy(),
            "f_mkt_ret_std": by_date["_market_ret_1"].std().to_numpy(),
            "f_mkt_up_ratio": by_date["_market_ret_1"].apply(lambda s: (s > 0).mean()).to_numpy(),
            "f_mkt_amount_log": np.log1p(by_date[AMOUNT_COL].sum()).to_numpy(),
            "f_mkt_volume_log": np.log1p(by_date[VOLUME_COL].sum()).to_numpy(),
        }
    )
    for w in (5, 20, 60):
        context[f"f_mkt_ret_mean_ma_{w}"] = context["f_mkt_ret_mean"].rolling(w, min_periods=2).mean()
        context[f"f_mkt_up_ratio_ma_{w}"] = context["f_mkt_up_ratio"].rolling(w, min_periods=2).mean()
        context[f"f_mkt_amount_chg_{w}"] = context["f_mkt_amount_log"].diff(w)
    work = work.drop(columns=["_market_ret_1"]).merge(context, on=DATE_COL, how="left")
    return work


def enrich_with_metadata(price_df: pd.DataFrame, data_dir: str | Path = "./data") -> pd.DataFrame:
    """Attach static, non-forward-looking stock metadata when crawler outputs exist."""
    data_dir = Path(data_dir)
    work = price_df.copy()
    meta_path = data_dir / "stock_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype={ID_COL: str}, encoding="utf-8-sig")
        meta.columns = [str(c).replace("\ufeff", "").strip() for c in meta.columns]
        if ID_COL in meta.columns:
            meta[ID_COL] = meta[ID_COL].map(normalize_stock_code)
            keep = [ID_COL]
            for col in ["市场", "行业", "证券类型", "上市状态"]:
                if col in meta.columns:
                    keep.append(col)
            meta = meta[keep].drop_duplicates(ID_COL, keep="last")
            work = work.merge(meta, on=ID_COL, how="left", suffixes=("", "_meta"))
            if "市场_meta" in work.columns:
                work["市场"] = work.get("市场", work["市场_meta"]).fillna(work["市场_meta"])
                work = work.drop(columns=["市场_meta"])
            if "市场" in work.columns:
                market = work["市场"].astype(str).str.upper()
                work["f_market_sh"] = (market == "SH").astype(float)
                work["f_market_sz"] = (market == "SZ").astype(float)
            if "行业" in work.columns:
                industry = work["行业"].astype(str).str.extract(r"([A-Z][0-9]*)", expand=False).fillna("UNK")
                work["f_industry_code"] = pd.factorize(industry, sort=True)[0].astype(float)
    return work


def _safe_div(a: pd.Series, b: pd.Series | np.ndarray | float) -> pd.Series:
    return a / (b + 1e-12)


def _days_since_event(mask: pd.Series) -> pd.Series:
    event = mask.fillna(False).to_numpy(dtype=bool)
    pos = np.arange(len(event), dtype=float)
    last_pos = pd.Series(np.where(event, pos, np.nan), index=mask.index).ffill()
    return pd.Series(pos, index=mask.index).sub(last_pos).fillna(999.0)


def _keyword_count(text: pd.Series, keywords: tuple[str, ...]) -> pd.Series:
    clean = text.fillna("").astype(str)
    count = pd.Series(0.0, index=text.index)
    for keyword in keywords:
        count += clean.str.contains(keyword, regex=False).astype(float)
    return count


def _build_stock_features(group: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = group.sort_values(DATE_COL).copy()
    open_ = g[OPEN_COL].astype(float)
    close = g[CLOSE_COL].astype(float)
    high = g[HIGH_COL].astype(float)
    low = g[LOW_COL].astype(float)
    volume = g[VOLUME_COL].astype(float).clip(lower=0)
    amount = g[AMOUNT_COL].astype(float).clip(lower=0)
    turnover = g[TURNOVER_COL].astype(float)
    amplitude = g[AMPLITUDE_COL].astype(float)
    free_turnover = g[FREE_TURNOVER_COL].astype(float) if FREE_TURNOVER_COL in g.columns else None
    volume_ratio = g[VOLUME_RATIO_COL].astype(float) if VOLUME_RATIO_COL in g.columns else None
    pe_ttm = g[PE_TTM_COL].astype(float) if PE_TTM_COL in g.columns else None
    pb = g[PB_COL].astype(float) if PB_COL in g.columns else None
    ps_ttm = g[PS_TTM_COL].astype(float) if PS_TTM_COL in g.columns else None
    total_mv = g[TOTAL_MV_COL].astype(float) if TOTAL_MV_COL in g.columns else None
    circ_mv = g[CIRC_MV_COL].astype(float) if CIRC_MV_COL in g.columns else None
    up_limit = g[UP_LIMIT_COL].astype(float) if UP_LIMIT_COL in g.columns else None
    down_limit = g[DOWN_LIMIT_COL].astype(float) if DOWN_LIMIT_COL in g.columns else None
    net_mf_amount = g[NET_MF_AMOUNT_COL].astype(float) if NET_MF_AMOUNT_COL in g.columns else None
    buy_lg_amount = g[BUY_LG_AMOUNT_COL].astype(float) if BUY_LG_AMOUNT_COL in g.columns else None
    sell_lg_amount = g[SELL_LG_AMOUNT_COL].astype(float) if SELL_LG_AMOUNT_COL in g.columns else None
    buy_elg_amount = g[BUY_ELG_AMOUNT_COL].astype(float) if BUY_ELG_AMOUNT_COL in g.columns else None
    sell_elg_amount = g[SELL_ELG_AMOUNT_COL].astype(float) if SELL_ELG_AMOUNT_COL in g.columns else None
    ths_hot_rank = g[THS_HOT_RANK_COL].astype(float) if THS_HOT_RANK_COL in g.columns else None
    ths_hot_pct_change = g[THS_HOT_PCT_CHANGE_COL].astype(float) if THS_HOT_PCT_CHANGE_COL in g.columns else None
    ths_hot_value = g[THS_HOT_VALUE_COL].astype(float) if THS_HOT_VALUE_COL in g.columns else None
    ths_hot_reason = g["ths_hot_reason"] if "ths_hot_reason" in g.columns else None
    ths_hot_concept = g["ths_hot_concept"] if "ths_hot_concept" in g.columns else None
    limit_direction = g[LIMIT_DIRECTION_COL] if LIMIT_DIRECTION_COL in g.columns else None
    top_net_amount = g[TOP_NET_AMOUNT_COL].astype(float) if TOP_NET_AMOUNT_COL in g.columns else None
    margin_fin_balance = g[MARGIN_FIN_BALANCE_COL].astype(float) if MARGIN_FIN_BALANCE_COL in g.columns else None
    margin_total_balance = g[MARGIN_TOTAL_BALANCE_COL].astype(float) if MARGIN_TOTAL_BALANCE_COL in g.columns else None

    prev_close = close.shift(1)
    vwap = _safe_div(amount, volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ret_1 = close.pct_change()
    open_ret_1 = open_.pct_change()
    amount_log = np.log1p(amount)
    volume_log = np.log1p(volume)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    feature_data: dict[str, pd.Series] = {
        "f_gap": _safe_div(open_ - prev_close, prev_close.abs()),
        "f_intraday": _safe_div(close - open_, open_.abs()),
        "f_high_low_pct": _safe_div(high - low, open_.abs()),
        "f_close_to_high": _safe_div(high - close, close.abs()),
        "f_close_to_low": _safe_div(close - low, close.abs()),
        "f_upper_shadow": _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), open_.abs()),
        "f_lower_shadow": _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, open_.abs()),
        "f_vwap_rel_close": _safe_div(vwap - close, close.abs()),
        "f_log_volume": volume_log,
        "f_log_amount": amount_log,
        "f_turnover": turnover,
        "f_amplitude": amplitude,
        "f_ret_1": ret_1,
        "f_open_ret_1": open_ret_1,
        "f_amount_chg_1": amount.pct_change(),
        "f_volume_chg_1": volume.pct_change(),
    }

    if free_turnover is not None:
        feature_data["f_free_turnover"] = free_turnover
        feature_data["f_free_turnover_minus_turnover"] = free_turnover - turnover
    if volume_ratio is not None:
        feature_data["f_volume_ratio"] = volume_ratio
    if up_limit is not None and down_limit is not None:
        feature_data["f_close_to_up_limit"] = _safe_div(up_limit - close, close.abs())
        feature_data["f_open_to_up_limit"] = _safe_div(up_limit - open_, open_.abs())
        feature_data["f_high_to_up_limit"] = _safe_div(up_limit - high, high.abs())
        feature_data["f_close_to_down_limit"] = _safe_div(close - down_limit, close.abs())
        feature_data["f_limit_span"] = _safe_div(up_limit - down_limit, close.abs())
        feature_data["f_hit_up_limit"] = (high >= up_limit * 0.999).astype(float)
        feature_data["f_close_near_up_limit"] = (close >= up_limit * 0.98).astype(float)
        feature_data["f_hit_down_limit"] = (low <= down_limit * 1.001).astype(float)
    if net_mf_amount is not None:
        large_net = None
        extra_large_net = None
        feature_data["f_net_mf_amount_rel"] = _safe_div(net_mf_amount, amount.replace(0, np.nan))
        feature_data["f_net_mf_amount_log_signed"] = np.sign(net_mf_amount) * np.log1p(net_mf_amount.abs())
        if buy_lg_amount is not None and sell_lg_amount is not None:
            large_net = buy_lg_amount - sell_lg_amount
            feature_data["f_large_order_net_rel"] = _safe_div(large_net, amount.replace(0, np.nan))
        if buy_elg_amount is not None and sell_elg_amount is not None:
            extra_large_net = buy_elg_amount - sell_elg_amount
            feature_data["f_extra_large_order_net_rel"] = _safe_div(extra_large_net, amount.replace(0, np.nan))
    if ths_hot_rank is not None:
        hot_rank = ths_hot_rank.fillna(999.0)
        hot_listed = ths_hot_rank.notna()
        hot_rank_score = np.maximum(0.0, 101.0 - hot_rank) / 100.0
        feature_data["f_ths_hot_listed"] = ths_hot_rank.notna().astype(float)
        feature_data["f_ths_hot_rank_inv"] = 1.0 / hot_rank.clip(lower=1.0)
        feature_data["f_ths_hot_rank_score"] = hot_rank_score
        if cfg.add_event_state:
            feature_data["f_ths_hot_days_since"] = _days_since_event(hot_listed)
            feature_data["f_ths_hot_rank_score_chg_1"] = hot_rank_score.diff()
            feature_data["f_ths_hot_rank_score_chg_3"] = hot_rank_score.diff(3)
            feature_data["f_ths_hot_rank_score_ewm_3"] = hot_rank_score.ewm(span=3, min_periods=1, adjust=False).mean()
            feature_data["f_ths_hot_rank_score_ewm_5"] = hot_rank_score.ewm(span=5, min_periods=1, adjust=False).mean()
    if ths_hot_pct_change is not None:
        feature_data["f_ths_hot_pct_change"] = ths_hot_pct_change.fillna(0.0) / 100.0
    if ths_hot_value is not None:
        hot_value_log = np.log1p(ths_hot_value.fillna(0.0).clip(lower=0.0))
        feature_data["f_ths_hot_value_log"] = hot_value_log
        if cfg.add_event_state:
            feature_data["f_ths_hot_value_log_chg_1"] = hot_value_log.diff()
    if cfg.add_event_text and (ths_hot_reason is not None or ths_hot_concept is not None):
        reason = ths_hot_reason if ths_hot_reason is not None else pd.Series("", index=g.index)
        concept = ths_hot_concept if ths_hot_concept is not None else pd.Series("", index=g.index)
        text = reason.fillna("").astype(str) + " " + concept.fillna("").astype(str)
        event_risk = _keyword_count(
            text,
            (
                "冲突缓和",
                "结束战争",
                "停火",
                "放通",
                "复航",
                "调整",
                "大跌",
                "净流出",
                "严重异常波动",
                "异常波动",
                "累计涨幅",
                "风险",
                "监管",
                "减持",
                "亏损",
                "下滑",
                "终止",
                "回撤",
            ),
        )
        event_positive = _keyword_count(
            text,
            (
                "业绩预增",
                "同比增长",
                "订单",
                "中标",
                "量产",
                "小批量",
                "获批",
                "投产",
                "合作",
                "受益",
                "需求",
                "突破",
                "创新药",
                "算力",
                "芯片",
            ),
        )
        event_geo = _keyword_count(text, ("霍尔木兹", "中东", "伊朗", "美伊", "俄乌", "红海", "海峡", "战争", "冲突"))
        event_shipping = _keyword_count(text, ("航运", "船", "油运", "集运", "运价", "港口", "天然气", "海峡"))
        event_ai_chip = _keyword_count(text, ("AI", "算力", "英伟达", "GPU", "芯片", "半导体", "存储", "光模块", "CPO"))
        event_drug = _keyword_count(text, ("创新药", "仿制药", "CRO", "减肥药", "医药", "临床"))
        event_energy = _keyword_count(text, ("电池", "固态电池", "储能", "电力", "天然气", "煤", "光伏", "风电"))
        feature_data["f_event_text_present"] = (text.str.len() > 0).astype(float)
        feature_data["f_event_text_risk"] = event_risk
        feature_data["f_event_text_positive"] = event_positive
        feature_data["f_event_text_net"] = event_positive - event_risk
        feature_data["f_event_text_geo"] = event_geo
        feature_data["f_event_text_shipping"] = event_shipping
        feature_data["f_event_text_geo_shipping_risk"] = ((event_geo > 0) & (event_shipping > 0) & (event_risk > 0)).astype(float)
        feature_data["f_event_text_ai_chip"] = event_ai_chip
        feature_data["f_event_text_drug"] = event_drug
        feature_data["f_event_text_energy"] = event_energy
    if limit_direction is not None:
        direction = limit_direction.astype(str).str.upper()
        limit_up = direction == "U"
        limit_down = direction == "D"
        feature_data["f_limit_list_up"] = limit_up.astype(float)
        feature_data["f_limit_list_down"] = limit_down.astype(float)
        if cfg.add_event_state:
            feature_data["f_limit_up_days_since"] = _days_since_event(limit_up)
            feature_data["f_limit_down_days_since"] = _days_since_event(limit_down)
            feature_data["f_limit_up_ewm_5"] = limit_up.astype(float).ewm(span=5, min_periods=1, adjust=False).mean()
            feature_data["f_limit_down_ewm_5"] = limit_down.astype(float).ewm(span=5, min_periods=1, adjust=False).mean()
    for col in [
        "limit_turnover_ratio",
        "limit_seal_amount",
        "limit_open_times",
        "limit_times",
        "top_net_rate",
        "top_amount_rate",
        "margin_fin_buy",
        "margin_fin_repay",
        "margin_sec_sell",
    ]:
        if col in g.columns:
            s = g[col].astype(float).fillna(0.0)
            feature_data[f"f_{col}"] = s
    if top_net_amount is not None:
        top_listed = top_net_amount.notna()
        top_net_rel = _safe_div(top_net_amount.fillna(0.0), amount.replace(0, np.nan))
        feature_data["f_top_listed"] = top_listed.astype(float)
        feature_data["f_top_net_amount_rel"] = top_net_rel
        if cfg.add_event_state:
            feature_data["f_top_days_since"] = _days_since_event(top_listed)
            feature_data["f_top_net_positive"] = (top_net_amount.fillna(0.0) > 0).astype(float)
            feature_data["f_top_net_amount_rel_chg_1"] = top_net_rel.diff()
    if margin_fin_balance is not None:
        feature_data["f_margin_fin_balance_log"] = np.log1p(margin_fin_balance.fillna(0.0).clip(lower=0.0))
        feature_data["f_margin_fin_balance_chg_5"] = margin_fin_balance.pct_change(5)
        feature_data["f_margin_fin_balance_chg_20"] = margin_fin_balance.pct_change(20)
        if cfg.add_event_state:
            feature_data["f_margin_fin_balance_chg_1"] = margin_fin_balance.pct_change(1)
        if cfg.add_event_state and "margin_fin_buy" in g.columns:
            margin_fin_buy_rel = _safe_div(g["margin_fin_buy"].astype(float).fillna(0.0), amount.replace(0, np.nan))
            feature_data["f_margin_fin_buy_rel"] = margin_fin_buy_rel
            feature_data["f_margin_fin_buy_rel_mean_5"] = margin_fin_buy_rel.rolling(5, min_periods=2).mean()
    if margin_total_balance is not None:
        feature_data["f_margin_total_balance_log"] = np.log1p(margin_total_balance.fillna(0.0).clip(lower=0.0))
        if cfg.add_event_state:
            feature_data["f_margin_total_balance_chg_5"] = margin_total_balance.pct_change(5)
    for col in [c for c in g.columns if c.startswith(TSF_PREFIX)]:
        feature_data[f"f_{col}"] = g[col].astype(float)
    for col in [c for c in g.columns if c.startswith(BOARD_HOT_PREFIX)]:
        feature_data[f"f_{col}"] = g[col].astype(float).fillna(0.0)
    for name, series in [
        ("pe_ttm", pe_ttm),
        ("pb", pb),
        ("ps_ttm", ps_ttm),
        ("total_mv", total_mv),
        ("circ_mv", circ_mv),
    ]:
        if series is None:
            continue
        clean = series.replace([np.inf, -np.inf], np.nan)
        feature_data[f"f_{name}"] = clean
        feature_data[f"f_log_{name}"] = np.log1p(clean.clip(lower=0))

    for w in cfg.windows:
        minp = max(2, min(w, 5))
        roll_ret = ret_1.rolling(w, min_periods=minp)
        roll_open_ret = open_ret_1.rolling(w, min_periods=minp)
        ma_close = close.rolling(w, min_periods=minp).mean()
        ema_close = close.ewm(span=w, min_periods=minp, adjust=False).mean()
        ma_volume = volume.rolling(w, min_periods=minp).mean()
        ma_amount = amount.rolling(w, min_periods=minp).mean()
        max_high = high.rolling(w, min_periods=minp).max()
        min_low = low.rolling(w, min_periods=minp).min()
        abs_path = close.diff().abs().rolling(w, min_periods=minp).sum()
        signed_move = close - close.shift(w)
        downside = ret_1.clip(upper=0)

        feature_data[f"f_close_ret_{w}"] = close.pct_change(w)
        feature_data[f"f_open_ret_{w}"] = open_.pct_change(w)
        feature_data[f"f_ret_mean_{w}"] = roll_ret.mean()
        feature_data[f"f_ret_std_{w}"] = roll_ret.std()
        feature_data[f"f_ret_down_std_{w}"] = downside.rolling(w, min_periods=minp).std()
        feature_data[f"f_ret_skew_{w}"] = roll_ret.skew()
        feature_data[f"f_ret_rank_{w}"] = close.rolling(w, min_periods=minp).rank(pct=True)
        feature_data[f"f_open_ret_mean_{w}"] = roll_open_ret.mean()
        feature_data[f"f_ma_close_rel_{w}"] = _safe_div(ma_close - close, close.abs())
        feature_data[f"f_ema_close_rel_{w}"] = _safe_div(ema_close - close, close.abs())
        feature_data[f"f_max_high_dist_{w}"] = _safe_div(max_high - close, close.abs())
        feature_data[f"f_min_low_dist_{w}"] = _safe_div(close - min_low, close.abs())
        feature_data[f"f_channel_pos_{w}"] = _safe_div(close - min_low, max_high - min_low)
        feature_data[f"f_volume_rel_{w}"] = _safe_div(volume, ma_volume)
        feature_data[f"f_amount_rel_{w}"] = _safe_div(amount, ma_amount)
        feature_data[f"f_turnover_mean_{w}"] = turnover.rolling(w, min_periods=minp).mean()
        if free_turnover is not None:
            feature_data[f"f_free_turnover_mean_{w}"] = free_turnover.rolling(w, min_periods=minp).mean()
        if volume_ratio is not None:
            feature_data[f"f_volume_ratio_mean_{w}"] = volume_ratio.rolling(w, min_periods=minp).mean()
        if up_limit is not None and down_limit is not None:
            hit_up = (high >= up_limit * 0.999).astype(float)
            near_up = (close >= up_limit * 0.98).astype(float)
            feature_data[f"f_hit_up_limit_count_{w}"] = hit_up.rolling(w, min_periods=minp).sum()
            feature_data[f"f_close_near_up_limit_count_{w}"] = near_up.rolling(w, min_periods=minp).sum()
            feature_data[f"f_close_to_up_limit_min_{w}"] = _safe_div(up_limit - close, close.abs()).rolling(w, min_periods=minp).min()
            feature_data[f"f_close_to_up_limit_mean_{w}"] = _safe_div(up_limit - close, close.abs()).rolling(w, min_periods=minp).mean()
        if net_mf_amount is not None:
            net_rel = _safe_div(net_mf_amount, amount.replace(0, np.nan))
            feature_data[f"f_net_mf_amount_sum_rel_{w}"] = _safe_div(
                net_mf_amount.rolling(w, min_periods=minp).sum(),
                amount.rolling(w, min_periods=minp).sum().replace(0, np.nan),
            )
            feature_data[f"f_net_mf_amount_rel_mean_{w}"] = net_rel.rolling(w, min_periods=minp).mean()
            if large_net is not None:
                feature_data[f"f_large_order_net_sum_rel_{w}"] = _safe_div(
                    large_net.rolling(w, min_periods=minp).sum(),
                    amount.rolling(w, min_periods=minp).sum().replace(0, np.nan),
                )
            if extra_large_net is not None:
                feature_data[f"f_extra_large_order_net_sum_rel_{w}"] = _safe_div(
                    extra_large_net.rolling(w, min_periods=minp).sum(),
                    amount.rolling(w, min_periods=minp).sum().replace(0, np.nan),
                )
        if ths_hot_rank is not None:
            listed = ths_hot_rank.notna().astype(float)
            rank_score = np.maximum(0.0, 101.0 - ths_hot_rank.fillna(999.0)) / 100.0
            feature_data[f"f_ths_hot_listed_count_{w}"] = listed.rolling(w, min_periods=minp).sum()
            feature_data[f"f_ths_hot_rank_score_mean_{w}"] = rank_score.rolling(w, min_periods=minp).mean()
            if cfg.add_event_state:
                feature_data[f"f_ths_hot_rank_score_max_{w}"] = rank_score.rolling(w, min_periods=minp).max()
        if ths_hot_value is not None:
            hot_log = np.log1p(ths_hot_value.fillna(0.0).clip(lower=0.0))
            feature_data[f"f_ths_hot_value_log_mean_{w}"] = hot_log.rolling(w, min_periods=minp).mean()
        if cfg.add_event_text and ths_hot_reason is not None:
            if "f_event_text_risk" in feature_data:
                feature_data[f"f_event_text_risk_sum_{w}"] = feature_data["f_event_text_risk"].rolling(w, min_periods=minp).sum()
                feature_data[f"f_event_text_positive_sum_{w}"] = feature_data["f_event_text_positive"].rolling(w, min_periods=minp).sum()
                feature_data[f"f_event_text_net_mean_{w}"] = feature_data["f_event_text_net"].rolling(w, min_periods=minp).mean()
                feature_data[f"f_event_text_geo_shipping_risk_sum_{w}"] = feature_data[
                    "f_event_text_geo_shipping_risk"
                ].rolling(w, min_periods=minp).sum()
        if limit_direction is not None:
            up = limit_direction.astype(str).str.upper().eq("U").astype(float)
            down = limit_direction.astype(str).str.upper().eq("D").astype(float)
            feature_data[f"f_limit_list_up_count_{w}"] = up.rolling(w, min_periods=minp).sum()
            feature_data[f"f_limit_list_down_count_{w}"] = down.rolling(w, min_periods=minp).sum()
        if top_net_amount is not None:
            top_listed = top_net_amount.notna().astype(float)
            feature_data[f"f_top_listed_count_{w}"] = top_listed.rolling(w, min_periods=minp).sum()
            if cfg.add_event_state:
                top_positive = (top_net_amount.fillna(0.0) > 0).astype(float)
                feature_data[f"f_top_net_positive_count_{w}"] = top_positive.rolling(w, min_periods=minp).sum()
            feature_data[f"f_top_net_amount_sum_rel_{w}"] = _safe_div(
                top_net_amount.fillna(0.0).rolling(w, min_periods=minp).sum(),
                amount.rolling(w, min_periods=minp).sum().replace(0, np.nan),
            )
        for col in [c for c in g.columns if c.startswith(BOARD_HOT_PREFIX)]:
            s = g[col].astype(float).fillna(0.0)
            feature_data[f"f_{col}_mean_{w}"] = s.rolling(w, min_periods=minp).mean()
        feature_data[f"f_amplitude_mean_{w}"] = amplitude.rolling(w, min_periods=minp).mean()
        feature_data[f"f_atr_pct_{w}"] = _safe_div(true_range.rolling(w, min_periods=minp).mean(), close.abs())
        feature_data[f"f_trend_eff_{w}"] = _safe_div(signed_move.abs(), abs_path)
        feature_data[f"f_sharpe_like_{w}"] = _safe_div(roll_ret.mean(), roll_ret.std())
        feature_data[f"f_money_flow_{w}"] = (
            np.sign(close - prev_close).fillna(0.0) * amount_log
        ).rolling(w, min_periods=minp).mean()

    if "f_mkt_ret_mean" in g.columns:
        mkt_ret = g["f_mkt_ret_mean"].astype(float)
        feature_data["f_excess_ret_1"] = ret_1 - mkt_ret
        for w in (5, 20, 60):
            if w in cfg.windows:
                minp = max(2, min(w, 5))
                mkt_var = mkt_ret.rolling(w, min_periods=minp).var()
                beta = ret_1.rolling(w, min_periods=minp).cov(mkt_ret) / (mkt_var + 1e-12)
                feature_data[f"f_beta_mkt_{w}"] = beta
                feature_data[f"f_corr_mkt_{w}"] = ret_1.rolling(w, min_periods=minp).corr(mkt_ret)
                feature_data[f"f_alpha_mkt_{w}"] = (ret_1 - beta * mkt_ret).rolling(w, min_periods=minp).mean()
                feature_data[f"f_excess_ret_mean_{w}"] = (ret_1 - mkt_ret).rolling(w, min_periods=minp).mean()

    if {"f_close_ret_5", "f_close_ret_20"}.issubset(feature_data):
        feature_data["f_mom_5_minus_20"] = feature_data["f_close_ret_5"] - feature_data["f_close_ret_20"]
    if {"f_close_ret_20", "f_close_ret_60"}.issubset(feature_data):
        feature_data["f_mom_20_minus_60"] = feature_data["f_close_ret_20"] - feature_data["f_close_ret_60"]
    if {"f_ret_std_5", "f_ret_std_20"}.issubset(feature_data):
        feature_data["f_vol_ratio_5_20"] = _safe_div(feature_data["f_ret_std_5"], feature_data["f_ret_std_20"])
    if {"f_ret_std_20", "f_ret_std_60"}.issubset(feature_data):
        feature_data["f_vol_ratio_20_60"] = _safe_div(feature_data["f_ret_std_20"], feature_data["f_ret_std_60"])
    if {"f_amount_rel_5", "f_amount_rel_20"}.issubset(feature_data):
        feature_data["f_amount_rel_5_minus_20"] = feature_data["f_amount_rel_5"] - feature_data["f_amount_rel_20"]
    if {"f_turnover_mean_5", "f_turnover_mean_20"}.issubset(feature_data):
        feature_data["f_turnover_5_minus_20"] = feature_data["f_turnover_mean_5"] - feature_data["f_turnover_mean_20"]

    feature_data["risk_vol_20"] = feature_data.get("f_ret_std_20", ret_1.rolling(20, min_periods=5).std())
    feature_data["risk_vol_60"] = feature_data.get("f_ret_std_60", ret_1.rolling(60, min_periods=10).std())
    feature_df = pd.DataFrame(feature_data, index=g.index)
    return pd.concat([g, feature_df], axis=1).copy()


def add_forward_labels(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    df = df.sort_values([ID_COL, DATE_COL]).copy()
    grouped = df.groupby(ID_COL, sort=False)
    open_t1 = grouped[OPEN_COL].shift(-1)
    open_t5 = grouped[OPEN_COL].shift(-cfg.horizon)
    df[LABEL_COL] = open_t5 / (open_t1 + 1e-12) - 1.0
    df.loc[open_t1 <= 1e-12, LABEL_COL] = np.nan
    return df


def make_relevance(df: pd.DataFrame, label_col: str = LABEL_COL, top_k: int = 5) -> pd.Series:
    def _score_day(s: pd.Series) -> pd.Series:
        ranks = s.rank(method="first", ascending=False)
        n = len(s)
        rel = np.maximum(0, top_k + 1 - ranks.to_numpy())
        rel += np.maximum(0, (n - ranks.to_numpy()) / max(n - 1, 1))
        return pd.Series(rel, index=s.index, dtype=float)

    return df.groupby(DATE_COL, group_keys=False)[label_col].apply(_score_day)


def build_feature_frame(raw_df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig(), with_labels: bool = True) -> tuple[pd.DataFrame, list[str]]:
    work = raw_df.copy()
    work[ID_COL] = work[ID_COL].map(normalize_stock_code)
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work = work.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    featured = pd.concat(
        [_build_stock_features(group, cfg) for _, group in work.groupby(ID_COL, sort=False)],
        ignore_index=True,
    )

    if cfg.add_industry_aggregate and "行业" in featured.columns:
        industry_base_cols = [
            col
            for col in [
                "f_ret_1",
                "f_close_ret_3",
                "f_close_ret_5",
                "f_close_ret_10",
                "f_close_ret_20",
                "f_amount_rel_5",
                "f_amount_rel_20",
                "f_turnover",
                "f_amplitude",
                "f_hit_up_limit",
                "f_close_near_up_limit",
            ]
            if col in featured.columns
        ]
        if industry_base_cols:
            daily = featured[[DATE_COL, "行业", *industry_base_cols]].copy()
            grouped = daily.groupby([DATE_COL, "行业"], sort=True)
            agg = grouped[industry_base_cols].mean().reset_index()
            agg = agg.sort_values(["行业", DATE_COL]).reset_index(drop=True)
            rename_map = {col: f"f_industry_mean_{col.removeprefix('f_')}" for col in industry_base_cols}
            agg = agg.rename(columns=rename_map)
            ind_feature_cols = list(rename_map.values())
            for col in ind_feature_cols:
                grp = agg.groupby("行业", sort=False)[col]
                for w in (3, 5, 10, 20):
                    minp = max(2, min(w, 5))
                    agg[f"{col}_ma_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=minp).mean())
                agg[f"{col}_chg_5"] = grp.diff(5)
            featured = featured.merge(agg, on=[DATE_COL, "行业"], how="left")

    if with_labels:
        featured = add_forward_labels(featured, cfg)

    feature_cols = [c for c in featured.columns if c.startswith("f_") or c.startswith("risk_")]
    if cfg.add_cross_sectional:
        cs_source_cols = list(feature_cols)
        by_date = featured.groupby(DATE_COL, sort=False)
        cs_data = {}
        for col in cs_source_cols:
            mean = by_date[col].transform("mean")
            std = by_date[col].transform("std")
            cs_data[f"{col}_cs_z"] = (featured[col] - mean) / (std + 1e-12)
            cs_data[f"{col}_cs_rank"] = by_date[col].rank(pct=True)
        featured = pd.concat([featured, pd.DataFrame(cs_data, index=featured.index)], axis=1).copy()
        feature_cols = [c for c in featured.columns if c.startswith("f_") or c.startswith("risk_")]

    if cfg.add_industry_relative and "行业" in featured.columns:
        industry_cols = [
            col for col in [
                "f_ret_1",
                "f_close_ret_5",
                "f_close_ret_20",
                "f_close_ret_60",
                "f_mom_5_minus_20",
                "f_mom_20_minus_60",
                "f_ret_std_20",
                "f_ret_std_60",
                "f_volume_rel_20",
                "f_amount_rel_20",
                "f_turnover",
                "f_beta_mkt_20",
                "risk_vol_20",
            ]
            if col in featured.columns
        ]
        if industry_cols:
            by_industry = featured.groupby([DATE_COL, "行业"], sort=False)
            industry_data = {}
            for col in industry_cols:
                mean = by_industry[col].transform("mean")
                std = by_industry[col].transform("std")
                industry_data[f"{col}_ind_z"] = (featured[col] - mean) / (std + 1e-12)
                industry_data[f"{col}_ind_rank"] = by_industry[col].rank(pct=True)
            featured = pd.concat([featured, pd.DataFrame(industry_data, index=featured.index)], axis=1).copy()
            feature_cols = [c for c in featured.columns if c.startswith("f_") or c.startswith("risk_")]

    featured[feature_cols] = featured[feature_cols].replace([np.inf, -np.inf], np.nan)
    featured[feature_cols] = featured.groupby(ID_COL, sort=False)[feature_cols].ffill()
    featured[feature_cols] = featured[feature_cols].fillna(0.0)
    featured = featured.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    return featured, feature_cols


def filter_trainable_rows(df: pd.DataFrame, feature_cols: list[str], cfg: FeatureConfig) -> pd.DataFrame:
    work = df.dropna(subset=[LABEL_COL]).copy()
    work["_history_count"] = work.groupby(ID_COL).cumcount() + 1
    work = work[work["_history_count"] >= cfg.min_history].copy()
    finite_mask = np.isfinite(work[feature_cols].to_numpy(dtype=np.float32)).all(axis=1)
    work = work.loc[finite_mask].drop(columns=["_history_count"])
    return work.reset_index(drop=True)
