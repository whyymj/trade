#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器用工具函数：配置、数据目录、akshare 拉取、DataFrame 转接口 JSON。
数据存储使用 MySQL（data.stock_repo），不再使用 CSV。

模块结构（便于拓展）:
- 配置与路径: get_data_dir, load_config, get_date_range_from_config
- 股票信息与拉取: get_stock_name, fetch_hist_remote, fetch_hist
- 存储与格式: save_stock_db, df_to_chart_result（读写在 DB）
- 批量与配置写入: update_all_stocks, add_stock_to_config, add_stock_and_fetch

接口与功能说明见项目根目录 docs/API.md。
"""

from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml

# 项目根目录（server 包上一级）
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"


def get_data_dir() -> Path:
    """从 config 读取 output_dir，未配置则用默认 data。"""
    if not CONFIG_PATH.exists():
        return DATA_DIR
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        out = (cfg.get("output_dir") or "").strip()
        if out:
            p = Path(out)
            if not p.is_absolute():
                p = ROOT / p
            return p
    except Exception:
        pass
    return DATA_DIR


def load_config() -> dict:
    """加载 config.yaml，失败返回空字典。"""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def is_a_share_stock(symbol: str) -> bool:
    """是否为 A 股代码：6 位数字。"""
    s = (symbol or "").strip()
    return s.isdigit() and len(s) == 6


def is_hk_stock(symbol: str) -> bool:
    """是否为港股代码：5 位数字，或 5 位数字 + .HK。"""
    s = (symbol or "").strip()
    if s.endswith(".HK") and len(s) == 8 and s[:5].isdigit():
        return True
    return s.isdigit() and len(s) == 5


def is_valid_stock_code(symbol: str) -> bool:
    """是否为支持的股票代码（A 股 6 位或港股 5 位/.HK）。"""
    return is_a_share_stock(symbol) or is_hk_stock(symbol)


def get_stock_name(symbol: str) -> str:
    """根据股票代码获取股票名称（akshare）。A 股用 individual_info，港股用 spot 列表匹配。"""
    symbol = (symbol or "").strip()
    if is_hk_stock(symbol):
        hk_code = symbol.replace(".HK", "").strip()
        try:
            df = ak.stock_hk_spot_em()
            if df is not None and not df.empty and "代码" in df.columns and "名称" in df.columns:
                row = df[df["代码"].astype(str).str.strip() == hk_code]
                if not row.empty:
                    return str(row["名称"].iloc[0]).strip()
        except Exception:
            pass
        return symbol
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is None or df.empty:
            return symbol
        cols = df.columns.tolist()
        if len(cols) >= 2:
            key_col, val_col = cols[0], cols[1]
            row = df[df[key_col].astype(str).str.strip() == "股票简称"]
            if not row.empty:
                return str(row[val_col].iloc[0]).strip()
        return symbol
    except Exception:
        return symbol


def fetch_hist_remote(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """强制从 akshare 拉取单只股票日线，不读本地。A 股用 stock_zh_a_hist，港股用 stock_hk_hist。"""
    symbol = (symbol or "").strip()
    print(f"fetch_hist_remote: {symbol}, {start_date}, {end_date}, {adjust}")
    try:
        if is_a_share_stock(symbol):
            return ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        if is_hk_stock(symbol):
            hk_symbol = symbol.replace(".HK", "").strip()
            return ak.stock_hk_hist(
                symbol=hk_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust or "",
            )
        return None
    except Exception:
        return None


def _market(symbol: str) -> str:
    """返回市场标识：a / hk。"""
    return "hk" if is_hk_stock(symbol) else "a"


def save_stock_db(symbol: str, df: pd.DataFrame, name: str | None = None) -> int:
    """将日线 DataFrame 写入数据库（stock_meta + stock_daily），返回写入行数。"""
    from data import stock_repo

    symbol = (symbol or "").strip()
    if not symbol or df is None or df.empty:
        return 0
    display_name = (name or get_stock_name(symbol)).strip()
    stock_repo.upsert_stock_meta(symbol, display_name or None, _market(symbol))
    return stock_repo.save_stock_daily_batch(symbol, df)


def fetch_hist(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """
    获取单只股票日线历史数据：优先从数据库读取，无数据则 akshare 拉取并写入 DB（A 股/港股自动区分）。
    """
    from data import stock_repo

    symbol = (symbol or "").strip()
    df = stock_repo.get_stock_daily_df(symbol, start_date, end_date)
    if df is not None and not df.empty:
        return df
    df = fetch_hist_remote(symbol, start_date, end_date, adjust)
    if df is not None and not df.empty:
        save_stock_db(symbol, df)
    return df


def df_to_chart_result(df: pd.DataFrame) -> dict:
    """
    将日线 DataFrame 转为与 /api/data 一致的 JSON 结构，供 ECharts 使用。
    要求 df 含「日期」列或首列为日期。
    """
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    df = df.sort_values(date_col).reset_index(drop=True)
    result = {"dates": df[date_col].astype(str).tolist()}
    for col in df.columns:
        if col == date_col:
            continue
        try:
            result[col] = df[col].tolist()
        except Exception:
            result[col] = df[col].astype(str).tolist()
    return result


def get_date_range_from_config() -> tuple[str, str, str]:
    """
    从 config 读取 start_date、end_date、adjust，未配置则默认近一年、前复权。
    返回 (start_date, end_date, adjust)，日期格式 YYYYMMDD。
    """
    cfg = load_config()
    start_date = (cfg.get("start_date") or "").strip().replace("-", "")
    end_date = (cfg.get("end_date") or "").strip().replace("-", "")
    adjust = (cfg.get("adjust") or "qfq").strip() or "qfq"
    today = datetime.now().date()
    if not end_date:
        end_date = today.strftime("%Y%m%d")
    if not start_date:
        one_year_ago = today - timedelta(days=365)
        start_date = one_year_ago.strftime("%Y%m%d")
    return start_date, end_date, adjust


def update_all_stocks() -> list[dict]:
    """
    按 config 中 stocks 列表，逐个从网络拉取并写入数据库。
    返回 [{ "symbol": "600519", "ok": True/False, "message": "..." }, ...]
    """
    cfg = load_config()
    stocks = [str(s).strip() for s in (cfg.get("stocks") or []) if str(s).strip()]
    start_date, end_date, adjust = get_date_range_from_config()
    result = []
    for symbol in stocks:
        df = fetch_hist_remote(symbol, start_date, end_date, adjust)
        if df is None or df.empty:
            result.append({"symbol": symbol, "ok": False, "message": "拉取失败或无数据"})
            continue
        try:
            n = save_stock_db(symbol, df)
            result.append({"symbol": symbol, "ok": True, "message": f"已更新 {n} 条"})
        except Exception as e:
            result.append({"symbol": symbol, "ok": False, "message": str(e)})
    return result


def update_daily_stocks() -> list[dict]:
    """
    每日增量更新：按 config.stocks 对每只股票只拉取「尚未入库」的交易日到今天的数据。
    若该股尚无 last_trade_date 则按 config 的日期范围全量拉取一次。
    返回 [{ "symbol": str, "ok": bool, "message": str }, ...]
    """
    from datetime import timedelta

    from data import stock_repo

    cfg = load_config()
    symbols = [str(s).strip() for s in (cfg.get("stocks") or []) if str(s).strip()]
    if not symbols:
        return []
    rows = stock_repo.get_symbols_for_daily_update()
    last_by_symbol = {r["symbol"]: r.get("last_trade_date") for r in rows}
    today = datetime.now().date()
    today_str = today.strftime("%Y%m%d")
    _, _, adjust = get_date_range_from_config()
    result = []
    for symbol in symbols:
        last = last_by_symbol.get(symbol)
        if last is not None:
            # last 可能是 date 或 datetime（来自 MySQL）
            start_d = last + timedelta(days=1) if hasattr(last, "__add__") else last
            start_str = start_d.strftime("%Y%m%d") if hasattr(start_d, "strftime") else str(start_d).replace("-", "")[:8]
            end_str = today_str
            if start_str > end_str:
                result.append({"symbol": symbol, "ok": True, "message": "已是最新"})
                continue
        else:
            start_str, _, _ = get_date_range_from_config()
            end_str = today_str
        df = fetch_hist_remote(symbol, start_str, end_str, adjust)
        if df is None or df.empty:
            result.append({"symbol": symbol, "ok": False, "message": "拉取失败或无新数据"})
            continue
        try:
            n = save_stock_db(symbol, df)
            result.append({"symbol": symbol, "ok": True, "message": f"已更新 {n} 条"})
        except Exception as e:
            result.append({"symbol": symbol, "ok": False, "message": str(e)})
    return result


def add_stock_to_config(symbol: str) -> bool:
    """将股票代码追加到 config.yaml 的 stocks 列表（已存在则不重复）。"""
    symbol = str(symbol).strip()
    if not symbol:
        return False
    cfg = load_config()
    stocks = [str(s).strip() for s in (cfg.get("stocks") or [])]
    if symbol in stocks:
        return True
    stocks.append(symbol)
    cfg["stocks"] = stocks
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return True
    except Exception:
        return False


def add_stock_and_fetch(symbol: str) -> dict:
    """
    抓取该股票数据并写入数据库，并将代码加入 config.stocks。
    支持 A 股 6 位数字、港股 5 位数字或 xxxxx.HK。
    返回 { "ok": bool, "message": str, "displayName": str 可选 }
    """
    symbol = str(symbol).strip()
    if not symbol or not is_valid_stock_code(symbol):
        return {"ok": False, "message": "股票代码需为 A股6位数字 或 港股5位数字/5位.HK"}
    start_date, end_date, adjust = get_date_range_from_config()
    df = fetch_hist_remote(symbol, start_date, end_date, adjust)
    if df is None or df.empty:
        return {"ok": False, "message": "拉取数据失败或暂无数据"}
    try:
        n = save_stock_db(symbol, df)
    except Exception as e:
        return {"ok": False, "message": f"保存失败: {e}"}
    display_name = get_stock_name(symbol)
    if not add_stock_to_config(symbol):
        return {"ok": True, "message": "数据已保存，写入配置失败", "displayName": display_name}
    return {"ok": True, "message": f"已抓取并加入配置，共 {n} 条", "displayName": display_name}


def sync_all_from_config(clear_first: bool = True) -> list[dict]:
    """
    按 config 全量同步：可选先清空数据库中的全部行情与元信息，再逐个拉取并写入 DB。
    返回 [{ "symbol": str, "ok": bool, "message": str }, ...]
    """
    from data import stock_repo

    if clear_first:
        try:
            stock_repo.clear_all_stock_data()
        except Exception:
            pass
    cfg = load_config()
    stocks = [str(s).strip() for s in (cfg.get("stocks") or []) if str(s).strip()]
    start_date, end_date, adjust = get_date_range_from_config()
    result = []
    for symbol in stocks:
        df = fetch_hist_remote(symbol, start_date, end_date, adjust)
        if df is None or df.empty:
            result.append({"symbol": symbol, "ok": False, "message": "拉取失败或无数据"})
            continue
        try:
            n = save_stock_db(symbol, df)
            result.append({"symbol": symbol, "ok": True, "message": f"已保存 {n} 条"})
        except Exception as e:
            result.append({"symbol": symbol, "ok": False, "message": str(e)})
    return result


def remove_stock_from_config(symbol: str, delete_data: bool = True) -> dict:
    """
    从 config.stocks 移除该代码；可选同时删除数据库中该股票的全部日线与元信息。
    返回 { "ok": bool, "message": str }
    """
    from data import stock_repo

    symbol = (symbol or "").strip()
    if not symbol:
        return {"ok": False, "message": "缺少代码"}
    cfg = load_config()
    stocks = [str(s).strip() for s in (cfg.get("stocks") or [])]
    if symbol not in stocks:
        return {"ok": True, "message": "已不在配置中"}
    stocks = [s for s in stocks if s != symbol]
    cfg["stocks"] = stocks
    if delete_data:
        try:
            stock_repo.delete_stock_data(symbol)
        except Exception:
            pass
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return {"ok": True, "message": "已移除"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def save_config(updates: dict) -> dict:
    """
    用 updates 合并当前 config 并写回 config.yaml。
    updates 可含 start_date, end_date, adjust, stocks, output_dir。
    返回 { "ok": bool, "message": str }
    """
    cfg = load_config()
    for k, v in updates.items():
        if v is not None and k in ("start_date", "end_date", "adjust", "output_dir", "stocks"):
            cfg[k] = v
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return {"ok": True, "message": "已保存"}
    except Exception as e:
        return {"ok": False, "message": str(e)}
