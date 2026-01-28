#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器用工具函数：配置、数据目录、akshare 拉取、DataFrame 转接口 JSON。
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


def get_stock_name(symbol: str) -> str:
    """根据股票代码获取股票名称（akshare）。"""
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
    """强制从 akshare 拉取单只股票日线，不读本地。"""
    print(f"fetch_hist_remote: {symbol}, {start_date}, {end_date}, {adjust}")
    try:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception:
        return None


def save_stock_csv(symbol: str, df: pd.DataFrame, start_date: str, end_date: str) -> Path | None:
    """将日线 DataFrame 按「股票名称（日期范围）.csv」保存到数据目录。"""
    name = get_stock_name(symbol)
    name_clean = name.replace(" ", "")
    safe_name = "".join(c for c in name_clean if c not in r'\/:*?"<>|')
    date_range_str = f"{start_date}-{end_date}"
    filename = f"{safe_name}（{date_range_str}）.csv"
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def get_local_stock_path(symbol: str) -> Path | None:
    """在数据目录下查找股票代码对应的本地 CSV，有则返回路径，否则返回 None。"""
    data_dir = get_data_dir()
    if not data_dir.exists():
        return None
    symbol = str(symbol).strip()
    for path in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", nrows=1)
            if "股票代码" in df.columns and str(df["股票代码"].iloc[0]).strip() == symbol:
                return path
        except Exception:
            continue
    return None


def fetch_hist(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """
    获取单只股票日线历史数据：本地已有对应 CSV 则直接读取，否则用 akshare 拉取。
    """
    local_path = get_local_stock_path(symbol)
    if local_path is not None:
        try:
            return pd.read_csv(local_path, encoding="utf-8-sig")
        except Exception:
            pass
    try:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception:
        return None


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
    按 config 中 stocks 列表，逐个从网络拉取并覆盖保存到数据目录。
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
            save_stock_csv(symbol, df, start_date, end_date)
            result.append({"symbol": symbol, "ok": True, "message": f"已更新 {len(df)} 条"})
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
    抓取该股票数据并保存到数据目录，并将代码加入 config.stocks。
    返回 { "ok": bool, "message": str, "displayName": str 可选 }
    """
    symbol = str(symbol).strip()
    if not symbol or not symbol.isdigit() or len(symbol) != 6:
        return {"ok": False, "message": "股票代码需为 6 位数字"}
    start_date, end_date, adjust = get_date_range_from_config()
    df = fetch_hist_remote(symbol, start_date, end_date, adjust)
    if df is None or df.empty:
        return {"ok": False, "message": "拉取数据失败或暂无数据"}
    try:
        save_stock_csv(symbol, df, start_date, end_date)
    except Exception as e:
        return {"ok": False, "message": f"保存失败: {e}"}
    if not add_stock_to_config(symbol):
        return {"ok": True, "message": "数据已保存，写入配置失败", "displayName": get_stock_name(symbol)}
    return {"ok": True, "message": f"已抓取并加入配置，共 {len(df)} 条", "displayName": get_stock_name(symbol)}
