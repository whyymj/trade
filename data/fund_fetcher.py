# -*- coding: utf-8 -*-
"""
基金数据抓取：支持多个数据源，自动切换。
"""

import akshare as ak
import pandas as pd
import requests
import re
import json
from typing import Optional
from datetime import datetime


def fetch_fund_nav(fund_code: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    获取基金净值历史，自动尝试多个数据源。
    返回 DataFrame：columns = [nav_date, unit_nav, accum_nav, daily_return]
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return None

    # 数据源列表，按优先级排序
    sources = [
        ("东方财富API", _fetch_from_eastmoney),
        ("天天基金", _fetch_from_tiantian),
        ("akshare", _fetch_from_akshare),
    ]

    for name, fetch_func in sources:
        try:
            df = fetch_func(fund_code, days)
            if df is not None and len(df) > 0:
                print(f"[FundFetcher] {fund_code} 从{name}获取到 {len(df)} 条数据")
                return df
        except Exception as e:
            print(f"[FundFetcher] {fund_code} 从{name}获取失败: {e}")

    return None


def _fetch_from_eastmoney(fund_code: str, days: int) -> Optional[pd.DataFrame]:
    """从东方财富获取 - 解析JS数据"""
    try:
        import time

        url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js?v={int(time.time())}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "http://fund.eastmoney.com/",
        }

        resp = requests.get(url, timeout=15, headers=headers)
        if resp.status_code != 200:
            return None

        text = resp.text

        # 提取 Data_netWorthTrend - 包含单位净值和日增长率
        match = re.search(r"var Data_netWorthTrend\s*=\s*(\[.*?\]);", text, re.DOTALL)
        if not match:
            return None

        data = json.loads(match.group(1))
        if not data:
            return None

        records = []
        for item in data:
            try:
                # x 是毫秒时间戳
                timestamp = item.get("x", 0) // 1000
                date = datetime.fromtimestamp(timestamp)
                unit_nav = item.get("y")
                daily_return = item.get("equityReturn", 0)

                if unit_nav:
                    records.append(
                        {
                            "nav_date": date,
                            "unit_nav": float(unit_nav),
                            "accum_nav": None,  # 东方财富接口没有累计净值
                            "daily_return": float(daily_return)
                            if daily_return
                            else 0.0,
                        }
                    )
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df = df.sort_values("nav_date")
            return df

    except Exception as e:
        print(f"东方财富获取失败: {e}")

    return None


def _fetch_from_tiantian(fund_code: str, days: int) -> Optional[pd.DataFrame]:
    """从天天基金获取"""
    try:
        import time

        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)

        # 构造URL
        url = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js?v={int(time.time())}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "http://fund.eastmoney.com/",
        }

        resp = requests.get(url, timeout=10, headers=headers)
        if resp.status_code != 200:
            return None

        text = resp.text

        # 提取净值数据
        # 格式: var Data_ACWFF = [[日期,单位净值,累计净值,日增长率],...]
        patterns = [
            r"Data_ACWFF\s*=\s*(\[.*?\]);",
            r"data\s*:\s*(\[.*?\])",
            r"\[(\d{4}-\d{2}-\d{2}),[\d.]+,[\d.]+,[\d.-]+",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    # 尝试解析
                    data_str = match.group(1) if match.lastindex else match.group(0)
                    # 简单解析
                    records = []
                    dates_match = re.findall(r"(\d{4}-\d{2}-\d{2})", data_str)
                    navs_match = re.findall(r",([\d.]+),", data_str)

                    if dates_match and len(navs_match) >= len(dates_match) * 2:
                        for i, date_str in enumerate(dates_match):
                            try:
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                unit_nav = (
                                    float(navs_match[i * 2])
                                    if i * 2 < len(navs_match)
                                    else None
                                )
                                accum_nav = (
                                    float(navs_match[i * 2 + 1])
                                    if i * 2 + 1 < len(navs_match)
                                    else None
                                )

                                if unit_nav:
                                    records.append(
                                        {
                                            "nav_date": date,
                                            "unit_nav": unit_nav,
                                            "accum_nav": accum_nav,
                                            "daily_return": None,
                                        }
                                    )
                            except:
                                continue

                        if records:
                            df = pd.DataFrame(records)
                            df = df.sort_values("nav_date")
                            # 计算日收益率
                            df["daily_return"] = df["unit_nav"].pct_change() * 100
                            return df
                except Exception as e:
                    print(f"解析失败: {e}")
                    continue

    except Exception as e:
        print(f"天天基金获取失败: {e}")

    return None


def _fetch_from_akshare(fund_code: str, days: int) -> Optional[pd.DataFrame]:
    """从akshare获取（备用方案）"""
    try:
        df = ak.fund_open_fund_daily_em()

        if "基金代码" not in df.columns:
            return None

        df = df[df["基金代码"] == fund_code]

        if df.empty:
            return None

        # 找到净值日期列
        nav_cols = [c for c in df.columns if "-单位净值" in c]

        if not nav_cols:
            return None

        records = []
        for col in nav_cols:
            date_str = col.replace("-单位净值", "")
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")

                accum_col = col.replace("单位净值", "累计净值")

                unit_nav = df.iloc[0][col] if col in df.columns else None
                accum_nav = df.iloc[0][accum_col] if accum_col in df.columns else None

                if pd.isna(unit_nav) or str(unit_nav).strip() == "":
                    continue

                daily_return = (
                    df.iloc[0]["日增长率"] if "日增长率" in df.columns else None
                )

                records.append(
                    {
                        "nav_date": date,
                        "unit_nav": float(str(unit_nav).replace(",", "")),
                        "accum_nav": float(str(accum_nav).replace(",", ""))
                        if accum_nav and not pd.isna(accum_nav)
                        else None,
                        "daily_return": float(
                            str(daily_return).replace("%", "").replace(",", "")
                        )
                        if daily_return and not pd.isna(daily_return)
                        else 0.0,
                    }
                )
            except Exception:
                continue

        if not records:
            return None

        result = pd.DataFrame(records)
        result = result.sort_values("nav_date")
        return result

    except Exception as e:
        print(f"akshare获取失败: {e}")
        return None

        # 只保留最近 days 天
        if days and days > 0:
            result = result.tail(days)

        return result

    except Exception as e:
        print(f"Error fetching fund nav: {e}")
        return None


def fetch_fund_info(fund_code: str, timeout: int = 3) -> Optional[dict]:
    """获取基金基本信息。"""
    try:
        # 直接从东方财富API获取
        import requests

        url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
        resp = requests.get(url, timeout=timeout)

        if resp.status_code == 200:
            text = resp.text
            # 尝试多种模式匹配基金名称
            import re

            patterns = [
                r'fundName\s*=\s*["\']([^"\']+)["\']',
                r'"fundName":"([^"]+)"',
                r'name\s*=\s*["\']([^"\']+)["\']',
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return {
                        "fund_code": fund_code,
                        "fund_name": match.group(1),
                    }

        # 如果获取不到名称，至少返回代码
        return {"fund_code": fund_code, "fund_name": None}

    except Exception as e:
        print(f"Error fetching fund info: {e}")
        return None


# 测试
if __name__ == "__main__":
    print("Testing fetch_fund_nav...")
    df = fetch_fund_nav("012729", days=30)
    if df is not None:
        print(f"Got {len(df)} records")
        print(df.tail())
    else:
        print("No data")
