import pandas as pd
from datetime import datetime
from typing import Optional

try:
    import akshare as ak

    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class MarketCrawler:
    """市场数据爬虫"""

    def fetch_money_flow(self) -> pd.DataFrame:
        """抓取资金流向数据"""
        if not AKSHARE_AVAILABLE:
            print("[MarketCrawler] akshare not available")
            return pd.DataFrame()

        try:
            df = ak.stock_individual_fund_flow(stock="000001", market="sh")
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(
                columns={
                    "日期": "trade_date",
                    "主力净流入-净额": "main_money",
                    "北向净流入-净额": "north_money",
                    "融资融券-融资买入额": "margin_buy",
                }
            )

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date", ascending=False).head(1)

            print(f"[MarketCrawler] Money flow: {len(df)} records")
            return df

        except Exception as e:
            print(f"[MarketCrawler] Money flow error: {e}")
            return pd.DataFrame()

    def fetch_sentiment(self) -> pd.DataFrame:
        """抓取市场情绪数据"""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            from datetime import datetime, timedelta

            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol="000001",
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(
                columns={
                    "日期": "trade_date",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "涨跌幅": "change_pct",
                    "振幅": "amplitude",
                    "换手率": "turnover_rate",
                }
            )

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date", ascending=False).head(1)

            df["up_count"] = 0
            df["down_count"] = 0
            df["advance_count"] = 0
            df["decline_count"] = 0

            print(f"[MarketCrawler] Sentiment: {len(df)} records")
            return df

        except Exception as e:
            print(f"[MarketCrawler] Sentiment error: {e}")
            return pd.DataFrame()

    def fetch_macro(self) -> pd.DataFrame:
        """抓取宏观经济数据"""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            df = ak.macro_china_人民资产负债表()
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(
                columns={
                    "月份": "period",
                    "广义货币(M2)": "m2",
                }
            )

            print(f"[MarketCrawler] Macro: {len(df)} records")
            return df

        except Exception as e:
            print(f"[MarketCrawler] Macro error: {e}")
            return pd.DataFrame()

    def fetch_global(self) -> pd.DataFrame:
        """抓取全球宏观数据"""
        if not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            df = ak.currency_latest()
            if df is None or df.empty:
                return pd.DataFrame()

            df["trade_date"] = datetime.now().date()

            print(f"[MarketCrawler] Global: {len(df)} records")
            return df

        except Exception as e:
            print(f"[MarketCrawler] Global error: {e}")
            return pd.DataFrame()

    def sync_all(self) -> dict:
        """同步全部市场数据"""
        results = {}
        results["macro"] = self.fetch_macro()
        results["money_flow"] = self.fetch_money_flow()
        results["sentiment"] = self.fetch_sentiment()
        results["global"] = self.fetch_global()
        return results


def get_crawler() -> MarketCrawler:
    """获取爬虫实例"""
    return MarketCrawler()
