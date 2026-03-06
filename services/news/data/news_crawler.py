#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻爬虫模块 - 支持多数据源、增量爬取、频率控制
"""

import random
import re
import json
from datetime import datetime, date
from typing import List, Optional
from dataclasses import dataclass
import requests


@dataclass
class NewsItem:
    """新闻项数据类"""

    title: str
    content: str
    source: str
    url: str
    published_at: Optional[datetime] = None
    category: str = "general"
    news_date: Optional[date] = None
    id: Optional[int] = None

    def __post_init__(self):
        if self.news_date is None:
            self.news_date = (
                self.published_at.date() if self.published_at else date.today()
            )


class NewsCrawler:
    """新闻爬虫 - 支持增量爬取、频率控制"""

    MIN_INTERVAL_HOURS = 4
    MAX_DAILY_FETCHES = 4
    REQUEST_DELAY_SECONDS = 2
    MAX_RETRIES = 3

    _last_fetch_time: Optional[datetime] = None
    _daily_count: int = 0
    _last_date: Optional[str] = None

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        self._reset_daily_count_if_needed()

    def _get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self.USER_AGENTS)

    def _add_delay(self):
        """添加请求间隔"""
        delay = self.REQUEST_DELAY_SECONDS + random.uniform(0, 1)
        import time

        time.sleep(delay)

    def _make_request(
        self, url: str, headers: dict = None, params: dict = None
    ) -> Optional[requests.Response]:
        """带重试的请求"""
        if headers is None:
            headers = {}
        headers["User-Agent"] = self._get_random_user_agent()

        import time

        for attempt in range(self.MAX_RETRIES):
            try:
                self._add_delay()
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code in (403, 429):
                    print(
                        f"[NewsCrawler] 请求被拒绝 (status={resp.status_code}), 等待更长时间..."
                    )
                    time.sleep(10)
                else:
                    print(f"[NewsCrawler] 请求失败 (status={resp.status_code})")
            except requests.exceptions.Timeout:
                print(
                    f"[NewsCrawler] 请求超时 (attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
            except Exception as e:
                print(f"[NewsCrawler] 请求异常: {e}")

            if attempt < self.MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 5
                print(f"[NewsCrawler] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        return None

    def _reset_daily_count_if_needed(self):
        """每日重置计数"""
        today = date.today().isoformat()
        if self._last_date != today:
            self._last_date = today
            self._daily_count = 0

    def can_fetch(self) -> bool:
        """检查是否可以爬取"""
        self._reset_daily_count_if_needed()

        if self._daily_count >= self.MAX_DAILY_FETCHES:
            return False

        if self._last_fetch_time:
            hours_since = (
                datetime.now() - self._last_fetch_time
            ).total_seconds() / 3600
            if hours_since < self.MIN_INTERVAL_HOURS:
                return False

        return True

    def fetch_today(self, sources: List[str] = None) -> List[NewsItem]:
        """
        抓取当天新闻（增量）

        Args:
            sources: 数据源列表，默认 ['eastmoney', 'cailian', 'wallstreet']
        """
        if sources is None:
            sources = ["eastmoney", "cailian", "wallstreet"]

        all_news = []

        for source in sources:
            try:
                if source == "eastmoney":
                    news = self.fetch_eastmoney()
                elif source == "cailian":
                    news = self.fetch_cailian()
                elif source == "wallstreet":
                    news = self.fetch_wallstreet()
                else:
                    continue
                all_news.extend(news)

                import time

                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"[NewsCrawler] {source} fetch error: {e}")

        all_news = self._deduplicate(all_news)

        if all_news:
            self._last_fetch_time = datetime.now()
            self._daily_count += 1

        return all_news

    def fetch_eastmoney(self) -> List[NewsItem]:
        """抓取东方财富财经新闻"""
        try:
            import akshare as ak

            news_list = []
            for symbol in ["全球", "A股", "美股"]:
                try:
                    df = ak.stock_news_em(symbol=symbol)
                    for _, row in df.head(20).iterrows():
                        try:
                            pub_time = row.get("发布时间")
                            if isinstance(pub_time, str):
                                pub_time = datetime.fromisoformat(
                                    pub_time.replace(" ", "T")
                                )
                            elif pub_time is None:
                                pub_time = datetime.now()

                            news = NewsItem(
                                title=row.get("新闻标题", ""),
                                content=str(row.get("新闻内容", ""))[:500],
                                source=row.get("文章来源", "东方财富"),
                                url=row.get("新闻链接", ""),
                                published_at=pub_time,
                                category=self._categorize(row.get("新闻标题", "")),
                            )
                            news_list.append(news)
                        except Exception:
                            continue
                except Exception:
                    continue

            print(f"[NewsCrawler] 东方财富获取 {len(news_list)} 条")
            return news_list

        except Exception as e:
            print(f"[NewsCrawler] 东方财富 error: {e}")
            return []

    def fetch_cailian(self) -> List[NewsItem]:
        """抓取财联社"""
        try:
            url = "https://www.cls.cn/nodeapi/update"
            headers = {
                "Referer": "https://www.cls.cn/",
            }

            resp = self._make_request(url, headers=headers)
            if not resp or resp.status_code != 200:
                return []

            data = resp.json()
            if data.get("code") != 0:
                return []

            news_list = []
            for item in data.get("data", {}).get("roll_data", []):
                try:
                    pub_time = datetime.fromtimestamp(item.get("ctime", 0))
                    if pub_time.date() != date.today():
                        continue

                    news = NewsItem(
                        title=item.get("title", ""),
                        content=item.get("content", "")[:500],
                        source="财联社",
                        url=f"https://www.cls.cn/detail/{item.get('id', '')}",
                        published_at=pub_time,
                        category=self._categorize(item.get("title", "")),
                    )
                    news_list.append(news)
                except Exception:
                    continue

            print(f"[NewsCrawler] 财联社获取 {len(news_list)} 条")
            return news_list

        except Exception as e:
            print(f"[NewsCrawler] 财联社 error: {e}")
            return []

    def fetch_wallstreet(self) -> List[NewsItem]:
        """抓取华尔街见闻"""
        try:
            url = "https://api.goldboot.cn/news/list"
            params = {"type": "flash", "limit": 20}

            resp = self._make_request(url, params=params)
            if not resp or resp.status_code != 200:
                return []

            data = resp.json()

            news_list = []
            for item in data.get("data", []):
                try:
                    title = item.get("title", "")
                    if not title:
                        continue

                    pub_time_str = item.get("pubTime", "")
                    if pub_time_str:
                        pub_time = datetime.strptime(pub_time_str, "%Y-%m-%d %H:%M:%S")
                        if pub_time.date() != date.today():
                            continue
                    else:
                        pub_time = datetime.now()

                    news = NewsItem(
                        title=title,
                        content=item.get("content", "")[:500],
                        source="华尔街见闻",
                        url=item.get("url", ""),
                        published_at=pub_time,
                        category=self._categorize(title),
                    )
                    news_list.append(news)
                except Exception:
                    continue

            print(f"[NewsCrawler] 华尔街见闻获取 {len(news_list)} 条")
            return news_list

        except Exception as e:
            print(f"[NewsCrawler] 华尔街见闻 error: {e}")
            return []

    def _categorize(self, title: str) -> str:
        """根据标题自动分类"""
        title_lower = title.lower()

        keywords = {
            "宏观": [
                "央行",
                "货币政策",
                "降息",
                "降准",
                "gdp",
                "cpi",
                "ppi",
                "社融",
                "m2",
                "利率",
            ],
            "政策": ["证监会", "银保监会", "财政部", "发改委", "政策", "监管"],
            "行业": [
                "新能源",
                "半导体",
                "芯片",
                "人工智能",
                "ai",
                "医药",
                "消费",
                "房地产",
            ],
            "公司": ["财报", "业绩", "营收", "利润", "上市", "融资"],
            "全球": ["美联储", "美元", "原油", "黄金", "美股", "欧股", "地缘"],
        }

        for category, words in keywords.items():
            for word in words:
                if word in title_lower:
                    return category

        return "general"

    def _deduplicate(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """根据URL去重"""
        seen = set()
        result = []
        for news in news_list:
            if news.url and news.url not in seen:
                seen.add(news.url)
                result.append(news)
        return result

    def get_fetch_status(self) -> dict:
        """获取爬取状态"""
        self._reset_daily_count_if_needed()

        can_fetch = self.can_fetch()
        next_fetch_time = None

        if not can_fetch and self._last_fetch_time:
            next_time = self._last_fetch_time.replace(
                hour=self._last_fetch_time.hour + 4
            )
            next_fetch_time = next_time.isoformat()

        return {
            "can_fetch": can_fetch,
            "daily_count": self._daily_count,
            "max_daily": self.MAX_DAILY_FETCHES,
            "last_fetch": self._last_fetch_time.isoformat()
            if self._last_fetch_time
            else None,
            "next_fetch_after": next_fetch_time,
        }


def get_crawler() -> NewsCrawler:
    """获取爬虫实例"""
    return NewsCrawler()
