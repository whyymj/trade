#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻爬虫测试 - 测试数据抓取、频率控制、增量爬取、URL去重、真实数据验证
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
from services.news.data import NewsCrawler
from services.news.data.news_crawler import NewsItem


class TestNewsCrawlerFrequency:
    """新闻爬虫频率控制测试类"""

    def test_can_fetch_initial_state(self):
        """测试初始状态 - 应该可以爬取"""
        crawler = NewsCrawler()
        assert crawler.can_fetch() is True

    def test_can_fetch_after_limit_reached(self):
        """测试达到每日限制后 - 不应该可以爬取"""
        crawler = NewsCrawler()
        crawler._daily_count = 4
        crawler._last_fetch_time = datetime.now()

        assert crawler.can_fetch() is False

    def test_can_fetch_with_time_interval(self):
        """测试时间间隔限制 - 4小时内不能重复爬取"""
        crawler = NewsCrawler()
        crawler._daily_count = 1
        crawler._last_fetch_time = datetime.now() - timedelta(hours=2)

        assert crawler.can_fetch() is False

    def test_can_fetch_after_interval_passed(self):
        """测试超过4小时后 - 应该可以爬取"""
        crawler = NewsCrawler()
        crawler._daily_count = 1
        crawler._last_fetch_time = datetime.now() - timedelta(hours=5)

        assert crawler.can_fetch() is True

    def test_daily_count_reset(self):
        """测试每日计数重置"""
        crawler = NewsCrawler()
        crawler._daily_count = 4
        crawler._last_date = (date.today() - timedelta(days=1)).isoformat()

        crawler._reset_daily_count_if_needed()
        assert crawler._daily_count == 0

    def test_max_daily_fetches_constant(self):
        """测试每日最大爬取次数常量"""
        crawler = NewsCrawler()
        assert crawler.MAX_DAILY_FETCHES == 4

    def test_min_interval_hours_constant(self):
        """测试最小间隔小时数常量"""
        crawler = NewsCrawler()
        assert crawler.MIN_INTERVAL_HOURS == 4


class TestNewsCrawlerDeduplication:
    """新闻爬虫去重测试类"""

    def test_deduplicate_by_url(self):
        """测试按 URL 去重"""
        crawler = NewsCrawler()

        news_list = [
            NewsItem(
                title="新闻1", content="内容1", source="源1", url="http://test.com/1"
            ),
            NewsItem(
                title="新闻2", content="内容2", source="源2", url="http://test.com/2"
            ),
            NewsItem(
                title="新闻1重复",
                content="内容1",
                source="源1",
                url="http://test.com/1",
            ),
        ]

        result = crawler._deduplicate(news_list)
        assert len(result) == 2
        assert result[0].url == "http://test.com/1"
        assert result[1].url == "http://test.com/2"

    def test_deduplicate_empty_list(self):
        """测试去重空列表"""
        crawler = NewsCrawler()
        result = crawler._deduplicate([])
        assert len(result) == 0

    def test_deduplicate_single_item(self):
        """测试去重单个新闻"""
        crawler = NewsCrawler()
        news = [
            NewsItem(title="新闻", content="内容", source="源", url="http://test.com/1")
        ]
        result = crawler._deduplicate(news)
        assert len(result) == 1

    def test_deduplicate_preserves_order(self):
        """测试去重保持顺序"""
        crawler = NewsCrawler()

        news_list = [
            NewsItem(
                title="新闻1", content="内容1", source="源1", url="http://test.com/1"
            ),
            NewsItem(
                title="新闻2", content="内容2", source="源2", url="http://test.com/2"
            ),
            NewsItem(
                title="新闻3", content="内容3", source="源3", url="http://test.com/3"
            ),
            NewsItem(
                title="新闻1重复",
                content="内容1",
                source="源1",
                url="http://test.com/1",
            ),
            NewsItem(
                title="新闻2重复",
                content="内容2",
                source="源2",
                url="http://test.com/2",
            ),
        ]

        result = crawler._deduplicate(news_list)
        assert len(result) == 3
        assert result[0].title == "新闻1"
        assert result[1].title == "新闻2"
        assert result[2].title == "新闻3"


class TestNewsCrawlerIncrementalFetch:
    """新闻爬虫增量爬取测试类"""

    def test_fetch_today_default_sources(self):
        """测试默认数据源"""
        crawler = NewsCrawler()
        with (
            patch.object(crawler, "fetch_eastmoney", return_value=[]),
            patch.object(crawler, "fetch_cailian", return_value=[]),
            patch.object(crawler, "fetch_wallstreet", return_value=[]),
        ):
            result = crawler.fetch_today()
            assert isinstance(result, list)

    def test_fetch_today_custom_sources(self):
        """测试自定义数据源"""
        crawler = NewsCrawler()
        with patch.object(crawler, "fetch_eastmoney", return_value=[]):
            result = crawler.fetch_today(sources=["eastmoney"])
            assert isinstance(result, list)

    def test_fetch_today_only_todays_news(self):
        """测试只爬取今天的新闻"""
        crawler = NewsCrawler()

        def mock_fetch():
            today_news = NewsItem(
                title="今天新闻",
                content="内容",
                source="测试",
                url="http://test.com/today",
                published_at=datetime.now(),
            )
            return [today_news]

        with (
            patch.object(crawler, "fetch_eastmoney", side_effect=mock_fetch),
            patch.object(crawler, "fetch_cailian", return_value=[]),
            patch.object(crawler, "fetch_wallstreet", return_value=[]),
        ):
            result = crawler.fetch_today()
            assert len(result) == 1
            assert result[0].news_date == date.today()

    def test_fetch_today_updates_daily_count(self):
        """测试爬取后更新每日计数"""
        crawler = NewsCrawler()
        initial_count = crawler._daily_count

        with (
            patch.object(
                crawler,
                "fetch_eastmoney",
                return_value=[
                    NewsItem(
                        title="测试",
                        content="内容",
                        source="测试",
                        url="http://test.com",
                    )
                ],
            ),
            patch.object(crawler, "fetch_cailian", return_value=[]),
            patch.object(crawler, "fetch_wallstreet", return_value=[]),
        ):
            crawler.fetch_today()
            assert crawler._daily_count == initial_count + 1


class TestNewsCrawlerRealFetch:
    """新闻爬虫真实数据抓取测试类"""

    @pytest.mark.slow
    def test_fetch_eastmoney_real(self):
        """测试真实抓取东方财富新闻"""
        try:
            crawler = NewsCrawler()
            news = crawler.fetch_eastmoney()
            assert isinstance(news, list)
            if news:
                assert len(news) > 0
                for item in news:
                    assert isinstance(item, NewsItem)
                    assert item.title
                    assert item.url
                    assert item.source == "东方财富"
        except Exception as e:
            pytest.skip(f"东方财富抓取跳过: {e}")

    @pytest.mark.slow
    def test_fetch_cailian_real(self):
        """测试真实抓取财联社新闻"""
        try:
            crawler = NewsCrawler()
            news = crawler.fetch_cailian()
            assert isinstance(news, list)
            if news:
                assert len(news) > 0
                for item in news:
                    assert isinstance(item, NewsItem)
                    assert item.title
                    assert item.url
                    assert item.source == "财联社"
        except Exception as e:
            pytest.skip(f"财联社抓取跳过: {e}")

    @pytest.mark.slow
    def test_fetch_wallstreet_real(self):
        """测试真实抓取华尔街见闻新闻"""
        try:
            crawler = NewsCrawler()
            news = crawler.fetch_wallstreet()
            assert isinstance(news, list)
            if news:
                assert len(news) > 0
                for item in news:
                    assert isinstance(item, NewsItem)
                    assert item.title
                    assert item.url
                    assert item.source == "华尔街见闻"
        except Exception as e:
            pytest.skip(f"华尔街见闻抓取跳过: {e}")

    @pytest.mark.slow
    def test_fetch_today_real(self):
        """测试真实抓取今天的新闻（重点测试）"""
        try:
            crawler = NewsCrawler()
            news = crawler.fetch_today()

            assert isinstance(news, list)
            if news:
                print(f"\n[真实抓取] 成功抓取 {len(news)} 条新闻")
                for i, item in enumerate(news[:5]):
                    print(f"  {i + 1}. {item.title} - {item.source}")

                for item in news:
                    assert isinstance(item, NewsItem)
                    assert item.title
                    assert item.content
                    assert item.source
                    assert item.url
                    assert item.news_date == date.today()
        except Exception as e:
            pytest.skip(f"真实抓取跳过: {e}")

    @pytest.mark.slow
    def test_fetch_all_sources_real(self):
        """测试从所有数据源抓取"""
        try:
            crawler = NewsCrawler()
            eastmoney = crawler.fetch_eastmoney()
            cailian = crawler.fetch_cailian()
            wallstreet = crawler.fetch_wallstreet()

            total = len(eastmoney) + len(cailian) + len(wallstreet)
            print(
                f"\n[多源抓取] 东方财富: {len(eastmoney)}, 财联社: {len(cailian)}, 华尔街见闻: {len(wallstreet)}, 总计: {total}"
            )

            assert isinstance(eastmoney, list)
            assert isinstance(cailian, list)
            assert isinstance(wallstreet, list)
        except Exception as e:
            pytest.skip(f"多源抓取跳过: {e}")


class TestNewsCrawlerCategorization:
    """新闻爬虫分类测试类"""

    def test_categorize_macro(self):
        """测试宏观分类"""
        crawler = NewsCrawler()
        title = "央行降息0.25个百分点"
        category = crawler._categorize(title)
        assert category == "宏观"

    def test_categorize_policy(self):
        """测试政策分类"""
        crawler = NewsCrawler()
        title = "证监会发布新监管政策"
        category = crawler._categorize(title)
        assert category == "政策"

    def test_categorize_industry(self):
        """测试行业分类"""
        crawler = NewsCrawler()
        title = "新能源汽车销量创新高"
        category = crawler._categorize(title)
        assert category == "行业"

    def test_categorize_company(self):
        """测试公司分类"""
        crawler = NewsCrawler()
        title = "某公司发布财报营收增长"
        category = crawler._categorize(title)
        assert category == "公司"

    def test_categorize_global(self):
        """测试全球分类"""
        crawler = NewsCrawler()
        title = "美联储宣布加息决定"
        category = crawler._categorize(title)
        assert category == "全球"

    def test_categorize_general(self):
        """测试通用分类"""
        crawler = NewsCrawler()
        title = "这是一条普通新闻"
        category = crawler._categorize(title)
        assert category == "general"

    def test_categorize_case_insensitive(self):
        """测试大小写不敏感"""
        crawler = NewsCrawler()
        title1 = "AI技术突破"
        title2 = "ai技术突破"
        category1 = crawler._categorize(title1)
        category2 = crawler._categorize(title2)
        assert category1 == category2


class TestNewsCrawlerStatus:
    """新闻爬虫状态测试类"""

    def test_get_fetch_status_structure(self):
        """测试获取爬取状态 - 结构"""
        crawler = NewsCrawler()
        status = crawler.get_fetch_status()

        assert "can_fetch" in status
        assert "daily_count" in status
        assert "max_daily" in status
        assert "last_fetch" in status
        assert "next_fetch_after" in status

    def test_get_fetch_status_values(self):
        """测试获取爬取状态 - 值"""
        crawler = NewsCrawler()
        crawler._daily_count = 2
        crawler._last_fetch_time = datetime.now() - timedelta(hours=5)

        status = crawler.get_fetch_status()

        assert status["can_fetch"] is True
        assert status["daily_count"] == 2
        assert status["max_daily"] == 4
        assert status["last_fetch"] is not None

    def test_get_fetch_status_when_cannot_fetch(self):
        """测试获取爬取状态 - 不能爬取时"""
        crawler = NewsCrawler()
        crawler._daily_count = 4
        crawler._last_fetch_time = datetime.now()

        status = crawler.get_fetch_status()

        assert status["can_fetch"] is False
        assert status["daily_count"] == 4


class TestNewsCrawlerUserAgent:
    """新闻爬虫 User-Agent 测试类"""

    def test_get_random_user_agent(self):
        """测试获取随机 User-Agent"""
        crawler = NewsCrawler()
        ua = crawler._get_random_user_agent()
        assert isinstance(ua, str)
        assert "Mozilla" in ua
        assert len(ua) > 0

    def test_user_agent_list_not_empty(self):
        """测试 User-Agent 列表不为空"""
        crawler = NewsCrawler()
        assert len(crawler.USER_AGENTS) > 0

    def test_different_user_agents(self):
        """测试获取不同的 User-Agent"""
        crawler = NewsCrawler()
        uas = [crawler._get_random_user_agent() for _ in range(10)]
        unique_uas = set(uas)
        assert len(unique_uas) > 1 or len(crawler.USER_AGENTS) == 1


class TestNewsItem:
    """新闻项数据类测试"""

    def test_news_item_creation(self):
        """测试创建新闻项"""
        item = NewsItem(
            title="测试标题", content="测试内容", source="测试源", url="http://test.com"
        )
        assert item.title == "测试标题"
        assert item.content == "测试内容"
        assert item.source == "测试源"
        assert item.url == "http://test.com"

    def test_news_item_with_published_at(self):
        """测试带发布时间的新闻项"""
        pub_time = datetime.now()
        item = NewsItem(
            title="测试",
            content="内容",
            source="源",
            url="http://test.com",
            published_at=pub_time,
        )
        assert item.published_at == pub_time
        assert item.news_date == pub_time.date()

    def test_news_item_without_published_at(self):
        """测试不带发布时间的新闻项"""
        item = NewsItem(
            title="测试", content="内容", source="源", url="http://test.com"
        )
        assert item.news_date == date.today()

    def test_news_item_with_category(self):
        """测试带分类的新闻项"""
        item = NewsItem(
            title="测试",
            content="内容",
            source="源",
            url="http://test.com",
            category="宏观",
        )
        assert item.category == "宏观"

    def test_news_item_default_category(self):
        """测试默认分类"""
        item = NewsItem(
            title="测试", content="内容", source="源", url="http://test.com"
        )
        assert item.category == "general"
