# -*- coding: utf-8 -*-
"""
新闻模块测试用例
运行方式: python -m pytest tests/test_news.py -v
或: python tests/test_news.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

# 确保项目根目录在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 加载 .env
from dotenv import load_dotenv

load_dotenv(_root / ".env")


# ---------- 新闻爬虫测试 ----------

def test_news_crawler_fetch():
    """测试新闻爬取"""
    from data.news import NewsCrawler, NewsItem
    
    crawler = NewsCrawler()
    
    # 测试 can_fetch 初始状态
    can_fetch = crawler.can_fetch()
    assert isinstance(can_fetch, bool)
    
    print("✅ 新闻爬虫抓取测试通过")


def test_news_crawler_fetch_mock():
    """测试新闻爬取（Mock外部依赖）"""
    from data.news import NewsCrawler, NewsItem
    
    crawler = NewsCrawler()
    
    # Mock 各个数据源
    with patch.object(crawler, 'fetch_eastmoney', return_value=[]):
        with patch.object(crawler, 'fetch_cailian', return_value=[]):
            with patch.object(crawler, 'fetch_wallstreet', return_value=[]):
                news = crawler.fetch_today(sources=['eastmoney'])
                assert isinstance(news, list)
    
    print("✅ 新闻爬虫Mock抓取测试通过")


def test_news_crawler_deduplicate():
    """测试去重功能"""
    from data.news import NewsCrawler, NewsItem
    from datetime import datetime
    
    crawler = NewsCrawler()
    
    # 创建重复的新闻
    news1 = NewsItem(
        title="测试新闻1",
        content="内容1",
        source="东方财富",
        url="http://example.com/news/1",
        published_at=datetime.now(),
        category="宏观"
    )
    news2 = NewsItem(
        title="测试新闻2", 
        content="内容2",
        source="财联社",
        url="http://example.com/news/1",  # 相同URL
        published_at=datetime.now(),
        category="政策"
    )
    news3 = NewsItem(
        title="测试新闻3",
        content="内容3",
        source="华尔街见闻",
        url="http://example.com/news/3",  # 不同URL
        published_at=datetime.now(),
        category="行业"
    )
    
    news_list = [news1, news2, news3]
    deduplicated = crawler._deduplicate(news_list)
    
    # 应该只保留2条（去除重复的news2）
    assert len(deduplicated) == 2
    assert deduplicated[0].url == "http://example.com/news/1"
    assert deduplicated[1].url == "http://example.com/news/3"
    
    print("✅ 新闻去重测试通过")


def test_news_crawler_frequency():
    """测试频率控制"""
    from data.news import NewsCrawler
    
    crawler = NewsCrawler()
    
    # 初始状态应该可以爬取
    assert crawler.can_fetch() == True
    
    # 模拟已经爬取过
    crawler._daily_count = 4  # 达到每日上限
    assert crawler.can_fetch() == False
    
    # 重置后再测试
    crawler._daily_count = 0
    crawler._last_fetch_time = datetime.now()  # 最近刚爬取过
    assert crawler.can_fetch() == False
    
    # 测试时间间隔
    crawler._last_fetch_time = datetime.now() - timedelta(hours=5)  # 5小时前
    assert crawler.can_fetch() == True
    
    print("✅ 频率控制测试通过")


def test_news_crawler_get_status():
    """测试获取爬虫状态"""
    from data.news import NewsCrawler
    
    crawler = NewsCrawler()
    status = crawler.get_fetch_status()
    
    assert "can_fetch" in status
    assert "daily_count" in status
    assert "max_daily" in status
    assert status["max_daily"] == 4
    
    print("✅ 获取爬虫状态测试通过")


def test_news_crawler_categorize():
    """测试新闻分类"""
    from data.news import NewsCrawler
    
    crawler = NewsCrawler()
    
    # 测试各个分类
    assert crawler._categorize("央行降息") == "宏观"
    assert crawler._categorize("证监会发布新规") == "政策"
    assert crawler._categorize("新能源车销量增长") == "行业"
    assert crawler._categorize("公司财报营收增长") == "公司"
    assert crawler._categorize("美联储加息") == "全球"
    assert crawler._categorize("这是一条普通新闻") == "general"
    
    print("✅ 新闻分类测试通过")


# ---------- 新闻仓储测试 ----------

def test_news_repo_save():
    """测试保存新闻"""
    from data.news import NewsRepo, NewsItem
    from datetime import datetime
    
    repo = NewsRepo()
    
    # 创建测试新闻
    test_news = [
        NewsItem(
            title=f"测试新闻_{datetime.now().timestamp()}",
            content="这是测试内容",
            source="测试来源",
            url=f"http://test.com/{datetime.now().timestamp()}",
            published_at=datetime.now(),
            category="测试"
        )
    ]
    
    # 保存新闻（可能返回0因为使用INSERT IGNORE）
    result = repo.save_news(test_news)
    assert isinstance(result, int)
    
    print("✅ 保存新闻测试通过")


def test_news_repo_get():
    """测试获取新闻"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    
    # 获取新闻
    news_list = repo.get_news(days=7, limit=10)
    assert isinstance(news_list, list)
    
    # 测试带分类过滤
    news_with_category = repo.get_news(days=7, category="宏观", limit=10)
    assert isinstance(news_with_category, list)
    
    print("✅ 获取新闻测试通过")


def test_news_repo_get_today():
    """测试获取今日新闻"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    news_list = repo.get_today_news()
    
    assert isinstance(news_list, list)
    print(f"✅ 获取今日新闻测试通过, 共 {len(news_list)} 条")


def test_news_repo_get_by_url():
    """测试根据URL获取新闻"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    
    # 先获取一条新闻
    news_list = repo.get_today_news()
    
    if len(news_list) > 0:
        news = news_list[0]
        found_news = repo.get_news_by_url(news.url)
        
        assert found_news is not None
        assert found_news.url == news.url
        print("✅ 根据URL获取新闻测试通过")
    else:
        print("⚠️  无新闻数据，跳过根据URL获取新闻测试")


def test_news_repo_count():
    """测试获取新闻数量"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    count = repo.get_news_count(days=7)
    
    assert isinstance(count, int)
    assert count >= 0
    print(f"✅ 获取新闻数量测试通过, 共 {count} 条")


def test_news_repo_cleanup():
    """测试清理过期新闻"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    
    # 清理7天前的新闻（测试用）
    deleted = repo.cleanup_old_news(keep_days=7)
    
    assert isinstance(deleted, int)
    assert deleted >= 0
    print(f"✅ 清理过期新闻测试通过, 删除 {deleted} 条")


def test_news_repo_categories():
    """测试获取分类统计"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    categories = repo.get_categories()
    
    assert isinstance(categories, list)
    print(f"✅ 获取分类统计测试通过, 共 {len(categories)} 个分类")


def test_news_repo_save_analysis():
    """测试保存分析结果"""
    from data.news import NewsRepo, AnalysisResult
    from datetime import datetime
    
    repo = NewsRepo()
    
    # 创建测试分析结果
    analysis = AnalysisResult(
        news_count=5,
        summary="测试摘要",
        deep_analysis="深度分析内容",
        market_impact="bullish",
        key_events=[{"title": "事件1", "source": "来源1"}],
        investment_advice="建议买入",
        analyzed_at=datetime.now()
    )
    
    result = repo.save_analysis(analysis)
    assert isinstance(result, bool)
    
    print("✅ 保存分析结果测试通过")


def test_news_repo_get_analysis():
    """测试获取分析结果"""
    from data.news import NewsRepo
    
    repo = NewsRepo()
    analysis = repo.get_latest_analysis()
    
    # analysis 可能是 None（如果没有历史数据）
    if analysis:
        assert hasattr(analysis, 'news_count')
        assert hasattr(analysis, 'summary')
        assert hasattr(analysis, 'market_impact')
        print("✅ 获取分析结果测试通过")
    else:
        print("⚠️  无分析结果，跳过测试")


# ---------- 新闻API测试 ----------

def test_news_api_list():
    """测试列表API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试最新新闻
    resp = client.get("/api/news/latest")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert data["code"] == 0
    assert "data" in data
    print("✅ 新闻列表API测试通过")
    
    # 测试状态API（验证服务正常）
    resp = client.get("/api/news/status")
    assert resp.status_code == 200
    print("✅ 新闻状态API测试通过")


def test_news_api_sync():
    """测试同步API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试同步（可能受频率限制）
    resp = client.post("/api/news/sync")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    assert "data" in data
    
    if data["code"] == 0:
        print(f"✅ 新闻同步API测试通过, 获取 {data['data'].get('fetched', 0)} 条")
    else:
        print(f"⚠️  频率限制: {data.get('message')}")


def test_news_api_analyze():
    """测试分析API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试分析（可能需要新闻数据）
    resp = client.post("/api/news/analyze", json={"days": 1})
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    
    if data["code"] == 0 and data["data"]:
        result = data["data"]
        assert "news_count" in result
        assert "summary" in result
        print(f"✅ 新闻分析API测试通过, 分析 {result.get('news_count', 0)} 条新闻")
    else:
        print("⚠️  无新闻数据或LLM不可用，跳过详细测试")


def test_news_api_detail():
    """测试详情API"""
    from data.news import NewsRepo
    from server.app import create_app
    import urllib.parse
    
    repo = NewsRepo()
    news_list = repo.get_today_news()
    
    app = create_app()
    client = app.test_client()
    
    if len(news_list) > 0:
        news = news_list[0]
        encoded_url = urllib.parse.quote(news.url, safe="")
        resp = client.get(f"/api/news/detail/{encoded_url}")
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["code"] == 0
        assert data["data"]["title"] == news.title
        print("✅ 新闻详情API测试通过")
    else:
        print("⚠️  无新闻数据，跳过测试")


def test_news_api_status():
    """测试状态API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    resp = client.get("/api/news/status")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert data["code"] == 0
    assert "data" in data
    assert "crawler" in data["data"]
    assert "news_count" in data["data"]
    print("✅ 新闻状态API测试通过")


def test_news_api_cleanup():
    """测试清理API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    resp = client.post("/api/news/cleanup", json={"keep_days": 30})
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert data["code"] == 0
    print("✅ 新闻清理API测试通过")


def test_news_api_analysis_latest():
    """测试获取最新分析API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    resp = client.get("/api/news/analysis/latest")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    print("✅ 获取最新分析API测试通过")


# ---------- NewsItem 数据类测试 ----------

def test_news_item_creation():
    """测试NewsItem创建"""
    from data.news import NewsItem
    from datetime import datetime
    
    # 带完整参数
    news = NewsItem(
        title="测试标题",
        content="测试内容",
        source="测试来源",
        url="http://test.com",
        published_at=datetime(2025, 1, 1, 10, 0, 0),
        category="宏观",
        news_date=date(2025, 1, 1)
    )
    
    assert news.title == "测试标题"
    assert news.category == "宏观"
    assert news.news_date == date(2025, 1, 1)
    
    # 不带news_date，自动填充
    news2 = NewsItem(
        title="测试2",
        content="内容2",
        source="来源2",
        url="http://test2.com",
        published_at=datetime(2025, 1, 1, 12, 0, 0)
    )
    
    assert news2.news_date == date(2025, 1, 1)
    
    print("✅ NewsItem数据类测试通过")


# ---------- 运行所有测试 ----------

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始新闻模块测试...")
    print("=" * 60)
    
    tests = [
        # 爬虫测试
        ("新闻爬取", test_news_crawler_fetch),
        ("新闻爬取(Mock)", test_news_crawler_fetch_mock),
        ("新闻去重", test_news_crawler_deduplicate),
        ("频率控制", test_news_crawler_frequency),
        ("获取爬虫状态", test_news_crawler_get_status),
        ("新闻分类", test_news_crawler_categorize),
        
        # 仓储测试
        ("保存新闻", test_news_repo_save),
        ("获取新闻", test_news_repo_get),
        ("获取今日新闻", test_news_repo_get_today),
        ("根据URL获取新闻", test_news_repo_get_by_url),
        ("获取新闻数量", test_news_repo_count),
        ("清理过期新闻", test_news_repo_cleanup),
        ("获取分类统计", test_news_repo_categories),
        ("保存分析结果", test_news_repo_save_analysis),
        ("获取分析结果", test_news_repo_get_analysis),
        
        # API测试
        ("列表API", test_news_api_list),
        ("同步API", test_news_api_sync),
        ("分析API", test_news_api_analyze),
        ("详情API", test_news_api_detail),
        ("状态API", test_news_api_status),
        ("清理API", test_news_api_cleanup),
        ("获取最新分析API", test_news_api_analysis_latest),
        
        # 数据类测试
        ("NewsItem创建", test_news_item_creation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            print(f"\n📌 测试: {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
