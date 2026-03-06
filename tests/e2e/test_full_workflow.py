import pytest
import requests
import time


@pytest.mark.e2e
class TestFullWorkflow:
    """完整流程 E2E 测试"""

    @pytest.mark.slow
    def test_news_workflow(self):
        """测试新闻完整流程：爬取 → 分类 → 分析"""
        base_url = "http://localhost:5050"

        # 1. 爬取新闻
        sync_response = requests.post(f"{base_url}/api/news/sync", timeout=60)
        assert sync_response.status_code in [200, 201]

        # 2. 获取新闻
        list_response = requests.get(f"{base_url}/api/news/list?days=1")
        assert list_response.status_code == 200
        news_data = list_response.json().get("data", [])

        # 3. 分析新闻
        if news_data:
            analyze_response = requests.post(
                f"{base_url}/api/news/analyze", json={"days": 1}, timeout=60
            )
            assert analyze_response.status_code == 200
            result = analyze_response.json()
            assert "summary" in result.get("data", {})

    @pytest.mark.slow
    def test_fund_analysis_workflow(self):
        """测试基金分析完整流程"""
        base_url = "http://localhost:5050"

        # 1. 获取基金列表
        list_response = requests.get(f"{base_url}/api/fund/list?page=1&size=1")
        assert list_response.status_code == 200
        funds = list_response.json().get("data", [])

        if funds:
            fund_code = funds[0]["fund_code"]

            # 2. 获取基金详情
            detail_response = requests.get(f"{base_url}/api/fund/{fund_code}")
            assert detail_response.status_code == 200

            # 3. 获取投资建议
            advice_response = requests.get(
                f"{base_url}/api/investment-advice/{fund_code}", timeout=30
            )
            assert advice_response.status_code == 200

            # 4. 分析基金行业
            industry_response = requests.post(
                f"{base_url}/api/fund-industry/analyze/{fund_code}", timeout=30
            )
            assert industry_response.status_code in [200, 201]

    @pytest.mark.slow
    def test_news_classification_and_fund_matching(self):
        """测试新闻分类和基金匹配完整流程"""
        base_url = "http://localhost:5050"

        # 1. 同步新闻
        sync_response = requests.post(f"{base_url}/api/news/sync", timeout=60)
        assert sync_response.status_code in [200, 201]

        # 2. 分类今日新闻
        classify_response = requests.post(
            f"{base_url}/api/news-classification/classify-today", timeout=60
        )
        assert classify_response.status_code in [200, 201]

        # 3. 为基金匹配新闻
        match_response = requests.get(f"{base_url}/api/fund-news/match/001302")
        assert match_response.status_code == 200
        data = match_response.json()
        assert "data" in data
