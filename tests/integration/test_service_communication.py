import pytest
import requests


@pytest.mark.integration
class TestServiceCommunication:
    """服务间通信集成测试"""

    @pytest.mark.slow
    def test_fund_service_to_llm_service(self):
        """测试基金服务调用 LLM 服务"""
        response = requests.post(
            "http://localhost:5050/api/investment-advice/001302", timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    @pytest.mark.slow
    def test_news_service_to_llm_service(self):
        """测试新闻服务调用 LLM 服务"""
        response = requests.post(
            "http://localhost:5050/api/news/analyze", json={"days": 1}, timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data.get("data", {})

    @pytest.mark.slow
    def test_fund_intel_service_multi_call(self):
        """测试基金智能服务多服务调用"""
        response = requests.post(
            "http://localhost:5050/api/fund-industry/analyze/001302", timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
