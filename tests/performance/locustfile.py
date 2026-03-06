from locust import HttpUser, task, between


class FundProphetUser(HttpUser):
    """FundProphet 性能测试用户"""

    wait_time = between(1, 3)

    @task(3)
    def get_fund_list(self):
        """获取基金列表"""
        self.client.get("/api/fund/list?page=1&size=20")

    @task(2)
    def get_fund_detail(self):
        """获取基金详情"""
        self.client.get("/api/fund/001302")

    @task(2)
    def get_news_list(self):
        """获取新闻列表"""
        self.client.get("/api/news/list?days=1")

    @task(1)
    def get_market_sentiment(self):
        """获取市场情绪"""
        self.client.get("/api/market/sentiment")
