import pytest


class TestTemplate:
    """测试模板"""

    def test_example(self):
        """示例测试"""
        # Arrange (准备)
        expected = 1 + 1

        # Act (执行)
        result = 2

        # Assert (断言)
        assert result == expected

    @pytest.mark.unit
    def test_with_marker(self):
        """带标记的测试"""
        assert True

    @pytest.mark.unit
    def test_sample_fund(self, sample_fund):
        """测试示例基金 fixture"""
        assert sample_fund["fund_code"] == "TEST001"
        assert sample_fund["fund_name"] == "测试基金"

    @pytest.mark.unit
    def test_sample_news(self, sample_news):
        """测试示例新闻 fixture"""
        assert sample_news["title"] == "测试新闻"
        assert sample_news["source"] == "测试来源"

    @pytest.mark.unit
    def test_with_db_cleanup(self, clean_db):
        """测试数据库清理 fixture（需要数据库连接）"""
        # 这个测试需要数据库连接
        assert True
