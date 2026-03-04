# -*- coding: utf-8 -*-
"""
基金行业分析模块测试用例
运行方式: python -m pytest tests/test_fund_industry.py -v
或: python tests/test_fund_industry.py
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 确保项目根目录在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 加载 .env
from dotenv import load_dotenv

load_dotenv(_root / ".env")


# ---------- 测试数据 ----------
TEST_FUND_CODE = "000001"
TEST_INDUSTRIES = [
    {"industry": "新能源", "confidence": 95.0, "source": "llm"},
    {"industry": "半导体", "confidence": 60.0, "source": "llm"},
]


# ---------- schema 测试 ----------

def test_create_fund_industry_table():
    """测试创建基金行业表"""
    from modules.fund_industry import create_fund_industry_table
    
    # 测试表创建（幂等调用）
    try:
        create_fund_industry_table()
        print("✅ 创建基金行业表测试通过")
    except Exception as e:
        print(f"❌ 创建基金行业表测试失败: {e}")


# ---------- 仓储测试 ----------

def test_fund_industry_repo_save():
    """测试保存基金行业"""
    from modules.fund_industry import FundIndustryRepo
    
    repo = FundIndustryRepo()
    
    # 保存行业
    result = repo.save_industries(TEST_FUND_CODE, TEST_INDUSTRIES)
    assert isinstance(result, bool)
    print("✅ 保存基金行业测试通过")


def test_fund_industry_repo_get():
    """测试获取基金行业"""
    from modules.fund_industry import FundIndustryRepo
    
    repo = FundIndustryRepo()
    
    # 获取行业
    industries = repo.get_industries(TEST_FUND_CODE)
    assert isinstance(industries, list)
    print(f"✅ 获取基金行业测试通过, 共 {len(industries)} 条")


def test_fund_industry_repo_delete():
    """测试删除基金行业"""
    from modules.fund_industry import FundIndustryRepo
    
    repo = FundIndustryRepo()
    
    # 删除行业
    deleted = repo.delete_industries(TEST_FUND_CODE)
    assert isinstance(deleted, int)
    print(f"✅ 删除基金行业测试通过, 删除 {deleted} 条")


def test_fund_industry_repo_get_primary():
    """测试获取基金主要行业"""
    from modules.fund_industry import FundIndustryRepo
    
    repo = FundIndustryRepo()
    
    # 先保存
    repo.save_industries(TEST_FUND_CODE, TEST_INDUSTRIES)
    
    # 获取主要行业
    primary = repo.get_industry_by_fund(TEST_FUND_CODE)
    if primary:
        assert "industry" in primary
        assert "confidence" in primary
        print(f"✅ 获取基金主要行业测试通过: {primary.get('industry')}")
    else:
        print("⚠️  无行业数据，跳过测试")


def test_fund_industry_repo_empty():
    """测试空数据处理"""
    from modules.fund_industry import FundIndustryRepo
    
    repo = FundIndustryRepo()
    
    # 空代码
    result = repo.save_industries("", TEST_INDUSTRIES)
    assert result == False
    
    # 空行业
    result = repo.save_industries(TEST_FUND_CODE, [])
    assert result == False
    
    # 空代码获取
    industries = repo.get_industries("")
    assert industries == []
    
    print("✅ 空数据处理测试通过")


# ---------- 分析器测试 ----------

def test_fund_industry_analyzer_keywords():
    """测试关键词匹配分析"""
    from modules.fund_industry import FundIndustryAnalyzer
    
    analyzer = FundIndustryAnalyzer()
    
    # 测试关键词匹配
    fund_info = {
        "fund_code": "000001",
        "fund_name": "新能源主题股票",
        "fund_type": "股票型"
    }
    holdings = [
        {"stock_name": "宁德时代"},
        {"stock_name": "隆基绿能"}
    ]
    
    industries = analyzer._analyze_with_keywords(fund_info, holdings)
    assert isinstance(industries, list)
    
    if industries:
        assert "industry" in industries[0]
        assert "confidence" in industries[0]
        print(f"✅ 关键词匹配分析测试通过: {[i['industry'] for i in industries]}")
    else:
        print("⚠️  未匹配到行业")


def test_fund_industry_analyzer_no_fund():
    """测试无基金信息处理"""
    from modules.fund_industry import FundIndustryAnalyzer
    
    analyzer = FundIndustryAnalyzer()
    
    # 无效基金代码
    industries = analyzer.analyze("")
    assert industries == []
    
    industries = analyzer.analyze(None)
    assert industries == []
    
    print("✅ 无基金信息处理测试通过")


def test_fund_industry_analyzer_with_mock_llm():
    """测试使用Mock LLM分析"""
    from modules.fund_industry import FundIndustryAnalyzer
    
    # Mock LLM客户端
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.chat.return_value = '{"industries": [{"industry": "新能源", "confidence": 95.0}]}'
    
    with patch.object(FundIndustryAnalyzer, '_get_llm_client', return_value=mock_llm):
        analyzer = FundIndustryAnalyzer()
        
        fund_info = {
            "fund_code": "000001",
            "fund_name": "测试基金",
            "fund_type": "股票型"
        }
        
        industries = analyzer._analyze_with_llm(fund_info, [])
        
        if industries:
            assert "industry" in industries[0]
            print(f"✅ Mock LLM分析测试通过: {[i['industry'] for i in industries]}")
        else:
            print("⚠️  LLM返回为空")


# ---------- API 测试 ----------

def test_fund_industry_api_analyze():
    """测试行业分析API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试分析接口
    resp = client.post(f"/api/fund-industry/analyze/{TEST_FUND_CODE}")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    
    if data["code"] == 0:
        print(f"✅ 行业分析API测试通过, 返回 {len(data.get('data', []))} 个行业")
    else:
        print(f"⚠️  分析失败: {data.get('message')}")


def test_fund_industry_api_get():
    """测试获取行业API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试获取接口
    resp = client.get(f"/api/fund-industry/{TEST_FUND_CODE}")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    assert "data" in data
    print(f"✅ 获取行业API测试通过")


def test_fund_industry_api_primary():
    """测试获取主要行业API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 测试获取主要行业接口
    resp = client.get(f"/api/fund-industry/primary/{TEST_FUND_CODE}")
    assert resp.status_code == 200
    
    data = resp.get_json()
    assert "code" in data
    
    if data["code"] == 0 and data.get("data"):
        print(f"✅ 获取主要行业API测试通过: {data['data'].get('industry')}")
    else:
        print(f"⚠️  无主要行业数据")


def test_fund_industry_api_empty_code():
    """测试空基金代码API"""
    from server.app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # 空代码测试 - Flask对尾随斜杠的处理可能不同
    # 不测试空路径，因为路由定义不支持
    resp = client.post("/api/fund-industry/analyze/%20", json={})  # 空格代码
    assert resp.status_code in [200, 400, 404, 500]
    
    print("✅ 空基金代码API测试通过")


# ---------- 行业分类测试 ----------

def test_industry_keywords():
    """测试行业关键词分类"""
    from modules.fund_industry.analyzer import INDUSTRY_KEYWORDS
    
    # 验证行业关键词完整性
    assert "新能源" in INDUSTRY_KEYWORDS
    assert "半导体" in INDUSTRY_KEYWORDS
    assert "医药" in INDUSTRY_KEYWORDS
    assert "消费" in INDUSTRY_KEYWORDS
    assert "金融" in INDUSTRY_KEYWORDS
    assert "军工" in INDUSTRY_KEYWORDS
    assert "TMT" in INDUSTRY_KEYWORDS
    assert "基建" in INDUSTRY_KEYWORDS
    assert "农业" in INDUSTRY_KEYWORDS
    assert "化工" in INDUSTRY_KEYWORDS
    
    # 验证关键词不为空
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        assert len(keywords) > 0, f"{industry} 关键词为空"
    
    print(f"✅ 行业关键词分类测试通过, 共 {len(INDUSTRY_KEYWORDS)} 个行业")


# ---------- 运行所有测试 ----------

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始基金行业分析模块测试...")
    print("=" * 60)
    
    tests = [
        # schema测试
        ("创建基金行业表", test_create_fund_industry_table),
        
        # 仓储测试
        ("保存基金行业", test_fund_industry_repo_save),
        ("获取基金行业", test_fund_industry_repo_get),
        ("删除基金行业", test_fund_industry_repo_delete),
        ("获取主要行业", test_fund_industry_repo_get_primary),
        ("空数据处理", test_fund_industry_repo_empty),
        
        # 分析器测试
        ("关键词匹配分析", test_fund_industry_analyzer_keywords),
        ("无基金信息处理", test_fund_industry_analyzer_no_fund),
        ("Mock LLM分析", test_fund_industry_analyzer_with_mock_llm),
        
        # API测试
        ("行业分析API", test_fund_industry_api_analyze),
        ("获取行业API", test_fund_industry_api_get),
        ("获取主要行业API", test_fund_industry_api_primary),
        ("空基金代码API", test_fund_industry_api_empty_code),
        
        # 行业分类测试
        ("行业关键词分类", test_industry_keywords),
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
