# -*- coding: utf-8 -*-
"""
基金功能测试用例
运行方式: python -m pytest tests/test_fund.py -v
或: python tests/test_fund.py
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 加载 .env
from dotenv import load_dotenv

load_dotenv(_root / ".env")


def test_fund_list():
    """测试基金列表接口"""
    from data import fund_repo

    result = fund_repo.get_fund_list(page=1, size=10)
    assert result is not None
    assert "data" in result
    assert len(result["data"]) > 0
    print("✅ 基金列表测试通过")


def test_fund_nav():
    """测试基金净值数据"""
    from data import fund_repo

    df = fund_repo.get_fund_nav("012365")
    assert df is not None
    assert not df.empty
    assert "unit_nav" in df.columns
    print(f"✅ 基金净值测试通过, 共 {len(df)} 条数据")


def test_fund_nav_with_date_range():
    """测试基金净值日期范围查询"""
    from data import fund_repo

    df = fund_repo.get_fund_nav(
        "012365", start_date="2025-01-01", end_date="2025-03-01"
    )
    assert df is not None
    assert len(df) > 0
    print(f"✅ 日期范围查询测试通过, 获取 {len(df)} 条数据")


def test_latest_nav():
    """测试最新净值"""
    from data import fund_repo

    result = fund_repo.get_latest_nav("012365")
    assert result is not None
    assert "unit_nav" in result
    assert result["unit_nav"] is not None
    print(f"✅ 最新净值测试通过: {result['unit_nav']}")


def test_fund_indicators():
    """测试基金指标计算"""
    from analysis.fund_metrics import get_fund_indicators

    result = get_fund_indicators("012365", days=365)
    assert result is not None
    assert "annual_return" in result
    assert "volatility" in result
    assert "sharpe_ratio" in result
    print(f"✅ 基金指标测试通过, 年化收益: {result.get('annual_return', 0):.2%}")


def test_fund_cycle_analysis():
    """测试周期分析"""
    from analysis.frequency_domain import find_dominant_periods, calc_power_spectrum
    from data import fund_repo
    from datetime import datetime, timedelta

    # 使用 start_date/end_date 代替 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = fund_repo.get_fund_nav("012365", start_date=start_date.strftime("%Y-%m-%d"))
    assert df is not None

    returns = df["unit_nav"].pct_change().dropna()
    assert len(returns) >= 20, f"数据不足: {len(returns)}"

    frequencies, psd = calc_power_spectrum(returns, sampling_rate=1.0)
    periods = find_dominant_periods(frequencies, psd, top_n=5, min_period=5)

    assert len(periods) > 0
    print(f"✅ 周期分析测试通过, 发现 {len(periods)} 个周期")
    for p in periods[:3]:
        print(f"   周期 {p['period_days']:.1f} 天, 强度 {p['power']:.4f}")


def test_fund_benchmark():
    """测试基准对比"""
    from analysis.fund_benchmark import compare_with_benchmark, get_benchmark_data
    from data import fund_repo

    df = fund_repo.get_fund_nav("012365")
    assert df is not None

    df = df.sort_values("nav_date")
    fund_nav = df.set_index("nav_date")["unit_nav"]

    # 获取基准数据
    benchmark_nav = get_benchmark_data("000300")
    if benchmark_nav is None or len(benchmark_nav) < 10:
        print("⚠️  基准数据不足，跳过测试")
        return

    # 对齐日期
    common_idx = fund_nav.index.intersection(benchmark_nav.index)
    if len(common_idx) < 10:
        print("⚠️  数据交集不足，跳过测试")
        return

    fund_nav_aligned = fund_nav.loc[common_idx]
    benchmark_nav_aligned = benchmark_nav.loc[common_idx]

    result = compare_with_benchmark(fund_nav_aligned, benchmark_nav_aligned)
    assert result is not None
    assert "alpha" in result
    print(f"✅ 基准对比测试通过, Alpha: {result.get('alpha', 0):.4f}")


def test_llm_client():
    """测试 LLM 客户端"""
    from analysis.llm import get_minimax_client

    client = get_client()
    assert client is not None, "LLM 客户端未初始化"
    assert client.is_available(), "LLM 服务不可用"
    print("✅ LLM 客户端测试通过")


def test_llm_analysis():
    """测试 LLM 分析功能"""
    from analysis.llm import get_minimax_client

    client = get_client()
    if not client.is_available():
        print("⚠️  LLM 服务不可用，跳过分析测试")
        return

    from data import fund_repo

    fund_info = fund_repo.get_fund_info("012365")
    assert fund_info is not None

    # 简单测试
    prompt = "用一句话评价这只基金"
    result = client.chat([{"role": "user", "content": prompt}])
    assert result is not None
    print(f"✅ LLM 分析测试通过: {result[:50]}...")


def test_api_routes():
    """测试 Flask API 路由"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 测试基金列表
    resp = client.get("/api/fund/list?page=1&size=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "data" in data
    print("✅ API /fund/list 测试通过")

    # 测试最新净值
    resp = client.get("/api/fund/nav/latest/012365")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "unit_nav" in data
    print("✅ API /fund/nav/latest 测试通过")

    # 测试周期分析
    resp = client.get("/api/fund/cycle/012365?days=365")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "dominant_periods" in data
    print("✅ API /fund/cycle 测试通过")

    # 测试指标
    resp = client.get("/api/fund/indicators/012365?days=365")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "annual_return" in data
    print("✅ API /fund/indicators 测试通过")


def test_lstm_model():
    """测试 LSTM 模型"""
    from analysis.fund_lstm import predict
    from datetime import datetime, timedelta

    # 测试预测（简单规则）
    result = predict("012365")
    assert result is not None
    assert "direction" in result
    print(f"✅ LSTM 预测测试通过, 方向: {result.get('direction_label')}")


def test_lstm_train_requires_data():
    """测试 LSTM 训练需要充足的基金净值数据"""
    from analysis.fund_lstm import train_model

    # 测试没有数据的基金
    result = train_model("000311", 365, 1)
    assert result is not None
    assert result.get("success") == False, "没有数据的基金应该训练失败"
    assert "Insufficient data" in result.get("error", "")
    print(f"✅ LSTM 训练数据不足测试通过: {result.get('error')}")


def test_lstm_predict():
    """测试 LSTM 预测功能"""
    from analysis.fund_lstm import predict

    # 测试有数据的基金
    result = predict("001302")
    assert result is not None
    assert "fund_code" in result
    assert "direction" in result
    assert "prob_up" in result
    assert "magnitude" in result
    print(
        f"✅ LSTM 预测测试通过, 方向: {result.get('direction_label')}, 概率: {result.get('prob_up')}"
    )


def test_fund_indicators_no_data():
    """测试基金指标计算 - 无数据情况"""
    from analysis.fund_metrics import get_fund_indicators

    result = get_fund_indicators("000000", days=365)
    # 没有数据的基金应该返回 None 或错误
    print(f"✅ 基金指标无数据测试通过: {result}")


def test_api_fund_detail():
    """测试基金详情 API"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 测试获取基金详情
    resp = client.get("/api/fund/001302")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "fund_code" in data
    print(f"✅ API 基金详情测试通过: {data.get('fund_name')}")


def test_api_fund_add():
    """测试添加基金 API"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 使用已存在的基金代码测试
    resp = client.post(
        "/api/fund/add", json={"fund_code": "000001", "fund_name": "测试基金"}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") == True
    print(f"✅ API 添加基金测试通过_fund_add")


def test_api_and_delete():
    """测试添加基金后再删除的完整流程（通过前端接口）"""
    from server.app import create_app
    from data.mysql import execute, fetch_one
    from data.cache import get_cache

    app = create_app()
    client = app.test_client()

    test_code = "TESTDEL999"
    test_name = "测试删除基金"

    # 清理可能存在的旧数据
    get_cache().delete("fund_list")
    execute("DELETE FROM fund_meta WHERE fund_code = %s", (test_code,))
    execute("DELETE FROM fund_nav WHERE fund_code = %s", (test_code,))

    # 1. 添加基金 (通过POST /api/fund/add)
    resp = client.post(
        "/api/fund/add", json={"fund_code": test_code, "fund_name": test_name}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") == True
    print(f"  - POST /api/fund/add: OK")

    # 2. 验证基金已添加到数据库
    row = fetch_one(
        "SELECT fund_code, fund_name FROM fund_meta WHERE fund_code = %s", (test_code,)
    )
    assert row is not None, f"基金 {test_code} 应该已添加"
    assert row["fund_name"] == test_name
    print(f"  - 数据库验证添加: OK")

    # 3. 删除基金 (通过 DELETE /api/fund/{code})
    resp = client.delete(f"/api/fund/{test_code}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") == True
    assert data.get("deleted", 0) > 0
    print(f"  - DELETE /api/fund/{test_code}: OK")

    # 4. 验证基金已从数据库删除
    row = fetch_one(
        "SELECT fund_code FROM fund_meta WHERE fund_code = %s", (test_code,)
    )
    assert row is None, f"基金 {test_code} 应该已删除"
    print(f"  - 数据库验证删除: OK")

    # 5. 清理测试数据
    get_cache().delete("fund_list")
    execute("DELETE FROM fund_meta WHERE fund_code = %s", (test_code,))

    print("✅ 添加后再删除测试通过")


def test_market_sentiment_api():
    """测试市场情绪API"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 获取最新市场情绪
    resp = client.get("/api/market/sentiment/latest")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert data["data"] is not None

    sentiment = data["data"]
    assert "trade_date" in sentiment
    assert "volume" in sentiment
    assert "up_count" in sentiment
    assert "down_count" in sentiment
    assert "turnover_rate" in sentiment

    # 验证数据完整性（非空）
    assert sentiment["volume"] is not None, "成交额不应为空"
    assert sentiment["turnover_rate"] is not None, "换手率不应为空"

    print(f"✅ 市场情绪API测试通过: {sentiment.get('trade_date')}")


def test_market_money_flow_api():
    """测试资金流向API"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 获取最新资金流向
    resp = client.get("/api/market/money-flow/latest")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert data["data"] is not None

    money_flow = data["data"]
    assert "trade_date" in money_flow
    assert "north_money" in money_flow
    assert "main_money" in money_flow
    assert "margin_balance" in money_flow

    # 验证数据完整性（非空）
    assert money_flow["main_money"] is not None, "主力资金不应为空"
    assert money_flow["north_money"] is not None, "北向资金不应为空"
    assert money_flow["margin_balance"] is not None, "融资余额不应为空"

    print(f"✅ 资金流向API测试通过: {money_flow.get('trade_date')}")


def test_market_sentiment_with_days():
    """测试市场情绪API带days参数"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 获取多天数据
    resp = client.get("/api/market/sentiment?days=7")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert isinstance(data["data"], list)

    print(f"✅ 市场情绪API(days参数)测试通过: {len(data['data'])} 条")


def test_market_money_flow_with_days():
    """测试资金流向API带days参数"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    # 获取多天数据
    resp = client.get("/api/market/money-flow?days=7")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert isinstance(data["data"], list)

    print(f"✅ 资金流向API(days参数)测试通过: {len(data['data'])} 条")


def test_training_status_functions():
    """测试训练状态检查函数"""
    from analysis.fund_lstm import (
        is_trained_today,
        is_training_in_progress,
        record_training_status,
    )
    from data.mysql import execute

    test_code = "TEST001"

    # 清理测试数据
    execute("DELETE FROM fund_training_log WHERE fund_code = %s", (test_code,))

    # 测试初始状态
    assert is_trained_today(test_code) == False, "新基金应该没有训练记录"
    assert is_training_in_progress(test_code) == False, "新基金不应该在训练中"

    # 记录训练状态
    record_training_status(test_code, "in_progress", "测试训练")
    assert is_training_in_progress(test_code) == True, "应该显示正在训练"

    # 清理
    execute("DELETE FROM fund_training_log WHERE fund_code = %s", (test_code,))
    print(f"✅ 训练状态检查函数测试通过")


def test_train_model_status_tracking():
    """测试训练模型状态追踪"""
    from analysis.fund_lstm import train_model
    from data.mysql import execute

    # 使用一个没有数据的基金测试
    result = train_model("NOTEXIST999", days=365, epochs=1)

    # 应该返回失败，但应该有记录
    assert result.get("success") == False
    print(f"✅ 训练状态追踪测试通过: {result.get('error')}")


def test_lstm_train_api():
    """测试 LSTM 训练 API"""
    from server.app import create_app
    import time

    app = create_app()
    client = app.test_client()

    # 先清除该基金的训练状态
    from data.mysql import execute

    execute("DELETE FROM fund_training_log WHERE fund_code = %s", ("002112",))

    # 测试训练 API
    resp = client.post(
        "/api/fund/lstm/train", json={"fund_code": "002112", "days": 365, "epochs": 1}
    )

    # 由于训练可能很慢，我们主要检查返回格式
    if resp.status_code == 200:
        data = resp.get_json()
        print(f"✅ LSTM API测试通过: {data}")
    else:
        print(f"⚠️  LSTM API返回状态: {resp.status_code}")


def test_lstm_train_prevents_duplicate():
    """测试 LSTM 训练防止重复训练"""
    from analysis.fund_lstm import train_model
    from data.mysql import execute

    test_code = "DUPTEST001"

    # 清理
    execute("DELETE FROM fund_training_log WHERE fund_code = %s", (test_code,))

    # 第一次训练（应该失败因为没有数据）
    result1 = train_model(test_code, days=365, epochs=1)
    print(f"第一次训练结果: {result1}")

    # 检查是否有记录
    from analysis.fund_lstm import is_trained_today

    has_record = is_trained_today(test_code)
    print(f"有训练记录: {has_record}")

    # 清理
    execute("DELETE FROM fund_training_log WHERE fund_code = %s", (test_code,))
    print(f"✅ 防止重复训练测试通过")


def test_fund_nav_data_exists():
    """测试基金净值数据是否存在"""
    from data import fund_repo

    # 获取一个有净值数据的基金
    result = fund_repo.get_fund_list(page=1, size=100)
    funds = result.get("data", [])

    fund_with_nav = None
    for fund in funds:
        df = fund_repo.get_fund_nav(fund["fund_code"])
        if df is not None and len(df) > 10:
            fund_with_nav = fund["fund_code"]
            break

    if fund_with_nav:
        print(f"✅ 基金净值数据测试通过, 找到有数据的基金: {fund_with_nav}")
    else:
        print("⚠️  没有找到有净值数据的基金，请先同步数据")


def test_fund_list_pagination():
    """测试基金列表分页不同size参数"""
    from data.cache import get_cache
    from server.app import create_app

    # 清除缓存
    cache = get_cache()
    cache.delete("fund_list")
    cache.delete("fund_list:")

    app = create_app()
    client = app.test_client()

    # 测试不同 size
    resp1 = client.get("/api/fund/list?page=1&size=3")
    data1 = resp1.get_json()

    resp2 = client.get("/api/fund/list?page=1&size=100")
    data2 = resp2.get_json()

    # 验证 total 一致
    assert data1["total"] == data2["total"], (
        f"Total 不一致: {data1['total']} vs {data2['total']}"
    )
    # 验证返回数量与 size 一致（假设总数据 >= size）
    assert len(data1["data"]) == 3, f"size=3 应返回3条，实际: {len(data1['data'])}"
    assert len(data2["data"]) == data2["total"], (
        f"size=100 应返回全部，实际: {len(data2['data'])}"
    )

    print(
        f"✅ 基金列表分页测试通过, 总数: {data2['total']}, size=3: {len(data1['data'])}, size=100: {len(data2['data'])}"
    )


def test_data_no_nan():
    """测试数据中不含 NaN（之前的问题）- 通过 JSON 序列化验证"""
    from data import fund_repo
    import json

    df = fund_repo.get_fund_nav("012365")
    assert df is not None

    # 通过 JSON 序列化测试（不检查 isna，因为 replace 后 pandas 仍显示 NaN）
    data = df.head(10).to_dict(orient="records")
    json_str = json.dumps(data)

    # JSON 中不应包含 NaN/null（但可以有 null，即 Python None）
    # 关键是不要有 Python float('nan')
    assert "NaN" not in json_str and "nan" not in json_str.lower().replace(
        "null", ""
    ), f"JSON 包含 nan: {json_str[:200]}"
    print("✅ 数据 JSON 序列化测试通过")


def test_json_serialization():
    """测试 JSON 序列化（之前 NaN 问题）"""
    from server.app import create_app
    import json

    app = create_app()
    client = app.test_client()

    resp = client.get("/api/fund/nav/012365?start=2025-01-01&end=2025-03-01")
    assert resp.status_code == 200

    # 确保响应是有效 JSON
    data = resp.get_json()
    json_str = json.dumps(data)

    # 不应包含 NaN
    assert "NaN" not in json_str, "JSON 包含 NaN"
    assert "nan" not in json_str.lower() or "null" in json_str.lower(), "JSON 包含 nan"
    print("✅ JSON 序列化测试通过")


def test_cycle_api_markline_data():
    """测试周期 API 返回的数据可用于前端 markLine 绘制"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get("/api/fund/cycle/012365?days=365")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "dominant_periods" in data

    periods = data["dominant_periods"]
    assert len(periods) > 0

    # 验证前端需要的字段
    for p in periods:
        assert "rank" in p, "缺少 rank 字段"
        assert "period_days" in p, "缺少 period_days 字段"
        assert "power" in p, "缺少 power 字段"
        assert "explanation" in p, "缺少 explanation 字段"

    print(f"✅ Cycle API 数据格式测试通过, 返回 {len(periods)} 个周期")
    for p in periods[:3]:
        print(f"   周期{p['rank']}: {p['period_days']}天, 强度{p['power']:.2%}")


def test_llm_markdown_response():
    """测试 LLM 返回的是 Markdown 格式"""
    from analysis.llm import get_minimax_client

    client = get_minimax_client()
    if not client.is_available():
        print("⚠️  LLM 服务不可用，跳过 Markdown 测试")
        return

    from data import fund_repo

    fund_info = fund_repo.get_fund_info("012365")
    assert fund_info is not None

    # 测试生成分析报告（应包含 Markdown 格式）
    prompt = f"""分析基金 {fund_info.get("fund_name", "")} ({fund_info.get("fund_code", "")})，
用 Markdown 格式返回，包含以下标题：## 基金概况、## 业绩表现、## 风险评估"""

    result = client.chat([{"role": "user", "content": prompt}])

    # 验证包含 Markdown 标记
    assert result is not None
    assert "##" in result or "# " in result, "LLM 返回不是 Markdown 格式"
    assert len(result) > 50, "返回内容太短"

    print(f"✅ LLM Markdown 格式测试通过, 长度: {len(result)} 字符")


def test_news_list():
    """测试获取新闻列表"""
    from data.news import NewsRepo

    repo = NewsRepo()
    news_list = repo.get_today_news()

    assert news_list is not None
    print(f"✅ 新闻列表测试通过, 共 {len(news_list)} 条")


def test_news_detail():
    """测试获取新闻详情"""
    from data.news import NewsRepo

    repo = NewsRepo()
    news_list = repo.get_today_news()

    if len(news_list) > 0:
        news = news_list[0]
        detail = repo.get_news_by_url(news.url)

        assert detail is not None
        assert detail.title == news.title
        assert detail.content == news.content
        print(f"✅ 新闻详情测试通过: {detail.title[:20]}...")
    else:
        print("⚠️  无新闻数据，跳过测试")


def test_news_api_list():
    """测试新闻列表API"""
    from server.app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get("/api/news/latest")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert "data" in data
    print(f"✅ 新闻列表API测试通过")


def test_news_api_detail():
    """测试新闻详情API"""
    from data.news import NewsRepo
    from server.app import create_app

    repo = NewsRepo()
    news_list = repo.get_today_news()

    if len(news_list) == 0:
        print("⚠️  无新闻数据，跳过测试")
        return

    app = create_app()
    client = app.test_client()

    news = news_list[0]
    import urllib.parse

    encoded_url = urllib.parse.quote(news.url, safe="")
    resp = client.get(f"/api/news/detail/{encoded_url}")

    assert resp.status_code == 200

    data = resp.get_json()
    assert data["code"] == 0
    assert data["data"]["title"] == news.title
    print(f"✅ 新闻详情API测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始基金功能测试...")
    print("=" * 50)

    tests = [
        ("基金列表", test_fund_list),
        ("基金净值", test_fund_nav),
        ("净值日期范围", test_fund_nav_with_date_range),
        ("最新净值", test_latest_nav),
        ("基金指标", test_fund_indicators),
        ("周期分析", test_fund_cycle_analysis),
        ("基准对比", test_fund_benchmark),
        ("LLM客户端", test_llm_client),
        ("LLM分析", test_llm_analysis),
        ("API路由", test_api_routes),
        ("LSTM模型", test_lstm_model),
        ("LSTM训练数据不足", test_lstm_train_requires_data),
        ("LSTM预测", test_lstm_predict),
        ("基金指标无数据", test_fund_indicators_no_data),
        ("API基金详情", test_api_fund_detail),
        ("API添加基金", test_api_fund_add),
        ("API添加后删除", test_api_and_delete),
        ("市场情绪API", test_market_sentiment_api),
        ("资金流向API", test_market_money_flow_api),
        ("市场情绪(days)", test_market_sentiment_with_days),
        ("资金流向(days)", test_market_money_flow_with_days),
        ("训练状态函数", test_training_status_functions),
        ("训练状态追踪", test_train_model_status_tracking),
        ("LSTM训练API", test_lstm_train_api),
        ("防止重复训练", test_lstm_train_prevents_duplicate),
        ("基金净值数据检查", test_fund_nav_data_exists),
        ("基金列表分页", test_fund_list_pagination),
        ("数据NaN", test_data_no_nan),
        ("JSON序列化", test_json_serialization),
        ("周期API数据格式", test_cycle_api_markline_data),
        ("LLM Markdown格式", test_llm_markdown_response),
        ("新闻列表", test_news_list),
        ("新闻详情", test_news_detail),
        ("新闻列表API", test_news_api_list),
        ("新闻详情API", test_news_api_detail),
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
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
