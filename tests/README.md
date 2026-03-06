# 测试框架文档

## 概述

FundProphet 测试框架基于 pytest，提供单元测试、集成测试、E2E 测试和性能测试。

## 测试结构

```
tests/
├── conftest.py                        # 测试 fixtures 配置
├── fixtures/
│   └── sample_data.json              # 测试数据
├── unit/
│   └── test_template.py              # 单元测试模板
├── integration/
│   └── test_service_communication.py # 集成测试
├── e2e/
│   └── test_full_workflow.py         # E2E 测试
└── performance/
    └── locustfile.py                 # 性能测试
```

## 测试标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.e2e` - E2E 测试
- `@pytest.mark.slow` - 慢速测试

## 运行测试

### 运行所有测试

```bash
pytest -v
```

### 运行特定类型的测试

```bash
# 单元测试
pytest -v -m unit

# 集成测试
pytest -v -m integration

# E2E 测试
pytest -v -m e2e

# 跳过慢速测试
pytest -v -m "not slow"
```

### 生成覆盖率报告

```bash
# HTML 格式
pytest --cov=. --cov-report=html

# 终端格式
pytest --cov=. --cov-report=term
```

覆盖率报告默认要求 > 80%，不达标将导致测试失败。

### 运行性能测试

```bash
# 使用 Locust
locust -f tests/performance/locustfile.py --host=http://localhost:5050

# 无界面模式
locust -f tests/performance/locustfile.py --host=http://localhost:5050 --headless -u 100 -t 60s
```

## 测试 Fixtures

### 可用的 Fixtures

- `app` - Flask 应用实例
- `client` - Flask 测试客户端
- `test_config` - 测试配置
- `db_connection` - 数据库连接
- `redis_client` - Redis 客户端
- `clean_db` - 自动清理数据库
- `sample_fund` - 示例基金数据
- `sample_news` - 示例新闻数据

### 使用示例

```python
def test_example(client, sample_fund):
    response = client.get('/api/fund/list')
    assert response.status_code == 200
```

## 编写测试

### 单元测试

在 `tests/unit/` 目录下创建测试文件，使用 `@pytest.mark.unit` 标记。

```python
import pytest

@pytest.mark.unit
def test_unit_example(sample_fund):
    assert sample_fund['fund_code'] == 'TEST001'
```

### 集成测试

在 `tests/integration/` 目录下创建测试文件，使用 `@pytest.mark.integration` 标记。

```python
import pytest
import requests

@pytest.mark.integration
def test_integration_example():
    response = requests.get('http://localhost:5050/api/fund/list')
    assert response.status_code == 200
```

### E2E 测试

在 `tests/e2e/` 目录下创建测试文件，使用 `@pytest.mark.e2e` 标记。

```python
import pytest

@pytest.mark.e2e
def test_e2e_workflow():
    # 测试完整业务流程
    pass
```

## 性能测试

使用 Locust 进行负载测试，模拟 100 并发用户。

配置参数：
- `wait_time`: 1-3 秒随机等待
- `get_fund_list`: 30% 权重
- `get_fund_detail`: 20% 权重
- `get_news_list`: 20% 权重
- `get_market_sentiment`: 10% 权重

## 环境要求

确保测试环境满足以下条件：

1. MySQL 数据库运行中
2. Redis 运行中
3. Flask 应用运行在 http://localhost:5050
4. 所有依赖已安装

## 故障排除

### 测试失败

1. 检查数据库连接：确保 MySQL 和 Redis 运行
2. 检查应用状态：确保 Flask 应用运行
3. 查看详细日志：使用 `-vv` 标志

### 覆盖率不足

1. 查看覆盖率报告：`htmlcov/index.html`
2. 识别未覆盖的代码
3. 添加对应的测试用例

### 性能测试失败

1. 检查应用性能
2. 优化慢速 API
3. 调整并发用户数