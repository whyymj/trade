# Fund Svc Agent - 基金服务 Agent

## 角色定义

你是 FundProphet 微服务架构的基金服务 Agent，负责基金数据爬取、LSTM 预测、指标计算和 API 开发。

## 核心职责

1. **服务创建**: 创建独立的 Flask 应用
2. **数据层迁移**: 迁移基金数据相关代码
3. **API 开发**: 开发基金相关 API 端点
4. **定时任务**: 配置基金净值同步任务

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/fund/app.py`

**实现要求**:
```python
# services/fund/app.py
from flask import Flask, jsonify
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'fund-service'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'fund-service',
        'uptime': '0'
    })

# 导入路由
from services.fund.routes import fund_bp
app.register_blueprint(fund_bp, url_prefix='/api/fund')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8002))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移数据层

**源文件**:
- `data/fund_repo.py`
- `data/fund_fetcher.py`

**目标目录**: `services/fund/data/`

**实现要求**:
```python
# services/fund/data/__init__.py
from .fund_repo import FundRepo
from .fund_fetcher import FundFetcher

__all__ = ['FundRepo', 'FundFetcher']

# services/fund/data/fund_repo.py
from shared.db import fetch_all, execute

class FundRepo:
    def get_fund_list(self, page=1, size=20):
        """获取基金列表"""
        offset = (page - 1) * size
        sql = """
        SELECT fund_code, fund_name, fund_type, risk_level
        FROM fund_meta
        LIMIT %s OFFSET %s
        """
        return fetch_all(sql, (size, offset))

    def get_fund_info(self, fund_code):
        """获取基金详情"""
        sql = "SELECT * FROM fund_meta WHERE fund_code = %s"
        result = fetch_all(sql, (fund_code,))
        return result[0] if result else None

    def get_fund_nav(self, fund_code, days=30):
        """获取基金净值"""
        sql = """
        SELECT nav_date, unit_nav, accum_nav, daily_return
        FROM fund_nav
        WHERE fund_code = %s
        ORDER BY nav_date DESC
        LIMIT %s
        """
        return fetch_all(sql, (fund_code, days))

    def upsert_fund_nav(self, fund_code, df):
        """批量插入/更新基金净值"""
        # 实现 upsert 逻辑
        pass

# services/fund/data/fund_fetcher.py
import requests
import pandas as pd

class FundFetcher:
    BASE_URL = "http://fund.eastmoney.com"

    def fetch_fund_nav(self, fund_code, days=30):
        """从天天基金网获取基金净值"""
        url = f"{self.BASE_URL}/{fund_code}.html"
        # 实现爬取逻辑
        pass
```

**测试文件**: `services/fund/tests/test_data.py`
```python
import pytest
from services.fund.data import FundRepo

def test_get_fund_list():
    """测试获取基金列表"""
    repo = FundRepo()
    result = repo.get_fund_list(page=1, size=10)
    assert len(result) <= 10

def test_get_fund_info():
    """测试获取基金详情"""
    repo = FundRepo()
    result = repo.get_fund_info('001302')
    assert result is not None
```

---

### 任务3: 基金列表 API

**目标文件**: `services/fund/routes/fund.py`

**实现要求**:
```python
# services/fund/routes/__init__.py
from flask import Blueprint
from .fund import fund_bp

__all__ = ['fund_bp']

# services/fund/routes/fund.py
from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.fund.data import FundRepo

fund_bp = Blueprint('fund', __name__)
cache = get_cache()
repo = FundRepo()

@fund_bp.route('/list', methods=['GET'])
def get_fund_list():
    """获取基金列表"""
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 20))

    # 缓存
    cache_key = f'fund_list_{page}_{size}'
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result})

    # 查询
    result = repo.get_fund_list(page=page, size=size)

    # 缓存 30 分钟
    cache.set(cache_key, result, ttl=1800)

    return jsonify({'success': True, 'data': result})
```

**测试文件**: `services/fund/tests/test_routes.py`
```python
import pytest
from services.fund.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_fund_list(client):
    """测试基金列表 API"""
    resp = client.get('/api/fund/list?page=1&size=10')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'data' in data
```

---

### 任务4: 基金详情 API

**目标文件**: `services/fund/routes/fund.py`

**实现要求**:
```python
@fund_bp.route('/<fund_code>', methods=['GET'])
def get_fund_detail(fund_code):
    """获取基金详情"""
    # 缓存
    cache_key = f'fund_detail_{fund_code}'
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result})

    # 查询
    result = repo.get_fund_info(fund_code)

    if not result:
        return jsonify({'success': False, 'message': '基金不存在'}), 404

    # 缓存 1 小时
    cache.set(cache_key, result, ttl=3600)

    return jsonify({'success': True, 'data': result})
```

---

### 任务5: LSTM 预测 API

**目标文件**: `services/fund/routes/predict.py`

**实现要求**:
```python
# services/fund/routes/predict.py
from flask import Blueprint, jsonify
from shared.cache import get_cache
from services.fund.analysis.fund_lstm import FundLSTMPredictor

predict_bp = Blueprint('predict', __name__)
cache = get_cache()
predictor = FundLSTMPredictor()

@predict_bp.route('/<fund_code>/predict', methods=['GET'])
def predict_fund(fund_code):
    """LSTM 预测基金走势"""
    # 缓存
    cache_key = f'fund_predict_{fund_code}'
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result})

    # 预测
    result = predictor.predict(fund_code)

    # 缓存 5 分钟
    cache.set(cache_key, result, ttl=300)

    return jsonify({'success': True, 'data': result})
```

---

### 任务6: 定时任务 - 基金净值同步

**目标文件**: `scheduler/fund_sync.py`

**实现要求**:
```python
# scheduler/fund_sync.py
import redis
from shared.messaging import get_mq, FUND_SYNC
from services.fund.data import FundRepo, FundFetcher

def sync_all_funds():
    """同步所有基金净值"""
    repo = FundRepo()
    fetcher = FundFetcher()

    result = repo.get_fund_list(page=1, size=1000)
    fund_codes = [f['fund_code'] for f in result.get('data', [])]

    success = 0
    for code in fund_codes:
        try:
            df = fetcher.fetch_fund_nav(code, days=30)
            if df is not None and not df.empty:
                repo.upsert_fund_nav(code, df)
                success += 1
        except Exception as e:
            print(f"同步失败 {code}: {e}")

    # 发布消息
    mq = get_mq()
    mq.publish(FUND_SYNC, {
        'success': success,
        'total': len(fund_codes)
    })

    print(f"[定时任务] 基金数据同步完成: {success}/{len(fund_codes)}")

if __name__ == '__main__':
    sync_all_funds()
```

**Cron 配置**: 每天 03:00

---

## 交付物

### 目录结构

```
services/fund/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── data/                     # 数据层
│   ├── __init__.py
│   ├── fund_repo.py
│   └── fund_fetcher.py
├── analysis/                 # 分析模块
│   ├── __init__.py
│   ├── fund_lstm.py
│   └── fund_metrics.py
├── routes/                   # 路由
│   ├── __init__.py
│   ├── fund.py
│   └── predict.py
├── tests/                    # 测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_routes.py
└── Dockerfile               # Docker 配置
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] 基金列表 API 正常
- [ ] 基金详情 API 正常
- [ ] LSTM 预测 API 正常
- [ ] 定时任务正常执行
- [ ] 单元测试通过
- [ ] API 文档完整

---

## 依赖

- Infra Agent: Redis, MySQL 连接池
- Coordinator Agent: 任务分发
- QA Agent: 测试验收

---

## 立即开始

你现在需要：

1. **开始任务1**: 创建 Flask 应用骨架
   - 创建 `services/fund/app.py`
   - 配置路由

2. **开始任务2**: 迁移数据层
   - 创建 `services/fund/data/`
   - 迁移 `fund_repo.py`, `fund_fetcher.py`
   - 编写测试

3. **开始任务3**: 开发基金列表 API
   - 创建 `services/fund/routes/fund.py`
   - 实现缓存逻辑
   - 编写测试

4. **开始任务4**: 开发基金详情 API
   - 在 `fund.py` 中添加详情接口
   - 实现缓存逻辑

5. **开始任务5**: 开发 LSTM 预测 API
   - 创建 `services/fund/routes/predict.py`
   - 迁移 `fund_lstm.py`

6. **开始任务6**: 配置定时任务
   - 创建 `scheduler/fund_sync.py`
   - 配置 Cron

**准备就绪了吗？开始开发基金服务！**