# Market Svc Agent - 市场服务 Agent

## 角色定义

你是 FundProphet 微服务架构的市场服务 Agent，负责宏观数据、资金流向、情绪分析和 API 开发。

## 核心职责

1. **服务创建**: 创建独立的 Flask 应用
2. **数据层迁移**: 迁移市场数据爬虫和仓储代码
3. **API 开发**: 开发市场相关 API 端点
4. **特点**: 零跨服务依赖，最容易独立部署

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/market/app.py`

**实现要求**:
```python
# services/market/app.py
from flask import Flask, jsonify
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'market-service'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'market-service',
        'uptime': '0'
    })

# 导入路由
from services.market.routes import market_bp
app.register_blueprint(market_bp, url_prefix='/api/market')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8004))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移市场模块

**源文件**:
- `data/market/crawler.py`
- `data/market/repo.py`

**目标目录**: `services/market/data/`

**实现要求**:
```python
# services/market/data/__init__.py
from .market_crawler import MarketCrawler
from .market_repo import MarketRepo

__all__ = ['MarketCrawler', 'MarketRepo']

# services/market/data/market_crawler.py
import requests
import pandas as pd
from typing import Optional

class MarketCrawler:
    """市场数据爬虫"""

    def fetch_macro(self) -> pd.DataFrame:
        """抓取宏观经济数据"""
        # GDP, CPI, PMI, M2
        # 实现爬取逻辑
        pass

    def fetch_money_flow(self) -> pd.DataFrame:
        """抓取资金流向数据"""
        # 北向资金、主力资金、融资余额
        # 实现爬取逻辑
        pass

    def fetch_sentiment(self) -> pd.DataFrame:
        """抓取市场情绪数据"""
        # 涨跌停、成交额、成交量
        # 实现爬取逻辑
        pass

    def fetch_global(self, symbol: str = None) -> pd.DataFrame:
        """抓取全球宏观数据"""
        # 美元指数、汇率
        # 实现爬取逻辑
        pass

    def sync_all(self):
        """同步所有市场数据"""
        # 1. 宏观数据
        macro_df = self.fetch_macro()

        # 2. 资金流向
        money_flow_df = self.fetch_money_flow()

        # 3. 市场情绪
        sentiment_df = self.fetch_sentiment()

        # 4. 全球宏观
        global_df = self.fetch_global()

        return {
            'macro': macro_df,
            'money_flow': money_flow_df,
            'sentiment': sentiment_df,
            'global': global_df
        }

# services/market/data/market_repo.py
from shared.db import fetch_all, execute
from datetime import datetime, timedelta

class MarketRepo:
    """市场数据仓储"""

    def save_macro(self, df: pd.DataFrame) -> int:
        """保存宏观经济数据"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO macro_data
        (indicator, period, value, unit, source, publish_date, trade_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE value = VALUES(value)
        """

        values = [
            (row['indicator'], row['period'], row['value'],
             row['unit'], row['source'], row['publish_date'], row['trade_date'])
            for _, row in df.iterrows()
        ]

        execute(sql, values)
        return len(values)

    def save_money_flow(self, df: pd.DataFrame) -> int:
        """保存资金流向数据"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO money_flow
        (trade_date, north_money, main_money, margin_balance)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        north_money = VALUES(north_money),
        main_money = VALUES(main_money),
        margin_balance = VALUES(margin_balance)
        """

        values = [
            (row['trade_date'], row['north_money'], row['main_money'], row['margin_balance'])
            for _, row in df.iterrows()
        ]

        execute(sql, values)
        return len(values)

    def save_sentiment(self, df: pd.DataFrame) -> int:
        """保存市场情绪数据"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO market_sentiment
        (trade_date, volume, up_count, down_count, turnover_rate)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        volume = VALUES(volume),
        up_count = VALUES(up_count),
        down_count = VALUES(down_count),
        turnover_rate = VALUES(turnover_rate)
        """

        values = [
            (row['trade_date'], row['volume'], row['up_count'], row['down_count'], row['turnover_rate'])
            for _, row in df.iterrows()
        ]

        execute(sql, values)
        return len(values)

    def get_macro(self, indicator: str = None, days: int = 30) -> list:
        """获取宏观经济数据"""
        sql = """
        SELECT * FROM macro_data
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        params = [days]

        if indicator:
            sql += " AND indicator = %s"
            params.append(indicator)

        sql += " ORDER BY trade_date DESC"

        return fetch_all(sql, params)

    def get_money_flow(self, days: int = 30) -> list:
        """获取资金流向数据"""
        sql = """
        SELECT * FROM money_flow
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY trade_date DESC
        """
        return fetch_all(sql, (days,))

    def get_sentiment(self, days: int = 30) -> list:
        """获取市场情绪数据"""
        sql = """
        SELECT * FROM market_sentiment
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY trade_date DESC
        """
        return fetch_all(sql, (days,))

    def get_global_macro(self, symbol: str = None, days: int = 30) -> list:
        """获取全球宏观数据"""
        sql = """
        SELECT * FROM global_macro
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        params = [days]

        if symbol:
            sql += " AND symbol = %s"
            params.append(symbol)

        sql += " ORDER BY trade_date DESC"

        return fetch_all(sql, params)

    def get_market_features(self, days: int = 30) -> dict:
        """获取市场特征（合并所有数据）"""
        return {
            'macro': self.get_macro(days=days),
            'money_flow': self.get_money_flow(days=days),
            'sentiment': self.get_sentiment(days=days),
            'global': self.get_global_macro(days=days)
        }
```

**测试文件**: `services/market/tests/test_data.py`
```python
import pytest
from services.market.data import MarketCrawler, MarketRepo

def test_market_crawler():
    """测试市场爬虫"""
    crawler = MarketCrawler()
    # 测试可以调用
    assert crawler is not None

def test_market_repo():
    """测试市场仓储"""
    repo = MarketRepo()
    features = repo.get_market_features(days=7)
    assert isinstance(features, dict)
    assert 'macro' in features
    assert 'money_flow' in features
    assert 'sentiment' in features
    assert 'global' in features
```

---

### 任务3: 宏观数据 API

**目标文件**: `services/market/routes/market.py`

**实现要求**:
```python
# services/market/routes/__init__.py
from flask import Blueprint
from .market import market_bp

__all__ = ['market_bp']

# services/market/routes/market.py
from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.market.data import MarketRepo

market_bp = Blueprint('market', __name__)
cache = get_cache()
repo = MarketRepo()

@market_bp.route('/macro', methods=['GET'])
def get_macro_data():
    """获取宏观经济数据"""
    indicator = request.args.get('indicator')
    days = int(request.args.get('days', 30))

    # 缓存（1天）
    cache_key = f"market_macro_{indicator}_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_macro(indicator=indicator, days=days)

    # 缓存
    cache.set(cache_key, result, ttl=86400)

    return jsonify({'success': True, 'data': result, 'cached': False})
```

---

### 任务4: 资金流向 API

**目标文件**: `services/market/routes/market.py`

**实现要求**:
```python
@market_bp.route('/money-flow', methods=['GET'])
def get_money_flow():
    """获取资金流向数据"""
    days = int(request.args.get('days', 30))

    # 缓存（1小时）
    cache_key = f"market_money_flow_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_money_flow(days=days)

    # 缓存
    cache.set(cache_key, result, ttl=3600)

    return jsonify({'success': True, 'data': result, 'cached': False})
```

---

### 任务5: 市场情绪 API

**目标文件**: `services/market/routes/market.py`

**实现要求**:
```python
@market_bp.route('/sentiment', methods=['GET'])
def get_sentiment():
    """获取市场情绪数据"""
    days = int(request.args.get('days', 30))

    # 缓存（1小时）
    cache_key = f"market_sentiment_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_sentiment(days=days)

    # 缓存
    cache.set(cache_key, result, ttl=3600)

    return jsonify({'success': True, 'data': result, 'cached': False})
```

---

### 任务6: 全球宏观 API

**目标文件**: `services/market/routes/market.py`

**实现要求**:
```python
@market_bp.route('/global', methods=['GET'])
def get_global_macro():
    """获取全球宏观数据"""
    symbol = request.args.get('symbol')
    days = int(request.args.get('days', 30))

    # 缓存（1小时）
    cache_key = f"market_global_{symbol}_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_global_macro(symbol=symbol, days=days)

    # 缓存
    cache.set(cache_key, result, ttl=3600)

    return jsonify({'success': True, 'data': result, 'cached': False})
```

---

### 任务7: 市场特征合并 API

**目标文件**: `services/market/routes/market.py`

**实现要求**:
```python
@market_bp.route('/features', methods=['GET'])
def get_market_features():
    """获取市场特征（合并所有数据）"""
    days = int(request.args.get('days', 30))

    # 缓存（1小时）
    cache_key = f"market_features_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_market_features(days=days)

    # 缓存
    cache.set(cache_key, result, ttl=3600)

    return jsonify({'success': True, 'data': result, 'cached': False})

@market_bp.route('/sync', methods=['POST'])
def sync_market_data():
    """同步市场数据"""
    from services.market.data import MarketCrawler, MarketRepo

    crawler = MarketCrawler()
    repo = MarketRepo()

    # 同步所有数据
    data = crawler.sync_all()

    saved = 0
    saved += repo.save_macro(data['macro'])
    saved += repo.save_money_flow(data['money_flow'])
    saved += repo.save_sentiment(data['sentiment'])

    return jsonify({
        'success': True,
        'message': 'Market data synced successfully',
        'data': {
            'saved': saved
        }
    })
```

---

## 交付物

### 目录结构

```
services/market/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── data/                     # 数据层
│   ├── __init__.py
│   ├── market_crawler.py
│   └── market_repo.py
├── routes/                   # 路由
│   ├── __init__.py
│   └── market.py
├── tests/                    # 测试
│   ├── __init__.py
│   └── test_data.py
└── Dockerfile               # Docker 配置
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] 宏观数据 API 正常
- [ ] 资金流向 API 正常
- [ ] 市场情绪 API 正常
- [ ] 全球宏观 API 正常
- [ ] 市场特征 API 正常
- [ ] 数据同步 API 正常
- [ ] 零跨服务依赖
- [ ] 单元测试通过

---

## 依赖

- Infra Agent: Redis, MySQL 连接池
- Coordinator Agent: 任务分发
- QA Agent: 测试验收

---

## 立即开始

你现在需要：

1. **开始任务1**: 创建 Flask 应用骨架
   - 创建 `services/market/app.py`
   - 配置路由

2. **开始任务2**: 迁移市场模块
   - 创建 `services/market/data/`
   - 迁移 `market_crawler.py`, `market_repo.py`
   - 编写测试

3. **开始任务3**: 开发宏观数据 API
   - 创建 `services/market/routes/market.py`
   - 实现缓存逻辑

4. **开始任务4**: 开发资金流向 API
   - 在 `market.py` 中添加资金流向接口

5. **开始任务5**: 开发市场情绪 API
   - 在 `market.py` 中添加市场情绪接口

6. **开始任务6**: 开发全球宏观 API
   - 在 `market.py` 中添加全球宏观接口

7. **开始任务7**: 开发市场特征合并 API
   - 在 `market.py` 中添加特征合并接口

**准备就绪了吗？开始开发市场服务！**

---

## 特点说明

Market Service 是最容易独立部署的服务，具有以下特点：

1. **零跨服务依赖**: 不依赖其他任何服务
2. **独立数据**: 拥有独立的数据表（macro_data, money_flow, market_sentiment, global_macro）
3. **简单架构**: 只包含爬虫、仓储和 API 三层
4. **优先部署**: 可以作为第一个拆分部署的服务验证微服务架构