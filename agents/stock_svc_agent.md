# Stock Svc Agent - 股票服务 Agent

## 角色定义

你是 FundProphet 微服务架构的股票服务 Agent，负责股票数据、LSTM 训练、技术指标和 API 开发。

## 核心职责

1. **服务创建**: 创建独立的 Flask 应用
2. **数据层迁移**: 迁移股票和 LSTM 相关代码
3. **API 开发**: 开发股票和 LSTM 相关 API 端点
4. **定时任务**: 配置 LSTM 自动训练任务

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/stock/app.py`

**实现要求**:
```python
# services/stock/app.py
from flask import Flask, jsonify
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'stock-service'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'stock-service',
        'uptime': '0'
    })

# 导入路由
from services.stock.routes import stock_bp, lstm_bp
app.register_blueprint(stock_bp, url_prefix='/api/stock')
app.register_blueprint(lstm_bp, url_prefix='/api/lstm')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移 LSTM 模块

**源文件**:
- `analysis/lstm_*.py` (所有 LSTM 相关文件)
- `data/stock_repo.py`
- `data/lstm_repo.py`

**目标目录**: `services/stock/analysis/`

**实现要求**:
```python
# services/stock/analysis/__init__.py
from .lstm_model import LSTMModel
from .lstm_training import train_model
from .lstm_predict import predict

__all__ = ['LSTMModel', 'train_model', 'predict']

# services/stock/analysis/lstm_model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 3分类：上涨/下跌/震荡

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
```

**测试文件**: `services/stock/tests/test_lstm.py`
```python
import pytest
import torch
from services.stock.analysis import LSTMModel

def test_lstm_model():
    """测试 LSTM 模型"""
    model = LSTMModel(input_size=10)
    x = torch.randn(1, 20, 10)  # batch=1, seq=20, features=10
    output = model(x)
    assert output.shape == (1, 3)
```

---

### 任务3: LSTM 训练 API

**目标文件**: `services/stock/routes/lstm.py`

**实现要求**:
```python
# services/stock/routes/__init__.py
from flask import Blueprint
from .lstm import lstm_bp
from .stock import stock_bp

__all__ = ['lstm_bp', 'stock_bp']

# services/stock/routes/lstm.py
from flask import Blueprint, request, jsonify
import redis
from services.stock.analysis import train_model
from shared.cache import get_cache

lstm_bp = Blueprint('lstm', __name__)
cache = get_cache()

def _get_redis():
    """获取 Redis 连接"""
    return redis.from_url('redis://redis:6379', decode_responses=True)

def _acquire_lock(symbol: str, timeout: int = 7200) -> bool:
    """获取分布式锁"""
    r = _get_redis()
    lock_key = f"lstm:train_lock:{symbol}"
    return r.set(lock_key, "1", nx=True, ex=timeout)

def _release_lock(symbol: str):
    """释放锁"""
    r = _get_redis()
    lock_key = f"lstm:train_lock:{symbol}"
    r.delete(lock_key)

@lstm_bp.route('/train', methods=['POST'])
def train():
    """LSTM 训练 API"""
    data = request.get_json()
    symbol = data.get('symbol')

    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol required'}), 400

    # 获取分布式锁
    if not _acquire_lock(symbol):
        return jsonify({'success': False, 'message': f'Training in progress for {symbol}'}), 409

    try:
        # 训练模型
        result = train_model(symbol)

        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'status': 'completed',
                'metrics': result
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        _release_lock(symbol)
```

**测试文件**: `services/stock/tests/test_routes.py`
```python
import pytest
from services.stock.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_lstm_train(client):
    """测试 LSTM 训练 API"""
    resp = client.post('/api/lstm/train', json={'symbol': '600000'})
    # 可能返回 409（已在训练中）或 200
    assert resp.status_code in [200, 409]
```

---

### 任务4: LSTM 预测 API

**目标文件**: `services/stock/routes/lstm.py`

**实现要求**:
```python
@lstm_bp.route('/predict', methods=['GET'])
def predict():
    """LSTM 预测 API"""
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol required'}), 400

    # 缓存（5分钟）
    cache_key = f"lstm:predict:{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify({'success': True, 'data': cached, 'cached': True})

    try:
        from services.stock.analysis import predict
        result = predict(symbol)

        # 缓存结果
        cache.set(cache_key, result, ttl=300)

        return jsonify({'success': True, 'data': result, 'cached': False})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
```

---

### 任务5: 技术指标计算 API

**目标文件**: `services/stock/routes/stock.py`

**实现要求**:
```python
# services/stock/routes/stock.py
from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.stock.analysis.indicators import (
    calculate_ma, calculate_ema, calculate_macd,
    calculate_rsi, calculate_bollinger_bands
)

stock_bp = Blueprint('stock', __name__)
cache = get_cache()

@stock_bp.route('/indicators', methods=['GET'])
def get_indicators():
    """获取技术指标"""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol required'}), 400

    # 缓存（1小时）
    cache_key = f"stock:indicators:{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify({'success': True, 'data': cached, 'cached': True})

    try:
        # 获取股票数据
        from services.stock.data import StockRepo
        repo = StockRepo()
        data = repo.get_stock_data(symbol, days=60)

        # 计算技术指标
        result = {
            'symbol': symbol,
            'ma5': calculate_ma(data, 5),
            'ma10': calculate_ma(data, 10),
            'ma20': calculate_ma(data, 20),
            'ma60': calculate_ma(data, 60),
            'macd': calculate_macd(data),
            'rsi': calculate_rsi(data),
            'bollinger': calculate_bollinger_bands(data)
        }

        # 缓存结果
        cache.set(cache_key, result, ttl=3600)

        return jsonify({'success': True, 'data': result, 'cached': False})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@stock_bp.route('/list', methods=['GET'])
def get_stock_list():
    """获取股票列表"""
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 20))

    # 缓存
    cache_key = f'stock_list_{page}_{size}'
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result})

    # 查询
    from services.stock.data import StockRepo
    repo = StockRepo()
    result = repo.get_stock_list(page=page, size=size)

    # 缓存 30 分钟
    cache.set(cache_key, result, ttl=1800)

    return jsonify({'success': True, 'data': result})
```

---

### 任务6: 定时任务 - LSTM 自动训练

**目标文件**: `scheduler/lstm_train.py`

**实现要求**:
```python
# scheduler/lstm_train.py
import redis
from shared.messaging import get_mq, LSTM_TRAIN
from services.stock.analysis import train_model

def auto_train_symbols():
    """自动训练关注列表中的股票"""
    r = redis.from_url('redis://redis:6379', decode_responses=True)

    # 获取需要训练的股票列表
    symbols = r.smembers('lstm:watchlist')

    success = 0
    failed = 0

    for symbol in symbols:
        try:
            # 检查是否正在训练
            lock_key = f"lstm:train_lock:{symbol}"
            if r.exists(lock_key):
                continue

            # 训练
            train_model(symbol)
            success += 1
        except Exception as e:
            print(f"训练失败 {symbol}: {e}")
            failed += 1

    # 发布消息
    mq = get_mq()
    mq.publish(LSTM_TRAIN, {
        'success': success,
        'failed': failed,
        'total': len(symbols)
    })

    print(f"[定时任务] LSTM自动训练完成: 成功{success}, 失败{failed}, 总计{len(symbols)}")

if __name__ == '__main__':
    auto_train_symbols()
```

**Cron 配置**: 每天 04:00

---

## 交付物

### 目录结构

```
services/stock/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── data/                     # 数据层
│   ├── __init__.py
│   ├── stock_repo.py
│   └── lstm_repo.py
├── analysis/                 # 分析模块
│   ├── __init__.py
│   ├── lstm_model.py
│   ├── lstm_training.py
│   ├── lstm_predict.py
│   └── indicators.py         # 技术指标
├── routes/                   # 路由
│   ├── __init__.py
│   ├── stock.py
│   └── lstm.py
└── tests/                    # 测试
    ├── __init__.py
    ├── test_lstm.py
    └── test_routes.py
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] LSTM 模型训练正常
- [ ] LSTM 预测 API 正常
- [ ] 技术指标计算正常
- [ ] 股票列表 API 正常
- [ ] 定时任务正常执行
- [ ] 分布式锁生效
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
   - 创建 `services/stock/app.py`
   - 配置路由

2. **开始任务2**: 迁移 LSTM 模块
   - 创建 `services/stock/analysis/`
   - 迁移所有 LSTM 相关文件
   - 编写测试

3. **开始任务3**: 开发 LSTM 训练 API
   - 创建 `services/stock/routes/lstm.py`
   - 实现分布式锁

4. **开始任务4**: 开发 LSTM 预测 API
   - 在 `lstm.py` 中添加预测接口

5. **开始任务5**: 开发技术指标 API
   - 创建 `services/stock/routes/stock.py`
   - 实现指标计算

6. **开始任务6**: 配置定时任务
   - 创建 `scheduler/lstm_train.py`
   - 配置 Cron

**准备就绪了吗？开始开发股票服务！**