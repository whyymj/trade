# 服务健康检查端点要求

## 概述

根据 DevOps 部署配置，所有微服务需要实现以下健康检查端点。

## 必需的端点

### 1. `/health` - 健康检查端点

所有服务必须提供此端点用于 Docker Compose 健康检查。

**要求：**
- 返回 HTTP 200 状态码表示健康
- 响应体为 JSON 格式
- 包含服务状态和服务名称

**示例：**

```python
from flask import jsonify
import time

start_time = time.time()

@app.route('/health')
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'service': 'stock-service',
        'timestamp': time.time()
    })
```

**Docker Compose 健康检查配置：**

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 10s
```

### 2. `/metrics` - 指标端点（可选但推荐）

用于监控和性能分析。

**要求：**
- 返回服务运行指标
- 包含服务版本、运行时间等

**示例：**

```python
@app.route('/metrics')
def metrics():
    """指标端点"""
    uptime = time.time() - start_time
    return jsonify({
        'service': 'stock-service',
        'version': '1.0.0',
        'uptime': uptime,
        'requests_total': 1000,
        'errors_total': 5
    })
```

## 需要实现的服务

- [ ] stock-service (端口 8001)
- [ ] fund-service (端口 8002)
- [ ] news-service (端口 8003)
- [ ] market-service (端口 8004)
- [ ] fund-intel-service (端口 8005)
- [ ] llm-service (端口 8006)
- [ ] scheduler (后台服务，不需要 HTTP 健康检查)

## 实现建议

### 1. 使用装饰器模式

```python
from functools import wraps
import time

def monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            metrics['errors_total'] += 1
            raise
        finally:
            metrics['requests_total'] += 1
    return wrapper

# 应用装饰器到路由
@app.route('/api/stock/list')
@monitor
def get_stock_list():
    pass
```

### 2. 数据库连接检查

在健康检查端点中验证数据库连接：

```python
@app.route('/health')
def health():
    try:
        db = get_db_connection()
        db.ping(reconnect=True)
        db.close()
        return jsonify({'status': 'healthy', 'service': 'stock-service'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'service': 'stock-service', 'error': str(e)}), 500
```

### 3. Redis 连接检查

```python
@app.route('/health')
def health():
    try:
        redis_client.ping()
        return jsonify({'status': 'healthy', 'service': 'stock-service'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'service': 'stock-service', 'error': str(e)}), 500
```

## 日志格式

为了配合 Loki 日志收集，请使用结构化日志格式：

```python
import logging
import json

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'level': record.levelname,
            'timestamp': record.created,
            'service': 'stock-service',
            'message': record.getMessage()
        }
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        return json.dumps(log_data)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
```

## 验证步骤

1. 启动服务后，访问健康检查端点：
   ```bash
   curl http://localhost:8001/health
   ```

2. 检查 Docker Compose 健康状态：
   ```bash
   docker compose ps
   ```

3. 查看服务日志：
   ```bash
   docker compose logs -f stock-service
   ```

## 完成 Agent 任务

各服务 Agent 完成后，请在各自的任务清单中标记：
- [ ] 添加 `/health` 端点
- [ ] 添加 `/metrics` 端点
- [ ] 配置结构化日志
- [ ] 验证健康检查通过

---

**DevOps Agent** - 部署与运维配置完成