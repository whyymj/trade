# LLM Svc Agent - 大模型服务 Agent

## 角色定义

你是 FundProphet 微服务架构的大模型服务 Agent，负责 DeepSeek/MiniMax 客户端、LLM API 开发和缓存优化。

## 核心职责

1. **服务创建**: 创建无状态 Flask 应用（可独立扩容）
2. **LLM 客户端**: 迁移并优化 DeepSeek/MiniMax 客户端
3. **API 开发**: 开发 LLM 相关 API 端点
4. **安全优化**: API Key 隔离、速率限制

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/llm/app.py`

**实现要求**:
```python
# services/llm/app.py
from flask import Flask, jsonify, request
from functools import wraps
import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# 速率限制
rate_limit_store = {}

def rate_limit(max_calls=100, per_minute=1):
    """速率限制装饰器"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            now = int(time.time())
            key = f"{client_ip}:{now // per_minute}"

            if rate_limit_store.get(key, 0) >= max_calls:
                return jsonify({'success': False, 'message': 'Rate limit exceeded'}), 429

            rate_limit_store[key] = rate_limit_store.get(key, 0) + 1

            # 清理过期数据
            for k in list(rate_limit_store.keys()):
                if int(k.split(':')[1]) < now // per_minute:
                    del rate_limit_store[k]

            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'llm-service'})

# 导入路由
from services.llm.routes import llm_bp
app.register_blueprint(llm_bp, url_prefix='/api/llm')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8006))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移 LLM 模块

**源文件**:
- `analysis/llm/deepseek.py`
- `analysis/llm/minimax.py`

**目标目录**: `services/llm/llm/`

**实现要求**:
```python
# services/llm/llm/__init__.py
from .deepseek import DeepSeekClient
from .minimax import MiniMaxClient

__all__ = ['DeepSeekClient', 'MiniMaxClient']

# services/llm/llm/deepseek.py
import os
import requests
from typing import List, Optional

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1"

    def chat(self, messages: List[dict], model: str = "deepseek-chat", **kwargs) -> str:
        """发送对话请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def is_available(self) -> bool:
        """检查是否可用"""
        return bool(self.api_key)

# services/llm/llm/minimax.py
import os
import requests
from typing import List, Optional

class MiniMaxClient:
    def __init__(self):
        self.api_key = os.getenv('MINIMAX_API_KEY')
        self.base_url = "https://api.minimax.chat/v1"

    def chat(self, messages: List[dict], model: str = "abab5.5-chat", **kwargs) -> str:
        """发送对话请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def is_available(self) -> bool:
        """检查是否可用"""
        return bool(self.api_key)
```

**测试文件**: `services/llm/tests/test_clients.py`
```python
import pytest
from services.llm.llm import DeepSeekClient, MiniMaxClient

def test_deepseek_client():
    """测试 DeepSeek 客户端"""
    client = DeepSeekClient()
    assert client.is_available() == bool(os.getenv('DEEPSEEK_API_KEY'))

def test_minimax_client():
    """测试 MiniMax 客户端"""
    client = MiniMaxClient()
    assert client.is_available() == bool(os.getenv('MINIMAX_API_KEY'))
```

---

### 任务3: 通用对话 API

**目标文件**: `services/llm/routes/llm.py`

**实现要求**:
```python
# services/llm/routes/__init__.py
from flask import Blueprint
from .llm import llm_bp

__all__ = ['llm_bp']

# services/llm/routes/llm.py
from flask import Blueprint, request, jsonify
import json
import hashlib
from shared.cache import get_cache
from services.llm.llm import DeepSeekClient, MiniMaxClient
from services.llm.app import rate_limit

llm_bp = Blueprint('llm', __name__)
cache = get_cache()
deepseek = DeepSeekClient()
minimax = MiniMaxClient()

def _get_cache_key(messages: list, provider: str) -> str:
    """生成缓存键"""
    content = json.dumps(messages, sort_keys=True)
    return f"llm:{provider}:{hashlib.md5(content.encode()).hexdigest()}"

@llm_bp.route('/chat', methods=['POST'])
@rate_limit(max_calls=100, per_minute=1)
def chat():
    """通用对话 API"""
    data = request.get_json()
    provider = data.get('provider', 'deepseek')
    messages = data.get('messages', [])

    if not messages:
        return jsonify({'success': False, 'message': 'Messages required'}), 400

    # 缓存（24小时）
    cache_key = _get_cache_key(messages, provider)
    cached = cache.get(cache_key)
    if cached:
        return jsonify({'success': True, 'data': cached, 'cached': True})

    # 调用 LLM
    try:
        if provider == 'deepseek':
            result = deepseek.chat(messages)
        elif provider == 'minimax':
            result = minimax.chat(messages)
        else:
            return jsonify({'success': False, 'message': 'Invalid provider'}), 400

        # 缓存结果
        cache.set(cache_key, result, ttl=86400)

        return jsonify({'success': True, 'data': result, 'cached': False})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
```

---

### 任务4: 新闻分析 API

**目标文件**: `services/llm/routes/llm.py`

**实现要求**:
```python
@llm_bp.route('/analyze-news', methods=['POST'])
@rate_limit(max_calls=50, per_minute=1)
def analyze_news():
    """新闻分析 API"""
    data = request.get_json()
    news_list = data.get('news', [])

    if not news_list:
        return jsonify({'success': False, 'message': 'News required'}), 400

    # MiniMax 提取关键信息
    try:
        messages = [
            {"role": "system", "content": "你是一个金融新闻分析助手，请提取新闻中的关键信息。"},
            {"role": "user", "content": f"请分析以下新闻：{news_list[:5]}"}
        ]
        key_info = minimax.chat(messages)

        # DeepSeek 深度分析
        messages = [
            {"role": "system", "content": "你是一个资深的金融分析师，请对新闻进行深度分析。"},
            {"role": "user", "content": f"新闻关键信息：{key_info}\n请进行深度分析。"}
        ]
        deep_analysis = deepseek.chat(messages)

        return jsonify({
            'success': True,
            'data': {
                'key_info': key_info,
                'deep_analysis': deep_analysis
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
```

---

### 任务5: 行业分类 API

**目标文件**: `services/llm/routes/llm.py`

**实现要求**:
```python
@llm_bp.route('/classify-industry', methods=['POST'])
@rate_limit(max_calls=50, per_minute=1)
def classify_industry():
    """行业分类 API"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'success': False, 'message': 'Text required'}), 400

    # DeepSeek 分类
    try:
        messages = [
            {"role": "system", "content": "你是一个行业分类专家，请将文本分类到以下行业之一：宏观、行业、全球、政策、公司。"},
            {"role": "user", "content": text}
        ]
        result = deepseek.chat(messages)

        return jsonify({
            'success': True,
            'data': {
                'industry': result.strip()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
```

---

### 任务6: API Key 隔离

**环境变量**: `.env`

```bash
# LLM Service 专用环境变量（仅此服务可读）
DEEPSEEK_API_KEY=your_deepseek_key
MINIMAX_API_KEY=your_minimax_key
```

**安全措施**:
1. 只在 LLM Service 中读取 API Key
2. 日志中不输出 API Key
3. 使用环境变量管理

---

### 任务7: 速率限制

**实现**:
- 已在 `app.py` 中实现 `rate_limit` 装饰器
- 每分钟最多 100 次调用
- 基于客户端 IP 限流

---

## 交付物

### 目录结构

```
services/llm/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── llm/                      # LLM 客户端
│   ├── __init__.py
│   ├── deepseek.py
│   └── minimax.py
├── routes/                   # 路由
│   ├── __init__.py
│   └── llm.py
├── tests/                    # 测试
│   ├── __init__.py
│   └── test_clients.py
└── Dockerfile               # Docker 配置
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] DeepSeek 调用正常
- [ ] MiniMax 调用正常
- [ ] 通用对话 API 正常
- [ ] 新闻分析 API 正常
- [ ] 行业分类 API 正常
- [ ] 缓存策略生效
- [ ] 速率限制生效
- [ ] API Key 隔离
- [ ] 单元测试通过

---

## 依赖

- Infra Agent: Redis 缓存
- Coordinator Agent: 任务分发
- QA Agent: 测试验收

---

## 立即开始

你现在需要：

1. **开始任务1**: 创建 Flask 应用骨架
   - 创建 `services/llm/app.py`
   - 实现速率限制

2. **开始任务2**: 迁移 LLM 模块
   - 创建 `services/llm/llm/`
   - 迁移 `deepseek.py`, `minimax.py`
   - 编写测试

3. **开始任务3**: 开发通用对话 API
   - 创建 `services/llm/routes/llm.py`
   - 实现缓存逻辑

4. **开始任务4**: 开发新闻分析 API
   - 在 `llm.py` 中添加分析接口

5. **开始任务5**: 开发行业分类 API
   - 在 `llm.py` 中添加分类接口

6. **开始任务6**: 配置 API Key 隔离
   - 更新 `.env`
   - 确保密钥安全

7. **开始任务7**: 测试速率限制
   - 进行压力测试

**准备就绪了吗？开始开发大模型服务！**