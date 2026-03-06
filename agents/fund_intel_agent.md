# Fund-Intel Agent - 基金智能服务 Agent

## 角色定义

你是 FundProphet 微服务架构的基金智能服务 Agent，负责行业分析、新闻分类、基金-新闻关联和投资建议。

## 核心职责

1. **服务创建**: 创建独立的 Flask 应用
2. **模块迁移**: 迁移业务模块代码
3. **跨服务对接**: 调用 Fund、News、LLM 服务
4. **API 开发**: 开发基金智能相关 API 端点
5. **事件订阅**: 订阅新闻爬取事件

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/fund-intel/app.py`

**实现要求**:
```python
# services/fund-intel/app.py
from flask import Flask, jsonify
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'fund-intel-service'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'fund-intel-service',
        'uptime': '0'
    })

# 导入路由
from services.fund_intel.routes import (
    fund_industry_bp,
    news_classification_bp,
    fund_news_bp,
    investment_advice_bp
)

app.register_blueprint(fund_industry_bp, url_prefix='/api/fund-industry')
app.register_blueprint(news_classification_bp, url_prefix='/api/news-classification')
app.register_blueprint(fund_news_bp, url_prefix='/api/fund-news')
app.register_blueprint(investment_advice_bp, url_prefix='/api/investment-advice')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8005))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移业务模块

**源文件**:
- `modules/fund_industry/`
- `modules/news_classification/`
- `modules/fund_news_association/`
- `modules/investment_advice/`

**目标目录**: `services/fund-intel/modules/`

**实现要求**:
```python
# services/fund_intel/modules/__init__.py
from .fund_industry import FundIndustryAnalyzer
from .news_classification import NewsClassifier
from .fund_news_association import FundNewsMatcher
from .investment_advice import InvestmentAdviceGenerator

__all__ = [
    'FundIndustryAnalyzer',
    'NewsClassifier',
    'FundNewsMatcher',
    'InvestmentAdviceGenerator'
]

# services/fund_intel/modules/fund_industry/__init__.py
from .analyzer import FundIndustryAnalyzer

# services/fund_intel/modules/fund_industry/analyzer.py
class FundIndustryAnalyzer:
    """基金行业分析器"""

    def analyze(self, fund_code: str) -> dict:
        """分析基金行业配置"""
        # 获取基金持仓
        # 分析行业分布
        # 生成报告
        pass
```

---

### 任务3: 调用 Fund Service

**目标文件**: `services/fund-intel/clients/fund_client.py`

**实现要求**:
```python
# services/fund_intel/clients/__init__.py
from .fund_client import FundClient
from .news_client import NewsClient
from .llm_client import LLMClient

__all__ = ['FundClient', 'NewsClient', 'LLMClient']

# services/fund_intel/clients/fund_client.py
import requests
import os
from shared.cache import get_cache

class FundClient:
    """基金服务客户端"""

    def __init__(self):
        self.base_url = os.getenv('FUND_SERVICE_URL', 'http://fund-service:8002')
        self.cache = get_cache()

    def get_fund_info(self, fund_code: str) -> dict:
        """获取基金信息"""
        # 缓存（1小时）
        cache_key = f"fund_info_{fund_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/fund/{fund_code}")
        if resp.status_code == 200:
            data = resp.json().get('data')
            self.cache.set(cache_key, data, ttl=3600)
            return data
        return None

    def get_fund_nav(self, fund_code: str, days: int = 30) -> list:
        """获取基金净值"""
        # 缓存（30分钟）
        cache_key = f"fund_nav_{fund_code}_{days}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/fund/{fund_code}/nav?days={days}")
        if resp.status_code == 200:
            data = resp.json().get('data')
            self.cache.set(cache_key, data, ttl=1800)
            return data
        return []
```

---

### 任务4: 调用 News Service

**目标文件**: `services/fund-intel/clients/news_client.py`

**实现要求**:
```python
# services/fund_intel/clients/news_client.py
import requests
import os
from shared.cache import get_cache

class NewsClient:
    """新闻服务客户端"""

    def __init__(self):
        self.base_url = os.getenv('NEWS_SERVICE_URL', 'http://news-service:8003')
        self.cache = get_cache()

    def get_news(self, days: int = 1, category: str = None, limit: int = 100) -> list:
        """获取新闻"""
        # 缓存（30分钟）
        cache_key = f"news_list_{days}_{category}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {'days': days, 'limit': limit}
        if category:
            params['category'] = category

        resp = requests.get(f"{self.base_url}/api/news/list", params=params)
        if resp.status_code == 200:
            data = resp.json().get('data')
            self.cache.set(cache_key, data, ttl=1800)
            return data
        return []

    def get_news_by_industry(self, industry: str, days: int = 7) -> list:
        """按行业获取新闻"""
        return self.get_news(days=days, category=industry)
```

---

### 任务5: 调用 LLM Service

**目标文件**: `services/fund-intel/clients/llm_client.py`

**实现要求**:
```python
# services/fund_intel/clients/llm_client.py
import requests
import os
import json
import hashlib
from shared.cache import get_cache

class LLMClient:
    """LLM 服务客户端"""

    def __init__(self):
        self.base_url = os.getenv('LLM_SERVICE_URL', 'http://llm-service:8006')
        self.cache = get_cache()

    def _get_cache_key(self, messages: list, provider: str) -> str:
        """生成缓存键"""
        content = json.dumps(messages, sort_keys=True)
        return f"llm:{provider}:{hashlib.md5(content.encode()).hexdigest()}"

    def chat(self, messages: list, provider: str = 'deepseek', use_cache: bool = True) -> str:
        """通用对话"""
        # 缓存（24小时）
        if use_cache:
            cache_key = self._get_cache_key(messages, provider)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        resp = requests.post(f"{self.base_url}/api/llm/chat", json={
            'provider': provider,
            'messages': messages
        })

        if resp.status_code == 200:
            data = resp.json()
            result = data.get('data')

            if use_cache:
                self.cache.set(cache_key, result, ttl=86400)

            return result
        return None

    def analyze_news(self, news_list: list) -> dict:
        """分析新闻"""
        messages = [
            {"role": "system", "content": "你是一个金融新闻分析助手，请对新闻进行深度分析。"},
            {"role": "user", "content": f"请分析以下新闻：{news_list[:5]}"}
        ]
        result = self.chat(messages, provider='deepseek')
        return {'analysis': result}

    def classify_industry(self, text: str) -> str:
        """分类行业"""
        messages = [
            {"role": "system", "content": "你是一个行业分类专家，请将文本分类到以下行业之一：宏观、行业、全球、政策、公司。"},
            {"role": "user", "content": text}
        ]
        result = self.chat(messages, provider='deepseek')
        return result.strip()

    def generate_investment_advice(self, fund_info: dict, news: list, industry: dict) -> str:
        """生成投资建议"""
        messages = [
            {"role": "system", "content": "你是一个资深的投资顾问，请基于基金信息、新闻和行业分析生成投资建议。"},
            {"role": "user", "content": f"""
基金信息：{fund_info}
相关新闻：{news[:3]}
行业分析：{industry}

请生成投资建议。
            """}
        ]
        result = self.chat(messages, provider='deepseek')
        return result
```

---

### 任务6: 基金行业分析 API

**目标文件**: `services/fund_intel/routes/fund_industry.py`

**实现要求**:
```python
# services/fund_intel/routes/__init__.py
from flask import Blueprint
from .fund_industry import fund_industry_bp
from .news_classification import news_classification_bp
from .fund_news import fund_news_bp
from .investment_advice import investment_advice_bp

__all__ = [
    'fund_industry_bp',
    'news_classification_bp',
    'fund_news_bp',
    'investment_advice_bp'
]

# services/fund_intel/routes/fund_industry.py
from flask import Blueprint, jsonify
from services.fund_intel.clients import FundClient, LLMClient
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer

fund_industry_bp = Blueprint('fund_industry', __name__)
fund_client = FundClient()
llm_client = LLMClient()
analyzer = FundIndustryAnalyzer()

@fund_industry_bp.route('/analyze/<fund_code>', methods=['POST'])
def analyze_fund_industry(fund_code):
    """分析基金行业配置"""
    # 1. 获取基金信息
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({'success': False, 'message': 'Fund not found'}), 404

    # 2. 分析行业分布
    industry_result = analyzer.analyze(fund_code)

    # 3. LLM 分析
    llm_messages = [
        {"role": "system", "content": "你是一个基金行业分析专家。"},
        {"role": "user", "content": f"请分析以下基金的行业配置：{industry_result}"}
    ]
    llm_analysis = llm_client.chat(llm_messages, provider='deepseek')

    return jsonify({
        'success': True,
        'data': {
            'fund_code': fund_code,
            'industry_distribution': industry_result,
            'llm_analysis': llm_analysis
        }
    })

@fund_industry_bp.route('/<fund_code>', methods=['GET'])
def get_fund_industry(fund_code):
    """获取基金行业分析结果"""
    # 从缓存或数据库获取已保存的分析结果
    pass

@fund_industry_bp.route('/primary/<fund_code>', methods=['GET'])
def get_primary_industry(fund_code):
    """获取基金主要行业"""
    # 获取前三大行业
    pass
```

---

### 任务7: 新闻分类 API

**目标文件**: `services/fund_intel/routes/news_classification.py`

**实现要求**:
```python
# services/fund_intel/routes/news_classification.py
from flask import Blueprint, request, jsonify
from services.fund_intel.clients import NewsClient, LLMClient
from shared.messaging import get_mq, NEWS_CRAWLED
import threading

news_classification_bp = Blueprint('news_classification', __name__)
news_client = NewsClient()
llm_client = LLMClient()

def _start_news_listener():
    """启动新闻事件监听器"""
    import redis
    from shared.cache import get_cache

    r = redis.from_url('redis://redis:6379', decode_responses=True)

    # 创建消费者组
    try:
        r.xgroup_create(NEWS_CRAWLED, 'classification_group', id='0', mkstream=True)
    except redis.ResponseError:
        pass

    while True:
        try:
            messages = r.xreadgroup(
                'classification_group', 'classification_consumer',
                {NEWS_CRAWLED: '>'}, count=1, block=5000
            )

            if messages:
                for stream, msgs in messages:
                    for msg_id, msg in msgs:
                        # 自动分类今日新闻
                        _classify_today_news()
                        r.xack(NEWS_CRAWLED, 'classification_group', msg_id)
        except Exception as e:
            print(f"监听错误: {e}")
            import time
            time.sleep(5)

# 启动监听器
listener_thread = threading.Thread(target=_start_news_listener, daemon=True)
listener_thread.start()

@news_classification_bp.route('/classify', methods=['POST'])
def classify_news():
    """分类单条新闻"""
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'success': False, 'message': 'Text required'}), 400

    industry = llm_client.classify_industry(text)

    return jsonify({
        'success': True,
        'data': {
            'text': text,
            'industry': industry
        }
    })

@news_classification_bp.route('/classify-today', methods=['POST'])
def classify_today_news():
    """分类今日所有新闻"""
    # 1. 获取今日新闻
    news_list = news_client.get_news(days=1)

    # 2. 批量分类
    classified = []
    for news in news_list:
        industry = llm_client.classify_industry(news['content'])
        classified.append({
            'id': news['id'],
            'industry': industry
        })

    # 3. 保存分类结果
    # ... 保存到数据库

    return jsonify({
        'success': True,
        'data': {
            'total': len(news_list),
            'classified': len(classified),
            'results': classified
        }
    })

@news_classification_bp.route('/industries', methods=['GET'])
def get_industries():
    """获取所有行业分类"""
    # 返回行业列表
    return jsonify({
        'success': True,
        'data': ['宏观', '行业', '全球', '政策', '公司']
    })

@news_classification_bp.route('/industry/<industry>', methods=['GET'])
def get_news_by_industry(industry):
    """按行业获取新闻"""
    days = int(request.args.get('days', 7))
    news_list = news_client.get_news_by_industry(industry, days=days)

    return jsonify({
        'success': True,
        'data': news_list
    })

@news_classification_bp.route('/stats', methods=['GET'])
def get_industry_stats():
    """获取行业统计"""
    # 返回各行业的新闻数量统计
    pass

@news_classification_bp.route('/today', methods=['GET'])
def get_today_classified():
    """获取今日已分类新闻"""
    # 返回今日已分类的新闻列表
    pass
```

---

### 任务8: 基金-新闻关联 API

**目标文件**: `services/fund_intel/routes/fund_news.py`

**实现要求**:
```python
# services/fund_intel/routes/fund_news.py
from flask import Blueprint, jsonify
from services.fund_intel.clients import FundClient, NewsClient, LLMClient
from services.fund_intel.modules.fund_news_association import FundNewsMatcher

fund_news_bp = Blueprint('fund_news', __name__)
fund_client = FundClient()
news_client = NewsClient()
llm_client = LLMClient()
matcher = FundNewsMatcher()

@fund_news_bp.route('/match/<fund_code>', methods=['GET'])
def match_fund_news(fund_code):
    """为基金匹配相关新闻"""
    # 1. 获取基金信息
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({'success': False, 'message': 'Fund not found'}), 404

    # 2. 获取基金行业
    industries = matcher.get_fund_industries(fund_code)

    # 3. 获取相关新闻
    related_news = []
    for industry in industries:
        news = news_client.get_news_by_industry(industry, days=7)
        related_news.extend(news)

    # 4. 去重并排序
    unique_news = matcher.deduplicate_and_rank(related_news, fund_info)

    return jsonify({
        'success': True,
        'data': {
            'fund_code': fund_code,
            'industries': industries,
            'news_count': len(unique_news),
            'news': unique_news[:20]  # 返回前20条
        }
    })

@fund_news_bp.route('/summary/<fund_code>', methods=['GET'])
def get_fund_news_summary(fund_code):
    """获取基金新闻摘要"""
    # 1. 匹配相关新闻
    # 2. LLM 生成摘要
    pass

@fund_news_bp.route('/list', methods=['GET'])
def get_funds_with_news():
    """获取有关联新闻的基金列表"""
    # 返回已匹配新闻的基金列表
    pass
```

---

### 任务9: 投资建议 API

**目标文件**: `services/fund_intel/routes/investment_advice.py`

**实现要求**:
```python
# services/fund_intel/routes/investment_advice.py
from flask import Blueprint, request, jsonify
from services.fund_intel.clients import FundClient, NewsClient, LLMClient
from services.fund_intel.modules.investment_advice import InvestmentAdviceGenerator

investment_advice_bp = Blueprint('investment_advice', __name__)
fund_client = FundClient()
news_client = NewsClient()
llm_client = LLMClient()
advice_generator = InvestmentAdviceGenerator()

@investment_advice_bp.route('/<fund_code>', methods=['GET'])
def get_investment_advice(fund_code):
    """获取投资建议"""
    # 1. 获取基金信息
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({'success': False, 'message': 'Fund not found'}), 404

    # 2. 获取基金净值历史
    fund_nav = fund_client.get_fund_nav(fund_code, days=30)

    # 3. 获取相关新闻
    news_list = news_client.get_news(days=7)

    # 4. 获取基金行业分析
    # industry_analysis = ...

    # 5. 生成投资建议
    advice = llm_client.generate_investment_advice(
        fund_info=fund_info,
        news=news_list,
        industry={}
    )

    return jsonify({
        'success': True,
        'data': {
            'fund_code': fund_code,
            'fund_name': fund_info.get('fund_name'),
            'advice': advice,
            'generated_at': datetime.now().isoformat()
        }
    })

@investment_advice_bp.route('/batch', methods=['POST'])
def get_batch_investment_advice():
    """批量获取投资建议"""
    data = request.get_json()
    fund_codes = data.get('fund_codes', [])

    if not fund_codes:
        return jsonify({'success': False, 'message': 'Fund codes required'}), 400

    results = []
    for fund_code in fund_codes:
        try:
            # 获取投资建议
            advice = get_investment_advice_logic(fund_code)
            results.append({
                'fund_code': fund_code,
                'success': True,
                'advice': advice
            })
        except Exception as e:
            results.append({
                'fund_code': fund_code,
                'success': False,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'data': results
    })
```

---

## 交付物

### 目录结构

```
services/fund-intel/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── clients/                  # 服务客户端
│   ├── __init__.py
│   ├── fund_client.py
│   ├── news_client.py
│   └── llm_client.py
├── modules/                  # 业务模块
│   ├── fund_industry/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   ├── news_classification/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── fund_news_association/
│   │   ├── __init__.py
│   │   └── matcher.py
│   └── investment_advice/
│       ├── __init__.py
│       └── generator.py
├── routes/                   # 路由
│   ├── __init__.py
│   ├── fund_industry.py
│   ├── news_classification.py
│   ├── fund_news.py
│   └── investment_advice.py
├── tests/                    # 测试
│   ├── __init__.py
│   ├── test_clients.py
│   └── test_routes.py
└── Dockerfile               # Docker 配置
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] 基金行业分析 API 正常
- [ ] 新闻分类 API 正常
- [ ] 基金-新闻关联 API 正常
- [ ] 投资建议 API 正常
- [ ] 跨服务调用正常
- [ ] 事件订阅正常
- [ ] 单元测试通过

---

## 依赖

- Infra Agent: Redis, MySQL 连接池
- Fund Svc Agent: 基金数据
- News Svc Agent: 新闻数据
- LLM Svc Agent: LLM 分析
- Coordinator Agent: 任务分发
- QA Agent: 测试验收

---

## 立即开始

你现在需要：

1. **开始任务1**: 创建 Flask 应用骨架
   - 创建 `services/fund-intel/app.py`
   - 配置路由

2. **开始任务2**: 迁移业务模块
   - 创建 `services/fund-intel/modules/`
   - 迁移所有业务模块

3. **开始任务3**: 开发 Fund 客户端
   - 创建 `services/fund-intel/clients/fund_client.py`
   - 实现缓存逻辑

4. **开始任务4**: 开发 News 客户端
   - 创建 `services/fund-intel/clients/news_client.py`

5. **开始任务5**: 开发 LLM 客户端
   - 创建 `services/fund-intel/clients/llm_client.py`

6. **开始任务6**: 开发基金行业分析 API
   - 创建 `services/fund-intel/routes/fund_industry.py`

7. **开始任务7**: 开发新闻分类 API
   - 创建 `services/fund-intel/routes/news_classification.py`
   - 实现事件监听

8. **开始任务8**: 开发基金-新闻关联 API
   - 创建 `services/fund-intel/routes/fund_news.py`

9. **开始任务9**: 开发投资建议 API
   - 创建 `services/fund-intel/routes/investment_advice.py`

**准备就绪了吗？开始开发基金智能服务！**

---

## 特点说明

Fund-Intel Service 是最复杂的跨服务依赖服务：

1. **跨服务依赖**: 依赖 Fund、News、LLM 三个服务
2. **业务逻辑复杂**: 包含4个业务模块
3. **事件驱动**: 需要订阅新闻爬取事件
4. **优先级高**: 是项目的核心智能分析功能
5. **测试复杂**: 需要完整的集成测试