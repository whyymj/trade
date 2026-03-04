# 新闻分析系统 - 完整架构设计文档

## 一、模块独立性设计

### 1.1 设计原则

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           模块设计原则                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  1. 单一职责：每个模块只负责一项功能                                     │
│  2. 接口隔离：模块间通过接口通信，不直接依赖具体实现                     │
│  3. 依赖倒置：上层模块依赖抽象接口，不依赖下层具体类                     │
│  4. 可替换性：同一接口可替换不同实现                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖关系

```
                    ┌─────────────────────────────────────────┐
                    │           API 层 (Routes)               │
                    │    /api/news/*  /api/market/*           │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           Service 层                      │
                    │  NewsService   MarketService   LLMService │
                    └─────────────────┬───────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼───────┐          ┌─────────▼─────────┐        ┌────────▼────────┐
│  Crawler 层   │          │    Repo 层       │        │    LLM 层      │
│  新闻爬虫     │          │    数据仓储      │        │   大模型       │
│  市场爬虫     │          │    市场数据      │        │ MiniMax/DeepSeek│
└───────────────┘          └──────────────────┘        └────────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           Data Layer                    │
                    │        MySQL / Cache                    │
                    └─────────────────────────────────────────┘
```

### 1.3 目录结构

```
trade/
├── data/                           # 数据层
│   ├── __init__.py
│   ├── mysql.py                    # MySQL 连接
│   ├── schema.py                   # 表结构
│   ├── cache.py                   # 内存缓存
│   ├── fund_repo.py               # 基金仓储
│   ├── fund_fetcher.py            # 基金抓取
│   ├── index_repo.py              # 指数仓储
│   │
│   ├── market/                    # ★ 市场数据模块
│   │   ├── __init__.py
│   │   ├── interfaces.py          # 接口定义
│   │   ├── crawler.py            # 市场爬虫
│   │   ├── repo.py                # 市场仓储
│   │   └── tests/                 # ★ 测试模块
│   │       ├── __init__.py
│   │       ├── test_crawler.py
│   │       └── test_repo.py
│   │
│   └── news/                     # ★ 新闻数据模块
│       ├── __init__.py
│       ├── interfaces.py         # 接口定义
│       ├── crawler.py            # 新闻爬虫
│       ├── repo.py               # 新闻仓储
│       └── tests/                # ★ 测试模块
│           ├── __init__.py
│           ├── test_crawler.py
│           └── test_repo.py
│
├── analysis/                      # 分析层
│   ├── __init__.py
│   ├── fund_metrics.py
│   ├── fund_lstm.py
│   │
│   └── llm/                     # ★ LLM 模块
│       ├── __init__.py
│       ├── interfaces.py         # 接口定义
│       ├── minimax.py             # MiniMax 客户端
│       ├── deepseek.py            # DeepSeek 客户端
│       ├── news_analyzer.py       # 新闻分析器
│       └── tests/                # ★ 测试模块
│           ├── __init__.py
│           ├── test_minimax.py
│           ├── test_deepseek.py
│           └── test_analyzer.py
│
├── server/                       # Web 层
│   ├── app.py
│   ├── routes/
│   │   ├── api.py
│   │   ├── market.py             # ★ 市场 API
│   │   └── news.py               # ★ 新闻 API
│   └── tests/                   # ★ 测试模块
│       ├── __init__.py
│       ├── test_routes.py
│       └── test_integration.py
│
├── tests/                        # 集成测试
│   ├── __init__.py
│   ├── conftest.py               # pytest 配置
│   ├── fixtures/                 # 测试数据
│   │   └── sample_news.json
│   ├── test_fund.py              # 基金测试
│   ├── test_news_flow.py         # ★ 新闻流程测试
│   └── test_market_flow.py       # ★ 市场流程测试
│
└── docs/
    ├── ARCHITECTURE.md
    ├── MARKET_CRAWLER_ARCHITECTURE.md
    └── NEWS_ANALYSIS_ARCHITECTURE.md
```

---

## 二、接口定义

### 2.1 市场数据接口

```python
# data/market/interfaces.py

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class MarketDataPort(ABC):
    """市场数据端口 - 抽象接口"""
    
    @abstractmethod
    def get_macro_data(self, indicator: str, days: int = 30) -> Optional[pd.DataFrame]:
        """获取宏观经济数据"""
        pass
    
    @abstractmethod
    def get_money_flow(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取资金流向数据"""
        pass
    
    @abstractmethod
    def get_sentiment(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取市场情绪数据"""
        pass
    
    @abstractmethod
    def get_global_macro(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """获取全球宏观数据"""
        pass
    
    @abstractmethod
    def save_macro_data(self, df: pd.DataFrame) -> int:
        """保存宏观经济数据"""
        pass
    
    @abstractmethod
    def save_money_flow(self, df: pd.DataFrame) -> int:
        """保存资金流向数据"""
        pass


class MarketCrawlerPort(ABC):
    """市场爬虫端口 - 抽象接口"""
    
    @abstractmethod
    def fetch_macro(self) -> pd.DataFrame:
        """抓取宏观经济数据"""
        pass
    
    @abstractmethod
    def fetch_money_flow(self) -> pd.DataFrame:
        """抓取资金流向数据"""
        pass
    
    @abstractmethod
    def fetch_sentiment(self) -> pd.DataFrame:
        """抓取市场情绪数据"""
        pass
```

### 2.2 新闻数据接口

```python
# data/news/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class NewsItem:
    """新闻条目"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str = "general"


@dataclass
class AnalysisResult:
    """分析结果"""
    news_count: int
    summary: str
    deep_analysis: str
    market_impact: str
    key_events: List[dict]
    investment_advice: str
    analyzed_at: datetime


class NewsCrawlerPort(ABC):
    """新闻爬虫端口 - 抽象接口"""
    
    @abstractmethod
    def fetch_cailian(self) -> List[NewsItem]:
        """抓取财联社"""
        pass
    
    @abstractmethod
    def fetch_wallstreet(self) -> List[NewsItem]:
        """抓取华尔街见闻"""
        pass
    
    @abstractmethod
    def fetch_all(self) -> List[NewsItem]:
        """聚合全部新闻"""
        pass


class NewsRepoPort(ABC):
    """新闻仓储端口 - 抽象接口"""
    
    @abstractmethod
    def save_news(self, news_list: List[NewsItem]) -> int:
        """保存新闻"""
        pass
    
    @abstractmethod
    def get_news(self, days: int = 1, category: str = None) -> List[NewsItem]:
        """获取新闻"""
        pass
    
    @abstractmethod
    def save_analysis(self, result: AnalysisResult) -> bool:
        """保存分析结果"""
        pass
    
    @abstractmethod
    def get_latest_analysis(self) -> Optional[AnalysisResult]:
        """获取最新分析结果"""
        pass
```

### 2.3 LLM 接口

```python
# analysis/llm/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional


class LLMClientPort(ABC):
    """LLM 客户端端口 - 抽象接口"""
    
    @abstractmethod
    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送对话请求"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class NewsAnalyzerPort(ABC):
    """新闻分析器端口 - 抽象接口"""
    
    @abstractmethod
    def extract_key_info(self, news_list: List[NewsItem]) -> str:
        """提取关键信息 (MiniMax)"""
        pass
    
    @abstractmethod
    def analyze(self, news_list: List[NewsItem], use_deepseek: bool = False) -> AnalysisResult:
        """
        综合分析
        
        Args:
            news_list: 新闻列表
            use_deepseek: 是否使用 DeepSeek (测试环境为 False, 生产为 True)
        """
        pass
    
    @abstractmethod
    def get_available_provider(self) -> str:
        """获取当前可用的 LLM 提供商"""
        pass
```

---

## 三、测试模块设计

### 3.1 测试金字塔

```
                    ▲
                   /│\        E2E 测试 (端到端)
                  / │ \       覆盖核心业务流程
                 /  │  \
                /───┼───\
               /    │    \      集成测试
              /     │     \     模块间协作
             /──────┼──────\
            /       │       \   单元测试
           /        │        \  单个模块功能
          ┌─────────┴─────────┐
          │    Mock/Stub       │
          │    隔离依赖        │
          └────────────────────┘
```

### 3.2 测试目录结构

```
tests/
├── conftest.py                   # pytest 配置 & fixtures
├── fixtures/                      # 测试数据
│   ├── sample_news.json
│   ├── sample_market.json
│   └── sample_fund.json
│
├── unit/                         # 单元测试
│   ├── test_market_crawler.py
│   ├── test_market_repo.py
│   ├── test_news_crawler.py
│   ├── test_news_repo.py
│   ├── test_minimax.py
│   ├── test_deepseek.py
│   └── test_news_analyzer.py
│
├── integration/                  # 集成测试
│   ├── test_news_flow.py        # 新闻流程: 爬取->存储->分析
│   ├── test_market_flow.py      # 市场流程: 爬取->存储->特征
│   └── test_llm_flow.py         # LLM 流程: 提取->分析
│
└── e2e/                          # 端到端测试
    ├── test_api_news.py          # 新闻 API 完整流程
    └── test_api_market.py        # 市场 API 完整流程
```

### 3.3 测试示例

#### 3.3.1 单元测试示例

```python
# tests/unit/test_news_crawler.py

import pytest
from unittest.mock import Mock, patch
from data.news.crawler import NewsCrawler


class TestNewsCrawler:
    """新闻爬虫单元测试"""
    
    def test_fetch_cailian_returns_list(self):
        """测试财联社返回新闻列表"""
        crawler = NewsCrawler()
        with patch('requests.get') as mock_get:
            mock_get.return_value = Mock(
                status_code=200,
                json=lambda: {"data": [{"title": "测试新闻"}]}
            )
            result = crawler.fetch_cailian()
            assert isinstance(result, list)
    
    def test_fetch_cailian_handles_error(self):
        """测试错误处理"""
        crawler = NewsCrawler()
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            result = crawler.fetch_cailian()
            assert result == []
    
    def test_deduplicate_by_url(self):
        """测试 URL 去重"""
        crawler = NewsCrawler()
        news = [
            NewsItem(title="A", content="", source="", url="http://a.com", published_at=None),
            NewsItem(title="A", content="", source="", url="http://a.com", published_at=None),
            NewsItem(title="B", content="", source="", url="http://b.com", published_at=None),
        ]
        deduplicated = crawler._deduplicate(news)
        assert len(deduplicated) == 2
```

#### 3.3.2 集成测试示例

```python
# tests/integration/test_news_flow.py

import pytest
from data.news.crawler import NewsCrawler
from data.news.repo import NewsRepo
from data.news.interfaces import NewsItem


class TestNewsFlow:
    """新闻流程集成测试"""
    
    def test_full_flow(self, db_connection):
        """测试完整流程: 爬取 -> 存储 -> 查询"""
        # 1. 爬取
        crawler = NewsCrawler()
        with patch.object(crawler, 'fetch_cailian') as mock:
            mock.return_value = [
                NewsItem(
                    title="测试新闻",
                    content="内容",
                    source="财联社",
                    url="http://test.com/1",
                    published_at=None
                )
            ]
            news = crawler.fetch_cailian()
        
        # 2. 存储
        repo = NewsRepo()
        saved = repo.save_news(news)
        assert saved > 0
        
        # 3. 查询
        retrieved = repo.get_news(days=1)
        assert len(retrieved) > 0
        assert retrieved[0].title == "测试新闻"
```

#### 3.3.3 E2E 测试示例

```python
# tests/e2e/test_api_news.py

import pytest


class TestNewsAPI:
    """新闻 API 端到端测试"""
    
    def test_news_list_api(self, client):
        """测试新闻列表 API"""
        resp = client.get("/api/news/list?days=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data
    
    def test_news_sync_api(self, client):
        """测试新闻同步 API"""
        resp = client.post("/api/news/sync")
        assert resp.status_code == 200
    
    def test_news_analyze_api(self, client):
        """测试新闻分析 API"""
        resp = client.post("/api/news/analyze", json={"days": 1})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "summary" in data["data"]
        assert "deep_analysis" in data["data"]
```

### 3.4 测试配置

```python
# tests/conftest.py

import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_news():
    """示例新闻数据"""
    import json
    path = Path(__file__).parent / "fixtures" / "sample_news.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def mock_minimax():
    """Mock MiniMax 客户端"""
    from unittest.mock import Mock
    client = Mock()
    client.chat.return_value = "测试回复"
    client.is_available.return_value = True
    return client


@pytest.fixture
def mock_deepseek():
    """Mock DeepSeek 客户端"""
    from unittest.mock import Mock
    client = Mock()
    client.chat.return_value = "深度分析结果"
    client.is_available.return_value = True
    return client


@pytest.fixture
def client():
    """Flask 测试客户端"""
    from server.app import create_app
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
```

---

## 四、API 接口设计

### 4.1 新闻接口

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/news/list` | 新闻列表 | 5分钟 |
| GET | `/api/news/latest` | 最新新闻 | 5分钟 |
| POST | `/api/news/sync` | 手动同步 | - |
| POST | `/api/news/analyze` | 分析新闻 | - |
| GET | `/api/news/analysis/latest` | 最新分析 | 1小时 |

### 4.2 市场接口

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/market/macro` | 宏观数据 | 1天 |
| GET | `/api/market/money-flow` | 资金流向 | 1小时 |
| GET | `/api/market/sentiment` | 市场情绪 | 1小时 |
| GET | `/api/market/global` | 全球宏观 | 1小时 |
| GET | `/api/market/features` | 特征合并 | 1小时 |
| POST | `/api/market/sync` | 同步数据 | - |

### 4.3 LLM 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/llm/status` | LLM 状态 |
| POST | `/api/llm/chat` | 通用对话 |

---

## 五、前端页面设计

### 5.1 页面结构

```
frontend/src/
├── views/
│   ├── FundHome.vue           # 基金列表 (已有)
│   ├── FundDetail.vue         # 基金详情 (已有)
│   ├── FundPredict.vue        # 预测中心 (已有)
│   │
│   ├── NewsHome.vue           # ★ 新闻首页
│   ├── NewsList.vue           # ★ 新闻列表
│   ├── NewsDetail.vue         # ★ 新闻详情
│   └── NewsAnalysis.vue       # ★ 新闻分析报告
│
├── components/
│   ├── NewsCard.vue           # ★ 新闻卡片组件
│   ├── NewsFilter.vue         # ★ 新闻筛选组件
│   ├── SentimentChart.vue     # ★ 市场情绪图表
│   └── MacroIndicator.vue     # ★ 宏观指标卡片
│
└── api/
    └── news.js                # ★ 新闻 API 请求
```

### 5.2 页面设计

#### 5.2.1 新闻首页 (NewsHome)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🗞️ 财经新闻                                              [刷新] [分析] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐           │
│  │  宏观       │  行业       │  公司       │  全球       │  全部    │
│  └─────────────┴─────────────┴─────────────┴─────────────┘           │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 📈 市场情绪                                    2024-01-15    │     │
│  │                                                           │     │
│  │  涨停: 45  │  跌停: 12  │  成交额: 8,520亿            │     │
│  │  北向资金: +25.6亿  │  主力: -120.5亿               │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 📰 最新新闻                                               更多 > │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ [重要] 央行：保持流动性合理充裕                          宏观 │     │
│  │ 财联社 · 10:30                                            │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ 美伊冲突升级 油价大涨超3%                                全球 │     │
│  │ 华尔街见闻 · 09:15                                        │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ 证监会：推进资本市场改革                                政策 │     │
│  │ 新华社 · 08:00                                            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 新闻列表 (NewsList)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📋 新闻列表                          [筛选 ▼] [日期范围 📅]          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  筛选条件:                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                   │
│  │ 来源: 全部 ▼ │ │ 分类: 全部 ▼ │ │ 重要性: 全部▼│                   │
│  └──────────────┘ └──────────────┘ └──────────────┘                   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ ☐ │ 央行：保持流动性合理充裕                          10:30 │     │
│  │   │ 宏观 │ 重要 │ 财联社                               │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ ☐ │ 美伊冲突升级 油价大涨                            09:15 │     │
│  │   │ 全球 │ 普通 │ 华尔街见闻                           │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ ☐ │ 证监会：推进资本市场改革                          08:00 │     │
│  │   │ 政策 │ 普通 │ 新华社                               │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  共 156 条新闻  [上一页] [1] [2] [3] ... [8] [下一页]                │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.3 新闻分析报告 (NewsAnalysis)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📊 今日新闻分析                                           2024-01-15 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 📌 核心要点                                                    │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │ • 央行：保持流动性合理充裕，利好股市                         │     │
│  │ • 美伊冲突：避险情绪升温，黄金原油上涨                       │     │
│  │ • 外资：北向资金净流入25.6亿，市场情绪偏多                   │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 📈 市场判断                                                    │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │                                                              │     │
│  │  影响: 🟢 看涨                                                │     │
│  │                                                              │     │
│  │  1. 宏观层面：央行流动性支持，估值有支撑                      │     │
│  │  2. 资金面：北向资金净流入，杠杆资金平稳                      │     │
│  │  3. 情绪面：涨停数大于跌停数，市场赚钱效应良好                │     │
│  │                                                              │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 💼 投资建议                                                    │     │
│  │ ─────────────────────────────────────────────────────────────│     │
│  │                                                              │     │
│  │  建议: 适度加仓，关注金融、科技、消费板块                     │     │
│  │                                                              │     │
│  │  风险提示：美伊冲突升级可能带来短期波动                       │     │
│  │                                                              │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  [分析新闻] [查看原始新闻]                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 前端 API 请求

```javascript
// frontend/src/api/news.js

import axios from 'axios'

export function getNewsList(params) {
  return axios.get('/api/news/list', { params })
}

export function getNewsLatest() {
  return axios.get('/api/news/latest')
}

export function syncNews() {
  return axios.post('/api/news/sync')
}

export function analyzeNews(data) {
  return axios.post('/api/news/analyze', data)
}

export function getNewsAnalysis() {
  return axios.get('/api/news/analysis/latest')
}

export function getMarketSentiment() {
  return axios.get('/api/market/sentiment')
}

export function getMarketFeatures() {
  return axios.get('/api/market/features')
}
```

### 5.4 路由配置

```javascript
// frontend/src/router/index.js

const routes = [
  {
    path: '/',
    name: 'FundHome',
    component: () => import('@/views/FundHome.vue'),
  },
  {
    path: '/fund/:code',
    name: 'FundDetail',
    component: () => import('@/views/FundDetail.vue'),
  },
  {
    path: '/predict',
    name: 'FundPredict',
    component: () => import('@/views/FundPredict.vue'),
  },
  // ★ 新闻相关路由
  {
    path: '/news',
    name: 'NewsHome',
    component: () => import('@/views/NewsHome.vue'),
    meta: { title: '财经新闻' },
  },
  {
    path: '/news/list',
    name: 'NewsList',
    component: () => import('@/views/NewsList.vue'),
    meta: { title: '新闻列表' },
  },
  {
    path: '/news/:id',
    name: 'NewsDetail',
    component: () => import('@/views/NewsDetail.vue'),
    meta: { title: '新闻详情' },
  },
  {
    path: '/news/analysis',
    name: 'NewsAnalysis',
    component: () => import('@/views/NewsAnalysis.vue'),
    meta: { title: '新闻分析' },
  },
]
```

### 5.5 页面组件说明

| 组件 | 说明 |
|------|------|
| `NewsHome.vue` | 新闻首页，展示市场情绪+最新新闻 |
| `NewsList.vue` | 新闻列表，支持筛选、分页 |
| `NewsDetail.vue` | 新闻详情页 |
| `NewsAnalysis.vue` | AI 分析报告页 |
| `NewsCard.vue` | 新闻卡片组件 |
| `NewsFilter.vue` | 筛选组件 |
| `SentimentChart.vue` | 市场情绪图表 (ECharts) |
| `MacroIndicator.vue` | 宏观指标卡片 |

---

## 六、数据库表设计

### 5.1 新闻表

```sql
CREATE TABLE news_data (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    news_date DATE COMMENT '新闻日期',
    title VARCHAR(256) NOT NULL,
    content TEXT,
    source VARCHAR(32),
    url VARCHAR(512),
    published_at DATETIME,
    category VARCHAR(32),
    sentiment VARCHAR(16),
    importance TINYINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_url (url),
    KEY idx_news_date (news_date),
    KEY idx_category (category),
    KEY idx_published_at (published_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 5.2 新闻存储策略

| 策略 | 说明 |
|------|------|
| **去重** | 基于 URL 唯一键，避免重复爬取 |
| **按日存储** | 新增 `news_date` 字段，标识新闻所属日期 |
| **自动清理** | 每天只保留最近 N 天的新闻（默认 30 天） |
| **增量爬取** | 只爬取当天新闻，跳过历史数据 |

### 5.3 爬取频率控制

| 时段 | 频率 | 说明 |
|------|------|------|
| 8:00 | 1次 | 早盘前新闻汇总 |
| 12:00 | 1次 | 午间新闻 |
| 16:00 | 1次 | 收盘后重要新闻 |
| 20:00 | 1次 | 晚间新闻汇总 |

**频率限制：**
- 每个数据源最小间隔 4 小时
- 单日最多 4 次爬取
- 失败后 exponential backoff 重试

### 5.4 新闻爬取实现

```python
# data/news/crawler.py

import time
from datetime import datetime, date
from typing import List, Optional
from dataclasses import dataclass
from threading import Lock


@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str = "general"
    news_date: date = None  # 新增：新闻日期

    def __post_init__(self):
        if self.news_date is None:
            self.news_date = self.published_at.date() if self.published_at else date.today()


class NewsCrawler:
    """新闻爬虫 - 支持增量爬取、频率控制"""
    
    # 频率控制
    MIN_INTERVAL_HOURS = 4
    MAX_DAILY_FETCHES = 4
    
    # 存储上次爬取时间
    _last_fetch: dict = {}
    _daily_count: dict = {}
    _lock = Lock()
    
    def __init__(self):
        self._reset_daily_count_if_needed()
    
    def _reset_daily_count_if_needed(self):
        """每日重置计数"""
        today = date.today().isoformat()
        if self._daily_count.get("date") != today:
            self._daily_count = {"date": today, "count": 0}
    
    def can_fetch(self, source: str) -> bool:
        """检查是否可以爬取"""
        with self._lock:
            self._reset_daily_count_if_needed()
            
            # 检查每日次数限制
            if self._daily_count["count"] >= self.MAX_DAILY_FETCHES:
                return False
            
            # 检查时间间隔
            last_time = self._last_fetch.get(source)
            if last_time:
                hours_since = (datetime.now() - last_time).total_seconds() / 3600
                if hours_since < self.MIN_INTERVAL_HOURS:
                    return False
            
            return True
    
    def fetch_and_save(self, repo) -> dict:
        """
        爬取并保存新闻
        返回: {"fetched": 数量, "saved": 数量, "duplicates": 数量}
        """
        if not self.can_fetch("all"):
            return {"error": "频率限制", "can_fetch_after": "4小时后"}
        
        # 1. 增量爬取（只爬当天）
        news_list = self.fetch_today_news()
        
        # 2. 去重保存
        saved = 0
        duplicates = 0
        for news in news_list:
            try:
                result = repo.save_news([news])
                if result > 0:
                    saved += 1
            except Exception as e:
                if "Duplicate" in str(e):
                    duplicates += 1
        
        # 3. 更新计数
        with self._lock:
            self._daily_count["count"] += 1
            self._last_fetch["all"] = datetime.now()
        
        return {"fetched": len(news_list), "saved": saved, "duplicates": duplicates}
    
    def fetch_today_news(self) -> List[NewsItem]:
        """只爬取今天的新闻（增量）"""
        # 实现具体爬取逻辑...
        pass
```

### 5.5 新闻存储实现

```python
# data/news/repo.py

class NewsRepo:
    """新闻仓储 - 去重、清理"""
    
    KEEP_DAYS = 30  # 默认保留30天
    
    def save_news(self, news_list: List[NewsItem]) -> int:
        """保存新闻（自动去重）"""
        if not news_list:
            return 0
        
        # 基于 URL 去重
        sql = """
        INSERT IGNORE INTO news_data 
        (news_date, title, content, source, url, published_at, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        # 执行插入...
    
    def cleanup_old_news(self, keep_days: int = None) -> int:
        """清理过期新闻"""
        keep_days = keep_days or self.KEEP_DAYS
        sql = """
        DELETE FROM news_data 
        WHERE news_date < DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        # 执行清理...
    
    def get_today_news(self) -> List[NewsItem]:
        """获取今天的新闻"""
        sql = "SELECT * FROM news_data WHERE news_date = CURDATE()"
        # ...
```

### 5.6 定时调度配置

```yaml
# config.yaml
scheduler:
  news:
    enabled: true
    keep_days: 30  # 保留30天
    
  jobs:
    - name: "news_crawl_morning"
      cron: "0 8 * * *"     # 每天8:00
      action: "news_crawler.fetch"
      
    - name: "news_crawl_noon"
      cron: "0 12 * * *"    # 每天12:00
      
    - name: "news_crawl_evening"
      cron: "0 16 * * *"    # 每天16:00
      
    - name: "news_crawl_night"
      cron: "0 20 * * *"    # 每天20:00
      
    - name: "news_cleanup"
      cron: "0 3 * * *"     # 每天3点清理过期数据
      
    - name: "news_analyze"
      cron: "0 16,20 * * 1-5"  # 收盘后分析
```

### 5.7 分析结果表

```sql
CREATE TABLE news_analysis (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    analysis_date DATE NOT NULL,
    news_count INT,
    summary TEXT,
    deep_analysis TEXT,
    market_impact VARCHAR(16),
    key_events JSON,
    investment_advice TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_date (analysis_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 5.8 市场数据表

```sql
-- 宏观经济
CREATE TABLE macro_data (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    indicator VARCHAR(32) NOT NULL,
    period VARCHAR(16) NOT NULL,
    value DECIMAL(20,4),
    unit VARCHAR(16),
    source VARCHAR(32),
    publish_date DATE,
    trade_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_indicator_period (indicator, period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 资金流向
CREATE TABLE money_flow (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL,
    north_money DECIMAL(20,2),
    main_money DECIMAL(20,2),
    margin_balance DECIMAL(20,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 市场情绪
CREATE TABLE market_sentiment (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL,
    volume DECIMAL(20,2),
    up_count INT,
    down_count INT,
    turnover_rate DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 全球宏观
CREATE TABLE global_macro (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    close_price DECIMAL(14,4),
    change_pct DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_symbol_date (symbol, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 六、实施计划

### Phase 1: 基础设施 (1天)

- [ ] 新增数据库表
- [ ] 创建模块目录结构
- [ ] 定义接口 (interfaces.py)
- [ ] 配置 pytest

### Phase 2: 新闻模块 (2.5天)

- [ ] 实现 news/crawler.py (支持增量爬取、频率控制)
- [ ] 实现 news/repo.py (去重、按日存储、自动清理)
- [ ] 单元测试 (test_crawler.py, test_repo.py)
- [ ] 集成测试 (test_news_flow.py)

### Phase 3: 市场模块 (2天)

- [ ] 实现 market/crawler.py
- [ ] 实现 market/repo.py
- [ ] 单元测试
- [ ] 集成测试

### Phase 4: LLM 模块 (1.5天)

- [ ] 实现 llm/deepseek.py
- [ ] 实现 llm/news_analyzer.py
- [ ] 单元测试
- [ ] 集成测试 (双 LLM 流程)

### Phase 5: API 层 (1天)

- [ ] 实现 server/routes/news.py
- [ ] 实现 server/routes/market.py
- [ ] E2E 测试

### Phase 6: 调度集成 (0.5天)

- [ ] 配置定时任务
- [ ] 整体联调

---

## 七、工作量估算

| 阶段 | 开发 | 测试 | 小计 |
|------|------|------|------|
| 基础设施 | 1天 | 0.5天 | 1.5天 |
| 新闻模块 | 1.5天 | 0.5天 | 2天 |
| 市场模块 | 1.5天 | 0.5天 | 2天 |
| LLM 模块 | 1天 | 0.5天 | 1.5天 |
| API 层 | 0.5天 | 0.5天 | 1天 |
| 调度集成 | 0.5天 | - | 0.5天 |
| **总计** | **6天** | **2.5天** | **8.5天** |

---

## 八、模块独立性与测试矩阵

### 8.1 测试覆盖

| 模块 | 单元测试 | 集成测试 | E2E |
|------|----------|----------|-----|
| news/crawler | ✓ | ✓ | - |
| news/repo | ✓ | ✓ | - |
| market/crawler | ✓ | ✓ | - |
| market/repo | ✓ | ✓ | - |
| llm/minimax | ✓ | - | - |
| llm/deepseek | ✓ | - | - |
| llm/news_analyzer | ✓ | ✓ | - |
| API routes | - | ✓ | ✓ |

### 8.2 Mock 策略

| 依赖 | Mock 方式 |
|------|-----------|
| requests.get | unittest.mock.patch |
| MySQL | fixture (db_connection) |
| Cache | unittest.mock.Mock |
| MiniMax | fixture (mock_minimax) |
| DeepSeek | fixture (mock_deepseek) |

---

## 九、模块化开发计划

### 9.1 Agent 架构

```
                         ┌─────────────────────┐
                         │    管理 Agent       │
                         │  (Coordinator)      │
                         │  负责任务分发/整合   │
                         └──────────┬──────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐           ┌───────────────┐
│  Agent 1     │         │  Agent 2     │           │  Agent 3     │
│  新闻模块    │         │  市场模块    │           │  LLM 模块    │
│  开发+测试   │         │  开发+测试   │           │  开发+测试   │
└───────────────┘         └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │    Agent 4          │
                         │  API + 调度集成     │
                         │  开发 + E2E测试    │
                         └─────────────────────┘
```

### 9.2 模块任务分配

| Agent | 模块 | 职责 | 输出 |
|-------|------|------|------|
| **管理Agent** | 协调者 | 任务分发、进度跟踪、模块整合、最终测试 | 整合系统 |
| **Agent 1** | 新闻模块 | 爬虫+仓储+单元测试 | news/ + tests/unit/test_news_* |
| **Agent 2** | 市场模块 | 爬虫+仓储+单元测试 | market/ + tests/unit/test_market_* |
| **Agent 3** | LLM模块 | 双LLM客户端+分析器+单元测试 | llm/ + tests/unit/test_llm_* |
| **Agent 4** | API集成 | 路由+调度+E2E测试 | routes/ + tests/e2e/ |

### 9.3 开发流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           开发流程                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 管理Agent: 初始化项目                                               │
│     - 创建模块目录结构                                                  │
│     - 定义接口 (interfaces.py)                                        │
│     - 配置 pytest fixtures                                              │
│                                                                         │
│  2. Agent 1-3: 并行开发各自模块                                        │
│     - 实现核心功能                                                      │
│     - 编写单元测试 (覆盖率 > 80%)                                      │
│     - 本地自测通过                                                     │
│                                                                         │
│  3. Agent 4: API集成开发                                               │
│     - 实现路由                                                          │
│     - 集成测试                                                          │
│     - E2E测试                                                          │
│                                                                         │
│  4. 管理Agent: 模块整合                                                │
│     - 合并代码                                                          │
│     - 解决冲突                                                          │
│     - 全面自测                                                          │
│     - 部署验证                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.4 各模块详细任务

#### Agent 1: 新闻模块

```
任务: 实现新闻爬取、存储、分析全流程

data/news/
├── interfaces.py          # 接口定义 (已有框架)
├── crawler.py             # 新闻爬虫
│   - fetch_cailian()      # 财联社
│   - fetch_wallstreet()    # 华尔街见闻  
│   - fetch_xinhua()       # 新华社
│   - fetch_today()        # 增量爬取
│   - can_fetch()          # 频率控制
│
├── repo.py                # 新闻仓储
│   - save_news()         # 去重保存
│   - get_news()          # 查询
│   - cleanup_old()       # 清理过期
│   - get_today()         # 当天新闻
│
└── tests/
    ├── __init__.py
    ├── test_crawler.py
    │   - test_fetch_returns_list
    │   - test_deduplicate_by_url
    │   - test_frequency_control
    │   - test_incremental_fetch
    │
    └── test_repo.py
        - test_save_news_deduplication
        - test_cleanup_old_news
        - test_get_today_news

验收标准:
- 单元测试覆盖率 > 80%
- 通过本地自测
```

#### Agent 2: 市场模块

```
任务: 实现市场数据爬取、存储

data/market/
├── interfaces.py          # 接口定义
├── crawler.py            # 市场爬虫
│   - fetch_macro()       # 宏观经济
│   - fetch_money_flow()  # 资金流向
│   - fetch_sentiment()   # 市场情绪
│   - fetch_global()      # 全球宏观
│
├── repo.py                # 市场仓储
│   - save_macro()
│   - get_macro()
│   - save_money_flow()
│   - get_money_flow()
│   - save_sentiment()
│   - get_sentiment()
│   - get_market_features()  # 合并特征
│
└── tests/
    ├── __init__.py
    ├── test_crawler.py
    └── test_repo.py

验收标准:
- 单元测试覆盖率 > 80%
- 通过本地自测
```

#### Agent 3: LLM模块

```
任务: 实现双LLM分析流程

analysis/llm/
├── interfaces.py          # 接口定义
├── minimax.py             # MiniMax客户端 (重构自client.py)
│   - chat()
│   - is_available()
│
├── deepseek.py            # DeepSeek客户端
│   - chat()
│   - is_available()
│
├── news_analyzer.py       # 新闻分析器
│   - extract_key_info()   # MiniMax提取
│   - analyze()            # 双LLM分析
│
└── tests/
    ├── __init__.py
    ├── test_minimax.py
    │   - test_chat
    │   - test_is_available
    │   - test_extract_key_info
    │
    ├── test_deepseek.py
    │   - test_chat
    │   - test_is_available
    │
    └── test_analyzer.py
        - test_extract_key_info
        - test_full_analyze_flow

验收标准:
- 单元测试覆盖率 > 80%
- Mock外部API测试通过
- 测试环境使用 MiniMax (免费)
- Production 环境使用 DeepSeek
```

#### Agent 4: API集成

```
任务: 路由开发、调度集成、E2E测试

server/routes/
├── news.py                # 新闻API
│   - GET /api/news/list
│   - GET /api/news/latest
│   - POST /api/news/sync
│   - POST /api/news/analyze
│   - GET /api/news/analysis/latest
│
├── market.py              # 市场API
│   - GET /api/market/macro
│   - GET /api/market/money-flow
│   - GET /api/market/sentiment
│   - GET /api/market/features
│   - POST /api/market/sync
│
├── scheduler.py          # 定时任务
│   - news_crawl_job
│   - news_analyze_job
│   - market_sync_job
│   - cleanup_job
│
tests/e2e/
├── __init__.py
├── test_news_api.py
│   - test_news_list
│   - test_news_sync
│   - test_news_analyze
│
└── test_market_api.py
    - test_market_features

验收标准:
- 所有API返回正确
- E2E测试通过
```

### 9.5 测试要求

#### 单元测试要求

```python
# 每个模块必须包含以下测试:

# 1. 正常流程测试
def test_xxx_success():
    result = module.method(input)
    assert result == expected

# 2. 异常处理测试
def test_xxx_error():
    with pytest.raises(Exception):
        module.method(invalid_input)

# 3. 边界条件测试
def test_xxx_edge_case():
    result = module.method(empty_input)
    assert result == default_value

# 4. Mock外部依赖测试
@patch('requests.get')
def test_xxx_with_mock(mock_get):
    mock_get.return_value = Mock(...)
    result = module.method(...)
    assert result == expected
```

#### 验收标准

| 指标 | 要求 |
|------|------|
| 单元测试覆盖率 | > 80% |
| 测试通过率 | 100% |
| API可用性 | 100% |
| E2E通过率 | 100% |

#### LLM 环境配置

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM 环境配置策略                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  测试环境 (.env)                                                        │
│  ┌─────────────────────────────────────────────┐                       │
│  │ LLM_PROVIDER=minimax                       │                       │
│  │ MINIMAX_API_KEY=sk-cp-xxx                  │  ← 免费使用           │
│  │ DEEPSEEK_API_KEY=(可选)                    │                       │
│  └─────────────────────────────────────────────┘                       │
│                                                                         │
│  Production 环境 (.env.prod)                                            │
│  ┌─────────────────────────────────────────────┐                       │
│  │ LLM_PROVIDER=deepseek                      │                       │
│  │ MINIMAX_API_KEY=sk-cp-xxx                  │                       │
│  │ DEEPSEEK_API_KEY=sk-xxx                    │  ← 付费使用           │
│  └─────────────────────────────────────────────┘                       │
│                                                                         │
│  配置加载逻辑:                                                          │
│  1. 读取 LLM_PROVIDER 指定使用哪个LLM                                  │
│  2. 如果指定provider不可用，自动fallback到可用provider                 │
│  3. 测试环境默认使用 minimax，节省费用                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# analysis/llm/factory.py

import os
from typing import Optional


class LLMFactory:
    """LLM 工厂 - 根据配置创建客户端"""
    
    @staticmethod
    def get_client(provider: str = None) -> 'LLMClientPort':
        """
        获取 LLM 客户端
        provider: 'minimax' | 'deepseek' | 'auto'
        """
        provider = provider or os.getenv('LLM_PROVIDER', 'minimax')
        
        if provider == 'deepseek':
            from .deepseek import DeepSeekClient
            return DeepSeekClient()
        
        # 默认使用 minimax
        from .minimax import MiniMaxClient
        return MiniMaxClient()
    
    @staticmethod
    def get_analyzer() -> 'NewsAnalyzerPort':
        """获取新闻分析器（自动选择可用LLM）"""
        from .news_analyzer import NewsAnalyzer
        
        # 优先使用配置的provider，fallback到可用
        provider = os.getenv('LLM_PROVIDER', 'minimax')
        
        try:
            if provider == 'deepseek':
                from .deepseek import DeepSeekClient
                deepseek = DeepSeekClient()
                if deepseek.is_available():
                    return NewsAnalyzer(minimax_client=None, deepseek_client=deepseek)
        except:
            pass
        
        # Fallback到 minimax
        from .minimax import MiniMaxClient
        minimax = MiniMaxClient()
        return NewsAnalyzer(minimax_client=minimax, deepseek_client=None)
```

### 9.6 模块联调流程

```
Agent 1-4 完成各自开发
           │
           ▼
    管理Agent: 代码合并
           │
           ▼
    解决冲突 (如有)
           │
           ▼
    运行全面测试
    - pytest tests/unit/
    - pytest tests/integration/
    - pytest tests/e2e/
           │
           ▼
    部署验证
           │
           ▼
    完成
```

---

## 十、时间估算

| Agent | 模块 | 开发 | 测试 | 单元测试 | 合计 |
|-------|------|------|------|----------|------|
| Agent 1 | 新闻模块 | 1.5天 | 0.5天 | 0.5天 | 2.5天 |
| Agent 2 | 市场模块 | 1.5天 | 0.5天 | 0.5天 | 2.5天 |
| Agent 3 | LLM模块 | 1天 | 0.5天 | 0.5天 | 2天 |
| Agent 4 | API集成 | 1天 | 0.5天 | 0.5天 | 2天 |
| 管理Agent | 整合 | 0.5天 | 1天 | - | 1.5天 |
| **总计** | - | **5.5天** | **3天** | **2天** | **10.5天** |

---

## 十一、进度追踪

### 11.1 里程碑

| 阶段 | 里程碑 | 验收 |
|------|--------|------|
| M1 | 基础设施完成 | 目录结构、接口定义、pytest配置 |
| M2 | 模块独立完成 | Agent 1-4各自测试通过 |
| M3 | 集成完成 | 全部代码合并，E2E通过 |
| M4 | 上线 | 部署验证，系统可用 |

### 11.2 每日检查

- 每日站会: 各Agent汇报进度
- 代码合并: 每日下班前合并到dev分支
- 测试报告: 每日生成测试覆盖率报告
