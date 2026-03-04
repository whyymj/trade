# 基金新闻关联分析 - 重构实施计划

## 设计原则

1. **模块独立性** - 各模块独立开发、测试、部署
2. **低耦合高内聚** - 模块间通过接口通信，减少直接依赖
3. **可替换性** - 底层实现可替换（如换数据源、换LLM）
4. **可测试性** - 每个模块独立测试，覆盖率 > 80%

---

## 模块架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         主Agent (协调层)                              │
│  - 模块协调                                                            │
│  - 流程编排                                                            │
│  - 错误处理                                                            │
└─────────────────────────────────────────────────────────────────────┘
           ↑                ↑                ↑                ↑
           │                │                │                │
    ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
    │  Agent 1   │  │  Agent 2   │  │  Agent 3   │  │  Agent 4   │
    │ 基金行业分析 │  │ 新闻行业分类 │  │ 关联匹配   │  │ 前端展示   │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 模块划分

### 模块1: 基金行业分析 (Agent 1)

**职责**: 分析基金行业标签

**子模块**:
```
fund_industry/
├── __init__.py
├── schema.py          # 基金行业表定义
├── repo.py            # 基金行业仓储
├── analyzer.py       # 行业分析器
├── fetcher.py        # 基金持仓数据获取 (akshare)
└── tests/
    ├── __init__.py
    ├── test_schema.py
    ├── test_repo.py
    ├── test_analyzer.py
    └── test_integration.py
```

**对外接口**:
```python
# fund_industry/__init__.py
class FundIndustryRepo:
    def save_industries(fund_code: str, industries: List[dict]) -> bool
    def get_industries(fund_code: str) -> List[dict]
    def delete_industries(fund_code: str) -> int

class FundIndustryAnalyzer:
    def analyze(fund_code: str) -> dict  # 返回 {"industries": [...], "confidence": 0.85}
```

**数据库表**:
```sql
CREATE TABLE fund_industry (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fund_code VARCHAR(20) NOT NULL,
    industry VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) DEFAULT 0,
    source VARCHAR(20) DEFAULT 'llm',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_fund_industry (fund_code, industry),
    INDEX idx_fund_code (fund_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**API接口**:
- `POST /api/fund-industry/analyze/:code` - 分析基金行业
- `GET /api/fund-industry/:code` - 获取基金行业

---

### 模块2: 新闻行业分类 (Agent 2)

**职责**: 对新闻进行行业分类

**子模块**:
```
news_industry/
├── __init__.py
├── schema.py          # 新闻行业表定义
├── repo.py            # 新闻行业仓储
├── classifier.py      # 行业分类器 (LLM + 关键词)
├── en_classifier.py   # 英文新闻分类器
├── translator.py      # 翻译器 (可选)
└── tests/
    ├── __init__.py
    ├── test_classifier.py
    ├── test_repo.py
    └── test_integration.py
```

**对外接口**:
```python
# news_industry/__init__.py
class NewsIndustryClassifier:
    def classify(news: dict) -> dict  # {"industry": "新能源", "confidence": 0.95, "keywords": [...]}
    def batch_classify(news_list: List[dict]) -> List[dict]

class NewsIndustryRepo:
    def save_analysis(news_id: str, industry: dict) -> bool
    def get_industry(news_id: str) -> dict
    def get_by_industry(industry: str, days: int) -> List[dict]
```

**数据库表**:
```sql
-- 新闻行业分类表
CREATE TABLE news_industry (
    id INT AUTO_INCREMENT PRIMARY KEY,
    news_id VARCHAR(100) NOT NULL,
    industry VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) DEFAULT 0,
    keywords JSON,
    source VARCHAR(20) DEFAULT 'llm',
    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_news_id (news_id),
    INDEX idx_industry (industry),
    INDEX idx_analyzed_at (analyzed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**API接口**:
- `POST /api/news-industry/classify` - 批量分类新闻
- `GET /api/news-industry/:news_id` - 获取新闻行业
- `GET /api/news-industry/list?industry=新能源` - 按行业获取新闻

---

### 模块3: 关联匹配引擎 (Agent 3)

**职责**: 基金与新闻的关联匹配

**子模块**:
```
association/
├── __init__.py
├── matcher.py         # 关联匹配器
├── scorer.py          # 相关度评分
├── cache.py           # 关联结果缓存
└── tests/
    ├── __init__.py
    ├── test_matcher.py
    ├── test_scorer.py
    └── test_integration.py
```

**对外接口**:
```python
# association/__init__.py
class NewsMatcher:
    def match_fund_news(fund_code: str, days: int = 7, limit: int = 10) -> dict
    def match_industry_news(industry: str, days: int = 7) -> List[dict]

class RelevanceScorer:
    def score(news_industry: str, fund_industries: List[str]) -> float
```

**算法流程**:
```
1. 获取基金的行业标签 (调用 Agent 1)
2. 获取近期新闻的行业标签 (调用 Agent 2)
3. 计算相关度 = 行业匹配度 × 关键词匹配度 × 时间衰减
4. 排序返回
```

**API接口**:
- `GET /api/association/fund-news/:code` - 获取基金相关新闻
- `GET /api/association/industry-news?industry=新能源` - 获取行业新闻
- `GET /api/association/trending` - 热门行业新闻

---

### 模块4: 前端展示 (Agent 4)

**职责**: 卡通风格的UI展示

**页面**:
```
frontend/src/
├── views/
│   ├── FundIndustry.vue       # 基金行业分析页
│   ├── NewsIndustry.vue       # 新闻行业分类页
│   └── FundNewsAssociation.vue # 基金新闻关联页
├── components/
│   ├── IndustryTag.vue        # 行业标签组件
│   ├── NewsCard.vue           # 新闻卡片组件
│   ├── RelevanceScore.vue     # 相关度评分组件
│   └── IndustryCloud.vue      # 行业云组件
└── api/
    ├── fundIndustry.js         # 基金行业API
    ├── newsIndustry.js         # 新闻行业API
    └── association.js          # 关联API
```

**UI设计 (卡通风格)**:
- 圆润边角 (border-radius: 16px)
- 柔和配色 (渐变背景、 pastel色调)
- 活泼图标 (emoji + 插画)
- 动效 (hover放大、淡入)

---

## 模块依赖关系

```
┌─────────────────┐
│   前端展示      │  ← 依赖所有后端模块
│   (Agent 4)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ↓         ↓
┌─────────┐  ┌──────────────┐
│ 基金行业 │  │  新闻行业分类 │
│ (Agent1)│  │   (Agent 2) │
└────┬────┘  └──────┬───────┘
     │              │
     └──────┬───────┘
            ↓
    ┌──────────────┐
    │  关联匹配引擎 │
    │   (Agent 3)  │
    └──────────────┘
```

---

## 测试用例要求

### 每个模块测试覆盖

| 模块 | 单元测试 | 集成测试 | E2E测试 |
|------|---------|---------|---------|
| 基金行业分析 | 15+ | 5 | 2 |
| 新闻行业分类 | 15+ | 5 | 2 |
| 关联匹配 | 10+ | 5 | 2 |
| 前端展示 | 10+ | 3 | 2 |

### 测试分类

1. **单元测试** - 每个函数独立测试
2. **集成测试** - 模块间调用测试
3. **E2E测试** - 完整流程测试

---

## 实施顺序

### Sprint 1: 基金行业分析 (Agent 1)
- [ ] 创建模块目录结构
- [ ] 实现数据库表
- [ ] 实现仓储层
- [ ] 实现分析器
- [ ] 添加单元测试 (15个)
- [ ] 添加集成测试 (5个)
- [ ] API接口开发

### Sprint 2: 新闻行业分类 (Agent 2)
- [ ] 创建模块目录结构
- [ ] 实现数据库表
- [ ] 实现分类器 (LLM + 关键词)
- [ ] 添加单元测试 (15个)
- [ ] 添加集成测试 (5个)
- [ ] API接口开发

### Sprint 3: 关联匹配引擎 (Agent 3)
- [ ] 创建模块目录结构
- [ ] 实现匹配算法
- [ ] 实现评分算法
- [ ] 添加单元测试 (10个)
- [ ] 添加集成测试 (5个)
- [ ] API接口开发

### Sprint 4: 前端展示 (Agent 4)
- [ ] 创建页面组件
- [ ] 实现API调用
- [ ] 实现卡通风格UI
- [ ] 添加组件测试 (10个)
- [ ] E2E测试

### Sprint 5: 集成与优化
- [ ] 模块对接
- [ ] 性能优化
- [ ] 整体E2E测试
- [ ] Bug修复

---

## 文件结构

```
trade/
├── server/
│   ├── app.py                    # Flask应用 (保持轻量)
│   └── routes/
│       ├── __init__.py
│       ├── fund.py               # 基金相关路由 (保留)
│       ├── news.py                # 新闻相关路由 (保留)
│       ├── market.py             # 市场相关路由 (保留)
│       └── association.py         # 新增: 关联路由
│
├── modules/                       # 新增: 独立模块
│   ├── __init__.py
│   ├── fund_industry/            # 模块1: 基金行业分析
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── repo.py
│   │   ├── analyzer.py
│   │   ├── fetcher.py
│   │   └── tests/
│   │
│   ├── news_industry/            # 模块2: 新闻行业分类
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── repo.py
│   │   ├── classifier.py
│   │   └── tests/
│   │
│   └── association/              # 模块3: 关联匹配
│       ├── __init__.py
│       ├── matcher.py
│       ├── scorer.py
│       └── tests/
│
├── frontend/src/
│   ├── views/
│   │   ├── FundIndustry.vue      # 基金行业分析页
│   │   ├── NewsIndustry.vue     # 新闻行业分类页
│   │   └── FundNewsAssociation.vue # 基金新闻关联页
│   ├── components/
│   │   ├── IndustryTag.vue
│   │   ├── NewsCard.vue
│   │   ├── RelevanceScore.vue
│   │   └── IndustryCloud.vue
│   └── api/
│       ├── fundIndustry.js
│       ├── newsIndustry.js
│       └── association.js
│
└── docs/
    └── FUND_NEWS_ASSOCIATION_PLAN.md
```

---

## 预计工作量

| 模块 | Agent | 工作量 | 测试用例 |
|------|-------|--------|---------|
| 基金行业分析 | Agent 1 | 1天 | 20+ |
| 新闻行业分类 | Agent 2 | 1天 | 20+ |
| 关联匹配 | Agent 3 | 1天 | 15+ |
| 前端展示 | Agent 4 | 1天 | 15+ |
| 集成对接 | 主Agent | 0.5天 | 5+ |

**总计: 4.5天 | 75+测试用例**
