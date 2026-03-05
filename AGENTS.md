# FundProphet 项目文档

## 项目概述

FundProphet 基金分析系统 - 基于 AI 的基金分析平台

## 技术栈

- **后端**: Flask + Python 3.12
- **前端**: Vue 3 + Vite
- **数据库**: MySQL
- **AI**: LSTM 预测、LLM 分析 (MiniMax/DeepSeek)

## 项目结构

```
trade/
├── server/                # Flask 后端 API
│   ├── app.py            # Flask 应用
│   ├── routes/           # API 路由
│   │   ├── api.py       # 基金核心 API
│   │   ├── news.py      # 新闻 API
│   │   ├── market.py    # 市场 API
│   │   ├── fund_industry.py        # 基金行业 API
│   │   ├── news_classification.py  # 新闻分类 API
│   │   ├── fund_news_association.py # 基金-新闻关联 API
│   │   └── investment_advice.py    # 投资建议 API
│   └── utils.py          # 工具函数
├── data/                  # 数据层
│   ├── fund_repo.py      # 基金数据仓储
│   ├── fund_holdings.py  # 基金持仓数据
│   ├── news/             # 新闻数据
│   │   ├── crawler.py   # 新闻爬虫
│   │   └── repo.py     # 新闻仓储
│   └── market/           # 市场数据
│       ├── crawler.py   # 市场爬虫
│       └── repo.py     # 市场仓储
├── analysis/             # 分析模块
│   ├── fund_metrics.py  # 基金指标
│   ├── fund_lstm.py     # LSTM 预测
│   ├── fund_analyzer.py # 基金自动分析器
│   └── llm/             # LLM 分析
│       ├── minimax.py   # MiniMax 客户端
│       ├── deepseek.py  # DeepSeek 客户端
│       └── news_analyzer.py # 新闻分析器
├── modules/              # 业务模块
│   ├── fund_industry/   # 基金行业分析
│   │   ├── analyzer.py # 行业分析器
│   │   ├── repo.py    # 行业数据仓储
│   │   ├── interfaces.py # 接口定义
│   │   └── schema.py  # 数据模型
│   ├── news_classification/ # 新闻行业分类
│   │   ├── analyzer.py # 分类器
│   │   ├── repo.py    # 分类数据仓储
│   │   └── interfaces.py # 接口定义
│   ├── fund_news_association/ # 基金-新闻关联
│   │   ├── analyzer.py # 匹配引擎
│   │   ├── repo.py    # 关联数据仓储
│   │   └── interfaces.py # 接口定义
│   └── investment_advice/ # 投资建议
│       └── __init__.py # 投资建议生成器
├── frontend/             # Vue 3 前端
│   └── src/
│       ├── views/      # 页面组件
│       ├── components/ # 通用组件
│       └── api/       # API 请求
└── tests/               # 测试用例
```

## 功能模块

### 1. 基金模块
- 基金列表展示、搜索、筛选
- 基金详情查看
- 基金净值历史
- 基金指标分析 (收益率、夏普比率等)
- LSTM 走势预测

### 2. 新闻模块
- 财经新闻爬取 (东方财富、财联社、华尔街见闻)
- 新闻分类 (宏观、行业、全球)
- LLM 新闻分析
- 市场情绪分析

### 3. 市场模块
- 宏观经济数据 (GDP、CPI、PMI、M2)
- 资金流向 (北向资金、融资融券)
- 市场情绪 (涨跌停、成交额)
- 全球宏观 (美元指数、汇率)

## API 接口

### 基金 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/fund/list | 基金列表 |
| GET | /api/fund/:code | 基金详情 |
| GET | /api/fund/nav/:code | 基金净值 |
| GET | /api/fund/indicators/:code | 基金指标 |
| GET | /api/fund/cycle/:code | 周期分析 |
| POST | /api/fund/predict/:code | LSTM 预测 |

### 新闻 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/news/latest | 最新新闻 |
| GET | /api/news/list | 新闻列表 |
| GET | /api/news/detail/:id | 新闻详情 |
| POST | /api/news/sync | 同步新闻 |
| POST | /api/news/analyze | 分析新闻 |
| GET | /api/news/analysis/latest | 最新分析 |

### 市场 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/market/macro | 宏观数据 |
| GET | /api/market/money-flow | 资金流向 |
| GET | /api/market/sentiment | 市场情绪 |
| GET | /api/market/global | 全球宏观 |
| GET | /api/market/features | 市场特征 |
| POST | /api/market/sync | 同步数据 |

### 基金行业 API
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/fund-industry/analyze/:code | 分析基金行业 |
| GET | /api/fund-industry/:code | 获取基金行业 |
| GET | /api/fund-industry/primary/:code | 获取基金主要行业 |

### 新闻分类 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/news-classification/industries | 获取所有行业分类 |
| POST | /api/news-classification/classify | 分类单条新闻 |
| POST | /api/news-classification/classify-today | 分类今日所有新闻 |
| GET | /api/news-classification/industry/:code | 按行业获取新闻 |
| GET | /api/news-classification/stats | 获取行业统计 |
| GET | /api/news-classification/today | 获取今日已分类新闻 |

### 基金-新闻关联 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/fund-news/match/:code | 为基金匹配相关新闻 |
| GET | /api/fund-news/summary/:code | 获取基金新闻摘要 |
| GET | /api/fund-news/list | 获取有关联新闻的基金列表 |
| POST | /api/fund-news/match-all | 批量匹配所有基金新闻 |

### 投资建议 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/investment-advice/:code | 获取投资建议 |
| POST | /api/investment-advice/batch | 批量获取投资建议 |

## 开发规范

### 代码风格
- Python: PEP 8，使用 ruff 检查
- Vue: Composition API + `<script setup>`
- 前端 UI 风格: 卡通风格 (cute 风格)

### 命名约定
- 文件: 蛇形命名 (snake_case)
- 组件: PascalCase
- API 路由: RESTful

### 测试要求
- 修改代码后运行 `python -m pytest`
- 确保测试通过后再提交

### Agent 开发协作规则

#### 1. 模块独立性原则
- 各功能模块应尽可能降低相互依赖，通过清晰的接口进行通信
- 避免循环依赖，每个模块只依赖于其直接需要的服务
- 模块间使用统一的接口契约（Interface/抽象基类）进行交互
- 后期维护时应能独立替换或升级某个模块而不影响其他模块

#### 2. 复杂功能拆分与并行开发
- 复杂功能应划分为多个独立、可并行开发的子模块
- 每个子模块交由独立的子 Agent 负责开发
- 主 Agent 负责整体架构设计、接口定义和模块间协调对接
- 模块间通过标准化接口进行集成，避免硬编码耦合

#### 3. 测试驱动与验收流程
- 每个模块必须包含必要的单元测试用例
- 子 Agent 开发完成后需先进行自测，确保测试通过
- 所有子 Agent 完成开发和自测后，移交主 Agent 进行对接测试
- 集成测试通过后，方可接入整个项目系统

#### 4. 回归测试与问题追踪
- 完成整体开发后，必须执行完整的测试用例套件
- 测试过程中发现的问题，修复后需补充对应的测试用例
- 形成"问题-修复-测试用例"的闭环，防止问题复发
- 定期回顾测试覆盖情况，持续提升测试质量

## 常用命令

```bash
# 启动后端
python server.py

# 启动前端
cd frontend && npm run dev

# 运行所有测试
python -m pytest -v

# 运行特定测试
python -m pytest tests/test_news.py -v
python -m pytest tests/test_market.py -v
python -m pytest tests/test_llm.py -v
python -m pytest tests/test_e2e.py -v

# 安装依赖
pip install -r requirements.txt
cd frontend && npm install
```

## 测试统计

| 测试文件 | 测试数 |
|---------|--------|
| tests/test_fund.py | 43 |
| tests/test_llm.py | 31 |
| tests/test_news.py | 23 |
| tests/test_market.py | 20 |
| tests/test_integration.py | 16 |
| tests/test_fund_industry.py | 14 |
| tests/test_e2e.py | 11 |
| tests/test_news_classification.py | 9 |
| tests/test_fund_news_matcher.py | 8 |
| tests/test_fund_news_association.py | 6 |
| **总计** | **181** |

## 环境变量

`.env` 文件中配置:
- `DB_HOST` - MySQL 主机
- `DB_USER` - MySQL 用户
- `DB_PASSWORD` - MySQL 密码
- `DB_NAME` - 数据库名
- `MINIMAX_API_KEY` - MiniMax API
- `DEEPSEEK_API_KEY` - DeepSeek API

## 访问地址

- 后端: http://localhost:5050
- 前端: http://localhost:5173
- API 文档: http://localhost:5050/api
