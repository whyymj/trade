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
├── server/              # Flask 后端 API
│   ├── app.py          # Flask 应用
│   ├── routes/         # API 路由
│   │   ├── fund.py    # 基金 API
│   │   ├── news.py    # 新闻 API
│   │   └── market.py  # 市场 API
│   └── utils/         # 工具函数
├── data/               # 数据层
│   ├── fund/          # 基金数据
│   ├── news/          # 新闻数据
│   │   ├── crawler.py # 新闻爬虫
│   │   └── repo.py   # 新闻仓储
│   └── market/        # 市场数据
│       ├── crawler.py # 市场爬虫
│       └── repo.py   # 市场仓储
├── analysis/          # 分析模块
│   ├── fund_metrics.py # 基金指标
│   ├── fund_lstm.py   # LSTM 预测
│   └── llm/           # LLM 分析
│       ├── minimax.py  # MiniMax 客户端
│       ├── deepseek.py # DeepSeek 客户端
│       └── news_analyzer.py # 新闻分析器
├── frontend/          # Vue 3 前端
│   └── src/
│       ├── views/    # 页面组件
│       ├── components/ # 通用组件
│       └── api/      # API 请求
└── tests/            # 测试用例
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
| tests/test_fund.py | 26 |
| tests/test_news.py | 23 |
| tests/test_market.py | 20 |
| tests/test_llm.py | 24 |
| tests/test_integration.py | 16 |
| tests/test_e2e.py | 11 |
| **总计** | **120+** |

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
