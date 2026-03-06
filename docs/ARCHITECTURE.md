# 项目功能模块与架构说明

> 基于当前代码实际状态编写，反映项目真实结构。

---

## 一、项目定位

本项目是一个**基金与股票数据分析平台**，集数据抓取、存储、展示、AI 分析于一体。

- 前后端分离：Flask API 后端 + Vue3 前端
- 数据存储：MySQL（持久化）+ SQLite（本地缓存）
- AI 能力：LSTM 深度学习预测 + LLM 大模型分析
- 部署方式：本地开发 / 生产一体 / Docker Compose

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端 (Vue3 + Vite)                      │
│  FundHome / FundDetail / FundPredict / FundIndustry         │
│  NewsHome / NewsList / NewsAnalysis / NewsClassification     │
│  MarketHome / FundNewsAssociation                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP /api/*
┌──────────────────────────▼──────────────────────────────────┐
│                     后端 (Flask)                            │
│  server/app.py  ←  server/routes/*  ←  APScheduler 定时任务 │
│                                                             │
│  api.py  news.py  market.py  fund_industry.py              │
│  fund_news_association.py  news_classification.py           │
│  investment_advice.py                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     数据层 (data/)                          │
│  stock_repo / fund_repo / fund_fetcher / fund_holdings      │
│  news / market / lstm_repo / index_repo / cache             │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────────────┐
│       MySQL         │   │       分析层 (analysis/)         │
│  stock_* / fund_*   │   │  时域 / 频域 / ARIMA / LSTM      │
│  news_* / market_*  │   │  基金分析 / LLM / 综合报告       │
└─────────────────────┘   └─────────────────────────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │    业务模块 (modules/)   │
                           │  fund_industry           │
                           │  fund_news_association   │
                           │  news_classification     │
                           │  investment_advice       │
                           └─────────────────────────┘
```

---

## 三、目录结构

```
trade/
├── server.py                        # 启动入口（Flask，端口 5050）
├── config.yaml                      # 全局配置（MySQL、股票列表、日期范围等）
│
├── server/                          # 后端包
│   ├── app.py                       # Flask 应用工厂，注册蓝图，APScheduler 定时任务
│   ├── utils.py                     # 配置读写、akshare 拉取、DB 存储工具
│   ├── logging_config.py            # 日志配置（文件轮转 + 请求访问日志）
│   └── routes/                      # API 路由蓝图
│       ├── api.py                   # 股票数据接口（/api/*）
│       ├── news.py                  # 新闻接口（/api/news/*）
│       ├── market.py                # 市场/宏观数据（/api/market/*）
│       ├── fund_industry.py         # 基金行业分析（/api/fund-industry/*）
│       ├── fund_news_association.py # 基金新闻关联（/api/fund-news/*）
│       ├── news_classification.py   # 新闻行业分类（/api/news-classification/*）
│       └── investment_advice.py     # 投资建议（/api/investment-advice/*）
│
├── data/                            # 数据访问层
│   ├── mysql.py                     # MySQL 连接管理
│   ├── schema.py                    # 建表 DDL（启动时自动建表）
│   ├── stock_repo.py                # 股票数据读写
│   ├── fund_repo.py                 # 基金数据读写
│   ├── fund_fetcher.py              # 基金数据抓取（akshare）
│   ├── fund_holdings.py             # 基金持仓数据
│   ├── index_repo.py                # 指数数据
│   ├── lstm_repo.py                 # LSTM 模型持久化
│   ├── cache.py                     # SQLite 本地缓存（cache.db）
│   ├── news/                        # 新闻爬虫与存储
│   └── market/                      # 宏观市场数据抓取与存储
│
├── analysis/                        # 分析算法层
│   ├── time_domain.py               # 时域统计、移动均线、最大回撤、STL 分解
│   ├── frequency_domain.py          # FFT、功率谱、小波变换、主导周期
│   ├── arima_model.py               # ARIMA 建模与预测
│   ├── shape_similarity.py          # 形态相似度、DTW
│   ├── complexity.py                # 近似熵等非线性复杂度指标
│   ├── technical.py                 # 技术指标（MACD、RSI 等）
│   ├── factor_library.py            # 因子库
│   ├── ensemble_models.py           # 集成模型
│   ├── lstm_model.py                # 股票 LSTM（60日特征 → 未来5日预测）
│   ├── lstm_*.py                    # LSTM 辅助模块（训练/验证/监控/版本等）
│   ├── fund_lstm.py                 # 基金 LSTM 预测
│   ├── fund_analyzer.py             # 基金综合分析
│   ├── fund_metrics.py              # 基金绩效指标（收益率/夏普/回撤等）
│   ├── fund_benchmark.py            # 基准对比
│   ├── fund_data.py                 # 基金数据处理
│   ├── full_report.py               # 股票综合分析报告生成
│   ├── full_fund_report.py          # 基金综合分析报告生成
│   ├── ensemble_report.py           # 集成模型报告
│   └── llm/                         # LLM 大模型分析接口
│
├── modules/                         # 业务逻辑模块（LLM 驱动）
│   ├── fund_industry/               # 基金行业识别
│   │   ├── analyzer.py
│   │   ├── repo.py
│   │   ├── schema.py
│   │   └── interfaces.py
│   ├── fund_news_association/       # 基金-新闻关联匹配
│   ├── news_classification/         # 新闻行业分类
│   │   ├── analyzer.py
│   │   ├── repo.py
│   │   └── interfaces.py
│   └── investment_advice/           # 投资建议生成
│
├── frontend/                        # 前端工程（Vue3 + Vite）
│   └── src/
│       ├── main.js                  # 入口，挂载 Pinia、Router
│       ├── App.vue                  # 根组件
│       ├── views/                   # 页面组件（11个页面）
│       ├── components/              # 公共组件（MacroIndicator、SentimentChart）
│       ├── router/index.js          # 路由配置
│       ├── stores/                  # Pinia 状态管理
│       ├── api/                     # 接口封装
│       ├── utils/                   # 工具函数
│       └── styles/                  # 全局样式
│
├── docs/                            # 文档
├── tests/                           # 测试用例
├── Makefile                         # 常用命令快捷方式
├── docker-compose.yml               # Docker 一键部署（应用 + MySQL）
└── requirements.txt                 # Python 依赖
```

---

## 四、功能模块详解

### 4.1 股票数据模块

核心文件：`server/routes/api.py`、`data/stock_repo.py`、`server/utils.py`

数据来源：akshare（A股、港股日线行情）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/list` | GET | 获取数据库中的股票列表 |
| `/api/data?file=<symbol>` | GET | 获取指定股票日线数据 |
| `/api/fetch_data/<code>` | GET | 优先读本地，无则从 akshare 拉取 |
| `/api/add_stock` | POST | 添加股票并抓取近5年数据 |
| `/api/update_all` | POST | 增量更新所有股票至今日 |
| `/api/sync_all` | POST | 全量重新同步所有股票 |
| `/api/remove_stock` | POST | 移除股票及其数据 |
| `/api/analyze` | GET | 触发综合分析（时域/频域/ARIMA等） |
| `/api/lstm/train` | POST | 训练股票 LSTM 模型 |
| `/api/lstm/predict` | GET | 使用已训练模型预测未来5日 |

数据库表：`stock_meta`（元信息）、`stock_daily`（日线行情 OHLCV）

---

### 4.2 基金数据模块

核心文件：`data/fund_repo.py`、`data/fund_fetcher.py`、`data/fund_holdings.py`

- 通过 akshare 抓取基金净值、持仓、基本信息
- 支持增量更新与全量同步
- 定时任务：每日凌晨 3:00 自动同步所有基金近30日净值
- 前端页面：`FundHome`（列表与搜索）、`FundDetail`（详情与净值曲线）

---

### 4.3 新闻模块

核心文件：`server/routes/news.py`、`data/news/`

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/news/list` | GET | 新闻列表（分页、按来源/关键词筛选） |
| `/api/news/detail/<id>` | GET | 新闻详情 |
| `/api/news/crawl` | POST | 触发新闻爬取 |
| `/api/news/analysis` | GET | 新闻情感与热点分析 |

前端页面：`NewsHome`、`NewsList`、`NewsDetail`、`NewsAnalysis`

---

### 4.4 新闻行业分类模块

核心文件：`server/routes/news_classification.py`、`modules/news_classification/`

将财经新闻自动归类到行业（新能源、半导体、消费等），支持关键词匹配 + LLM 辅助分类。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/news-classification/industries` | GET | 获取所有行业分类列表 |
| `/api/news-classification/classify` | POST | 对新闻进行行业分类 |

前端页面：`NewsClassification`

---

### 4.5 基金行业分析模块

核心文件：`server/routes/fund_industry.py`、`modules/fund_industry/`

通过 LLM 分析基金持仓，识别主要投资行业，返回行业名称与置信度。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/fund-industry/analyze/<code>` | POST | 分析基金主要投资行业 |

前端页面：`FundIndustry`

---

### 4.6 基金-新闻关联模块

核心文件：`server/routes/fund_news_association.py`、`modules/fund_news_association/`

将基金的行业标签与新闻行业分类进行匹配，为每只基金推荐相关财经新闻。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/fund-news/match/<code>` | GET | 为基金匹配相关新闻（支持 days、min_confidence 参数） |

前端页面：`FundNewsAssociation`

---

### 4.7 投资建议模块

核心文件：`server/routes/investment_advice.py`、`modules/investment_advice/`

综合基金净值走势、行业新闻、市场情绪，由 LLM 生成短/中/长期投资建议。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/investment-advice/<code>` | GET | 获取基金投资建议（支持 days 参数） |

---

### 4.8 市场宏观数据模块

核心文件：`server/routes/market.py`、`data/market/`

抓取并展示宏观经济指标（CPI、PMI、利率等）与市场情绪指数。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/market/macro` | GET | 获取宏观经济指标数据 |
| `/api/market/sentiment` | GET | 获取市场情绪指数 |

前端页面：`MarketHome`（含 `MacroIndicator`、`SentimentChart` 组件）

---

### 4.9 LSTM 预测模块

核心文件：`analysis/lstm_model.py`（股票）、`analysis/fund_lstm.py`（基金）

**股票 LSTM：**

| 项目 | 说明 |
|------|------|
| 输入 | 60日多维特征（OHLCV + 技术指标） |
| 输出 | 未来5日涨跌方向（分类）+ 涨跌幅（回归） |
| 特性 | 交叉验证、超参优化、SHAP 可解释性、模型版本管理 |
| 并发控制 | 同一股票同时只允许一个训练任务，超时2小时自动释放锁 |

**基金 LSTM：**

- 支持关注列表自动训练（定时任务，每日凌晨 4:00）
- 前端页面：`FundPredict`（预测中心）

LSTM 辅助模块（均在 `analysis/` 下）：

| 文件 | 职责 |
|------|------|
| `lstm_training.py` | 训练流程 |
| `lstm_validation.py` | 验证与评估 |
| `lstm_predict_flow.py` | 预测流程 |
| `lstm_monitoring.py` | 训练监控 |
| `lstm_versioning.py` | 模型版本管理 |
| `lstm_diagnostics.py` | 诊断工具 |
| `lstm_fallback.py` | 降级策略 |
| `lstm_triggers.py` | 触发条件 |
| `lstm_losses.py` | 自定义损失函数 |
| `lstm_volatility_features.py` | 波动率特征工程 |
| `lstm_constants.py` | 常量定义 |
| `lstm_spec.py` | 规格定义 |

---

### 4.10 综合分析报告模块

核心文件：`analysis/full_report.py`（股票）、`analysis/full_fund_report.py`（基金）

整合时域、频域、ARIMA、形态相似度、复杂度等多维分析，生成图表与文字报告，输出到 `output/` 目录。

- 通过 API 触发：`GET /api/analyze?symbol=&start=&end=`
- 命令行独立运行：`python -m analysis.full_report data/xxx.csv`

---

## 五、定时任务

在 `server/app.py` 中通过 APScheduler 注册，随后端启动自动运行：

| 任务 | 执行时间 | 说明 |
|------|----------|------|
| 基金数据同步 | 每日 03:00 | 同步所有基金近30日净值数据 |
| LSTM 自动训练 | 每日 04:00 | 训练关注列表中的基金 LSTM 模型 |

---

## 六、数据流

```
akshare（外部数据源）
    │
    ▼
data/fund_fetcher / server/utils（抓取层）
    │
    ├──▶ data/cache.py（SQLite 短期缓存，减少重复请求）
    │
    ▼
MySQL（持久化存储）
    │
    ▼
data/stock_repo / fund_repo / news / market（数据访问层）
    │
    ├──▶ analysis/（算法分析层）
    │         │
    │         ├──▶ LSTM 预测（PyTorch）
    │         ├──▶ 统计分析（时域/频域/ARIMA）
    │         └──▶ LLM 分析（大模型接口）
    │
    ▼
server/routes/（API 层）
    │
    ▼
前端 Vue3（展示层）
```

---

## 七、前端页面路由

前端统一挂载在 `/app` 前缀下（与后端 `/api` 区分）。

| 路径 | 页面组件 | 功能 |
|------|----------|------|
| `/app/` | FundHome | 基金列表、搜索、关注管理 |
| `/app/fund/:code` | FundDetail | 基金详情、净值曲线、持仓分析 |
| `/app/predict` | FundPredict | LSTM 预测中心 |
| `/app/fund-industry` | FundIndustry | 基金行业分析 |
| `/app/news` | NewsHome | 财经新闻首页 |
| `/app/news/list` | NewsList | 新闻列表与筛选 |
| `/app/news/analysis` | NewsAnalysis | 新闻情感分析 |
| `/app/news/classification` | NewsClassification | 新闻行业分类 |
| `/app/news/:id` | NewsDetail | 新闻详情 |
| `/app/market` | MarketHome | 宏观市场数据与情绪指数 |
| `/app/fund-news` | FundNewsAssociation | 基金新闻关联推荐 |

---

## 八、数据库表结构

后端启动时通过 `data/schema.py` 自动建表（连接失败则跳过）。

| 表名 | 说明 | 关键字段 |
|------|------|----------|
| `stock_meta` | 股票元信息 | `symbol`（唯一）、`name`、`first_trade_date`、`last_trade_date` |
| `stock_daily` | 股票日线行情 | `symbol`、`trade_date`、`open`、`high`、`low`、`close`、`volume` |
| `fund_meta` | 基金基本信息 | `fund_code`（唯一）、`fund_name`、`fund_type`、`watchlist` |
| `fund_nav` | 基金净值 | `fund_code`、`nav_date`、`unit_nav`、`accum_nav`、`daily_return` |
| `fund_prediction` | LSTM 预测记录 | `fund_code`、`predict_date`、`direction`、`magnitude`、`prob_up` |
| `fund_model` | LSTM 模型存储 | `fund_code`、`model_data`（BLOB） |
| `news_*` | 新闻数据 | 标题、来源、URL、发布时间、内容 |
| `market_*` | 宏观市场数据 | 指标名称、日期、数值 |

写入策略：`INSERT ... ON DUPLICATE KEY UPDATE`，支持全量覆盖与增量更新。

---

## 九、技术栈汇总

| 层级 | 技术 | 用途 |
|------|------|------|
| 后端框架 | Flask | API 服务、SPA 静态资源托管 |
| 数据抓取 | akshare | 股票/基金行情数据 |
| 数据处理 | pandas | 数据清洗与计算 |
| 持久化存储 | MySQL + PyMySQL | 主数据库 |
| 本地缓存 | SQLite | 短期缓存，减少重复抓取 |
| 定时任务 | APScheduler | 数据同步、模型自动训练 |
| 深度学习 | PyTorch | LSTM 预测模型 |
| 机器学习 | scikit-learn | 特征工程、交叉验证 |
| 可解释性 | SHAP | 模型特征重要性分析 |
| 大模型 | LLM（可配置） | 行业分析、新闻分类、投资建议 |
| 前端框架 | Vue 3 (Composition API) | 页面组件 |
| 构建工具 | Vite | 开发服务器、生产打包 |
| 状态管理 | Pinia | 全局状态 |
| 图表 | ECharts | 净值曲线、K线、宏观指标图 |
| 部署 | Docker Compose | 一键部署应用 + MySQL |

---

## 十、开发与部署

### 本地开发（前后端分离）

```bash
# 后端（端口 5050）
python server.py

# 前端（端口 5173，/api 代理到 5050）
cd frontend && pnpm install && pnpm run dev
```

### 本地生产（前后端一体）

```bash
cd frontend && pnpm run build   # 构建前端到 frontend/dist
python server.py                # 后端同时提供 API 与静态资源
# 访问 http://localhost:5050
```

### Docker 部署

```bash
docker compose up -d --build
# 访问 http://localhost:5050
```

### 一键安装依赖

```bash
make install   # pip install -r requirements.txt && cd frontend && pnpm install
```

---

## 十一、扩展指引

| 扩展点 | 操作 |
|--------|------|
| 新增后端接口 | 在 `server/routes/` 下新建或修改蓝图文件，在 `server/app.py` 中注册 |
| 新增前端页面 | 在 `frontend/src/views/` 下添加组件，在 `router/index.js` 中注册路由 |
| 新增分析算法 | 在 `analysis/` 下添加模块，在 `full_report.py` 中挂接 |
| 新增业务模块 | 在 `modules/` 下按 `analyzer / repo / interfaces` 结构新建目录 |
| 替换 LLM 后端 | 修改 `analysis/llm/` 下的客户端实现，业务代码无需改动 |
| 新增定时任务 | 在 `server/app.py` 的 `scheduler` 中添加 `add_job` |
