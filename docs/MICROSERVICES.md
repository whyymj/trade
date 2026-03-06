# 微服务架构拆分方案

> 基于当前代码实际状态分析，给出可落地的拆分建议。

---

## 一、现状分析

### 当前架构问题

| 问题 | 描述 |
|------|------|
| 单体 Flask 应用 | 7个蓝图全部注册在同一进程，任何模块崩溃影响全局 |
| `api.py` 过重 | 1184行，混合了股票、基金、指数、LSTM、同步等多种职责 |
| 进程内缓存 | `data/cache.py` 是内存单例，多实例部署时缓存不共享 |
| 无连接池 | `data/mysql.py` 每次请求新建连接，高并发下性能差 |
| 定时任务耦合 | APScheduler 跑在 Flask 进程内，扩容时任务会重复执行 |
| LLM 密钥暴露 | 所有模块都能读取 `DEEPSEEK_API_KEY`，无隔离 |

### 模块依赖关系

```
investment_advice
  └─ fund_industry + news_classification + fund_news_association
       └─ data.fund_repo + data.news.NewsRepo + analysis.llm
            └─ data.mysql + config.yaml
```

---

## 二、目标架构：6个微服务

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 Vue3 (Nginx)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                  API 网关 (Traefik)                          │
│         路由规则：/api/stock/* /api/fund/* /api/news/*       │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────────────┐
│Stock │ │Fund  │ │News  │ │Market│ │  Fund Intelligence   │
│ Svc  │ │ Svc  │ │ Svc  │ │ Svc  │ │  (fund-intel-svc)    │
│:8001 │ │:8002 │ │:8003 │ │:8004 │ │       :8005          │
└──────┘ └──────┘ └──────┘ └──────┘ └──────────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │    LLM Service     │
                                    │       :8006        │
                                    └────────────────────┘
         所有服务共享：Redis（缓存+消息队列）+ MySQL
```

---

## 三、服务职责划分

### Service 1：Stock Service（股票分析服务）

**端口：** 8001

**拥有的表：**
- `stock_meta`, `stock_daily`
- `lstm_training_run`, `lstm_current_version_per_symbol`, `lstm_prediction_log`
- `lstm_accuracy_record`, `lstm_training_failure`, `lstm_model_version`

**迁移的代码：**
- `data/stock_repo.py`
- `data/lstm_repo.py`
- `analysis/lstm_model.py`, `analysis/lstm_*.py`
- `analysis/technical.py`, `analysis/arima_model.py`
- `analysis/ensemble_models.py`, `analysis/factor_library.py`
- `analysis/full_report.py`
- `server/routes/api.py` 中股票相关路由

**对外接口：**
```
GET  /api/stock/list
GET  /api/stock/data?symbol=
POST /api/stock/add
POST /api/stock/sync
GET  /api/stock/analyze?symbol=
POST /api/lstm/train
GET  /api/lstm/predict?symbol=
```

**定时任务：** LSTM 自动训练（04:00）

---

### Service 2：Fund Service（基金分析服务）

**端口：** 8002

**拥有的表：**
- `fund_meta`, `fund_nav`, `fund_prediction`, `fund_model`, `index_data`

**迁移的代码：**
- `data/fund_repo.py`, `data/fund_fetcher.py`, `data/fund_holdings.py`
- `data/index_repo.py`
- `analysis/fund_lstm.py`, `analysis/fund_metrics.py`
- `analysis/fund_benchmark.py`, `analysis/fund_analyzer.py`
- `analysis/fund_data.py`, `analysis/full_fund_report.py`
- `analysis/time_domain.py`, `analysis/frequency_domain.py`
- `server/routes/api.py` 中基金相关路由

**对外接口：**
```
GET  /api/fund/list
GET  /api/fund/:code
POST /api/fund/add
GET  /api/fund/:code/nav
GET  /api/fund/:code/holdings
GET  /api/fund/:code/predict
POST /api/fund/sync
GET  /api/index/list
```

**定时任务：** 基金净值同步（03:00）

---

### Service 3：News Service（新闻服务）

**端口：** 8003

**拥有的表：**
- `news_data`, `news_analysis`

**迁移的代码：**
- `data/news/crawler.py`, `data/news/repo.py`, `data/news/interfaces.py`
- `server/routes/news.py`

**对外接口：**
```
GET  /api/news/list
GET  /api/news/:id
POST /api/news/crawl
GET  /api/news/analysis
GET  /api/news/analysis/latest
```

**事件发布：** 爬取完成后向 Redis Stream `news:crawled` 发布消息

---

### Service 4：Market Service（市场数据服务）

**端口：** 8004

**拥有的表：**
- `macro_data`, `money_flow`, `market_sentiment`, `global_macro`

**迁移的代码：**
- `data/market/crawler.py`, `data/market/repo.py`, `data/market/interfaces.py`
- `server/routes/market.py`

**对外接口：**
```
GET  /api/market/macro
GET  /api/market/money-flow
GET  /api/market/sentiment
GET  /api/market/global
POST /api/market/sync
```

**特点：** 零跨服务依赖，最容易独立部署

---

### Service 5：Fund Intelligence Service（基金智能服务）

**端口：** 8005

**拥有的表：**
- `fund_industry`（由 `modules/fund_industry/schema.py` 创建）
- `news_industry_classification`
- `fund_news_association`

**迁移的代码：**
- `modules/fund_industry/`
- `modules/news_classification/`
- `modules/fund_news_association/`
- `modules/investment_advice/`
- `server/routes/fund_industry.py`
- `server/routes/news_classification.py`
- `server/routes/fund_news_association.py`
- `server/routes/investment_advice.py`

**对外接口：**
```
POST /api/fund-industry/analyze/:code
GET  /api/fund-industry/:code
POST /api/news-classification/classify
GET  /api/news-classification/stats
GET  /api/fund-news/match/:code
GET  /api/investment-advice/:code
```

**跨服务调用：**
- 调用 Fund Service 获取基金信息
- 调用 News Service 获取新闻数据
- 调用 LLM Service 进行智能分析

**事件订阅：** 监听 Redis Stream `news:crawled`，自动触发新闻分类

---

### Service 6：LLM Service（大模型服务）

**端口：** 8006

**无数据库表**（无状态服务）

**迁移的代码：**
- `analysis/llm/deepseek.py`
- `analysis/llm/minimax.py`
- `analysis/llm/news_analyzer.py`
- `analysis/llm/interfaces.py`

**对外接口：**
```
POST /api/llm/chat
POST /api/llm/analyze-news
POST /api/llm/classify-industry
POST /api/llm/investment-advice
```

**特点：**
- 唯一持有 `DEEPSEEK_API_KEY` 的服务，密钥隔离
- 可独立扩容（LLM 调用是性能瓶颈）
- 内置 Redis 缓存（相同 prompt 24小时内不重复调用）

---

## 四、数据库策略

### 阶段一：共享数据库（推荐起步方式）

保持单个 MySQL 实例，但每个服务只访问自己的表：

```
trade_cache（共享 MySQL）
├── stock_*     → 只有 Stock Service 可写
├── lstm_*      → 只有 Stock Service 可写
├── fund_*      → 只有 Fund Service 可写
├── index_*     → 只有 Fund Service 可写
├── news_*      → News Service 可写，Fund Intel Service 可读
├── market_*    → 只有 Market Service 可写
└── fund_industry, news_industry_classification, fund_news_association
                → 只有 Fund Intel Service 可写
```

**优点：** 无数据迁移风险，跨服务查询仍可用，运维简单

### 阶段二：数据库分离（可选，流量增长后）

每个服务独立数据库，跨服务数据通过 API 获取：

```
stock-db:3307   → Stock Service 专用
fund-db:3308    → Fund Service 专用
news-db:3309    → News + Fund Intel Service 共用
market-db:3310  → Market Service 专用
```

---

## 五、通信模式

### 同步调用（REST）

适用于用户请求链路：

```
前端 → API 网关 → 目标服务
Fund Intel Svc → Fund Svc（获取基金信息，Redis 缓存1小时）
Fund Intel Svc → LLM Svc（LLM 分析，Redis 缓存24小时）
```

### 异步消息（Redis Streams）

适用于后台任务：

```
news:crawled    → News Svc 发布，Fund Intel Svc 订阅（触发自动分类）
fund:sync       → 定时触发，Fund Svc 消费
lstm:train      → 定时触发，Stock Svc 消费
```

**选择 Redis Streams 而非 RabbitMQ/Kafka 的原因：**
- Redis 已经是缓存依赖，不增加新组件
- 支持消费者组，保证 at-least-once 投递
- 当前规模不需要 Kafka 的吞吐量

---

## 六、基础设施变更

### 需要新增的组件

| 组件 | 用途 | 替换什么 |
|------|------|---------|
| Redis | 分布式缓存 + 消息队列 | `data/cache.py` 内存缓存 |
| Traefik | API 网关 + 路由 | Flask 直接暴露 |
| Nginx | 前端静态资源 | Flask 托管 `frontend/dist` |

### MySQL 连接池改造

当前 `data/mysql.py` 每次请求新建连接，需改为连接池：

```python
# 当前：每次新建连接
with get_connection() as conn:
    ...

# 改造后：使用 SQLAlchemy 连接池（接口兼容）
from sqlalchemy import create_engine
engine = create_engine(url, pool_size=10, max_overflow=20)
```

接口保持 `get_connection()` / `fetch_all()` / `execute()` 不变，所有 repo 文件零改动。

### 缓存改造

```python
# 当前：进程内单例
from data.cache import get_cache
cache = get_cache()
cache.set("fund_list", data, ttl=1800)

# 改造后：Redis（接口兼容）
import redis
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
r.setex("fund_list", 1800, json.dumps(data))
```

---

## 七、迁移路线图

### Phase 1：基础设施改造（1-2周）

**目标：** 不改变服务边界，为拆分做准备

1. `data/mysql.py` 改为 SQLAlchemy 连接池
2. `data/cache.py` 改为 Redis 客户端（保持接口兼容）
3. APScheduler 从 `server/app.py` 提取为独立 `scheduler.py` 进程
4. 所有蓝图添加 `/health` 端点
5. 添加 Redis 到 `docker-compose.yml`

**风险：低**（接口兼容改造）

---

### Phase 2：提取 LLM Service（第3-4周）

**目标：** 解耦最多被共享的依赖

1. 新建 `services/llm/` 目录，创建独立 Flask 应用
2. 迁移 `analysis/llm/` 到新服务
3. 暴露 REST 接口（`/api/llm/chat` 等）
4. 创建 `analysis/llm/http_client.py`，其他模块改用 HTTP 调用
5. 更新 `modules/news_classification/analyzer.py`、`modules/investment_advice/__init__.py`

**风险：中**（LLM 调用已有 fallback，降级安全）

---

### Phase 3：提取 News + Market Service（第5-6周）

**目标：** 提取两个最独立的服务

**News Service：**
1. 新建 `services/news/`
2. 迁移 `data/news/`、`server/routes/news.py`
3. 爬取完成后发布 `news:crawled` 事件

**Market Service：**
1. 新建 `services/market/`
2. 迁移 `data/market/`、`server/routes/market.py`

**风险：低**（Market Service 零跨服务依赖）

---

### Phase 4：提取 Fund Intelligence Service（第7-8周）

**目标：** 提取最复杂的跨切面模块

1. 新建 `services/fund-intel/`
2. 迁移 `modules/` 下全部4个子模块
3. 迁移对应4个路由文件
4. 将 `fund_repo.get_fund_info()` 改为调用 Fund Service HTTP API
5. 将 `NewsRepo` 调用改为调用 News Service HTTP API
6. 订阅 `news:crawled` 事件触发自动分类

**风险：中高**（跨服务依赖最多，需要充分测试）

---

### Phase 5：拆分 Stock + Fund Service（第9-12周）

**目标：** 拆解 1184 行的 `api.py` 单体

**Stock Service：**
1. 新建 `services/stock/`
2. 迁移 `data/stock_repo.py`、`data/lstm_repo.py`
3. 迁移 `api.py` 中股票/LSTM 相关路由
4. 迁移 `analysis/lstm_*.py`、`analysis/technical.py` 等
5. 将进程内训练锁改为 Redis 分布式锁

**Fund Service：**
1. 新建 `services/fund/`
2. 迁移 `data/fund_repo.py`、`data/fund_fetcher.py` 等
3. 迁移 `api.py` 中基金/指数/同步相关路由
4. 修复 `analysis/fund_lstm.py` 直接导入 `data.mysql` 的问题

**风险：高**（`api.py` 路由边界需仔细梳理）

---

## 八、目录结构（迁移后）

```
trade/
├── services/                    # 各微服务
│   ├── stock/                   # Stock Service
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── data/                # stock_repo, lstm_repo
│   │   ├── analysis/            # lstm_*.py, technical.py
│   │   └── requirements.txt     # 精简依赖（含 torch）
│   ├── fund/                    # Fund Service
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── data/                # fund_repo, fund_fetcher
│   │   ├── analysis/            # fund_lstm.py, fund_metrics.py
│   │   └── requirements.txt     # 精简依赖（含 torch）
│   ├── news/                    # News Service
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── data/                # news/crawler, news/repo
│   │   └── requirements.txt     # 无 torch，轻量
│   ├── market/                  # Market Service
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── data/                # market/crawler, market/repo
│   │   └── requirements.txt     # 最轻量
│   ├── fund-intel/              # Fund Intelligence Service
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── modules/             # fund_industry, news_classification 等
│   │   └── requirements.txt
│   └── llm/                     # LLM Service
│       ├── app.py
│       ├── routes/
│       ├── llm/                 # deepseek.py, minimax.py
│       └── requirements.txt     # 最轻量，无 torch
│
├── shared/                      # 共享代码（各服务 copy 或 pip 安装）
│   ├── db.py                    # SQLAlchemy 连接池封装
│   ├── cache.py                 # Redis 客户端封装
│   └── response.py              # success_response / error_response
│
├── frontend/                    # 前端（独立 Nginx 容器）
├── docker-compose.yml           # 本地开发：所有服务 + Redis + MySQL
└── docker-compose.prod.yml      # 生产：可选 K8s 替代
```

---

## 九、Docker Compose 配置示意

```yaml
version: "3.9"
services:
  traefik:
    image: traefik:v3
    ports: ["80:80", "8080:8080"]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  mysql:
    image: mysql:8
    environment:
      MYSQL_DATABASE: trade_cache
      MYSQL_ROOT_PASSWORD: ${MYSQL_PASSWORD}

  stock-service:
    build: ./services/stock
    labels:
      - "traefik.http.routers.stock.rule=PathPrefix(`/api/stock`) || PathPrefix(`/api/lstm`)"
    environment:
      REDIS_URL: redis://redis:6379
      MYSQL_HOST: mysql

  fund-service:
    build: ./services/fund
    labels:
      - "traefik.http.routers.fund.rule=PathPrefix(`/api/fund`) || PathPrefix(`/api/index`)"
    environment:
      REDIS_URL: redis://redis:6379
      MYSQL_HOST: mysql

  news-service:
    build: ./services/news
    labels:
      - "traefik.http.routers.news.rule=PathPrefix(`/api/news`)"

  market-service:
    build: ./services/market
    labels:
      - "traefik.http.routers.market.rule=PathPrefix(`/api/market`)"

  fund-intel-service:
    build: ./services/fund-intel
    labels:
      - "traefik.http.routers.fi.rule=PathPrefix(`/api/fund-industry`) || PathPrefix(`/api/investment-advice`)"
    environment:
      FUND_SERVICE_URL: http://fund-service:8002
      NEWS_SERVICE_URL: http://news-service:8003
      LLM_SERVICE_URL: http://llm-service:8006

  llm-service:
    build: ./services/llm
    labels:
      - "traefik.http.routers.llm.rule=PathPrefix(`/api/llm`)"
    environment:
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY}  # 只有此服务持有密钥

  frontend:
    image: nginx:alpine
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
    labels:
      - "traefik.http.routers.frontend.rule=PathPrefix(`/`)"
```

---

## 十、关键风险与应对

| 风险 | 描述 | 应对方案 |
|------|------|---------|
| 训练锁丢失 | `_training_symbols_in_progress` 是进程内字典 | 改为 Redis `SETNX` 分布式锁，TTL=2小时 |
| 定时任务重复 | APScheduler 在多实例时重复执行 | Scheduler 作为单独容器，不随 API 扩容 |
| `fund_lstm.py` 直接访问 MySQL | 绕过 repo 层，迁移时会断 | Phase 5 前先重构为使用 `fund_repo` |
| 跨服务事务 | 无法用 DB 事务保证原子性 | 利用幂等写入（`ON DUPLICATE KEY UPDATE`）+ 重试 |
| 前端路由 | 当前由 Flask 提供 SPA fallback | 改为 Nginx `try_files $uri /index.html` |

---

## 十一、推荐优先级

**如果只做一件事：** Phase 1（基础设施改造）—— 连接池 + Redis 缓存，收益最大，风险最低。

**如果要真正拆分：** 按 Phase 2 → 3 → 4 → 5 顺序，每个 Phase 完成后单独上线验证，不要一次性全部迁移。

**不建议做的：** 一开始就上 Kafka、Istio、Consul —— 当前规模完全不需要，Redis Streams + Traefik + Docker DNS 足够。
