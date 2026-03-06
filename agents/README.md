# FundProphet 微服务架构 - Agent 目录

## 概述

本目录包含 FundProphet 项目微服务架构的多 Agent 协作设计文档和各个 Agent 的详细任务定义。

## 文档结构

```
agents/
├── README.md                    # 本文件
├── coordinator_agent.md        # 主协调 Agent
├── infra_agent.md              # 基础设施 Agent
├── devops_agent.md             # 运维部署 Agent
├── qa_agent.md                 # 质量保障 Agent
├── fund_svc_agent.md           # 基金服务 Agent
├── stock_svc_agent.md          # 股票服务 Agent
├── news_svc_agent.md           # 新闻服务 Agent
├── market_svc_agent.md         # 市场服务 Agent
├── fund_intel_agent.md         # 基金智能服务 Agent
└── llm_svc_agent.md            # 大模型服务 Agent
```

## Agent 角色体系

### 1. Coordinator Agent (主协调 Agent)

**文档**: `coordinator_agent.md`

**职责**:
- 整体架构设计
- 任务分发与进度跟踪
- 模块对接与冲突解决
- 最终验收与文档整理

**关键决策**:
- 服务边界划分
- API 接口规范
- 数据共享策略
- 优先级排序

---

### 2. Infra Agent (基础设施 Agent)

**文档**: `infra_agent.md`

**职责**:
- MySQL 连接池改造
- Redis 缓存改造
- 定时任务提取
- API 网关配置 (Traefik)
- 消息队列配置 (Redis Streams)

**交付物**:
- `shared/db.py` - 数据库连接池
- `shared/cache.py` - Redis 缓存
- `scheduler/app.py` - 定时任务
- `traefik/traefik.yml` - API 网关配置
- `shared/messaging.py` - 消息队列

---

### 3. DevOps Agent (运维部署 Agent)

**文档**: `devops_agent.md`

**职责**:
- Docker Compose 配置
- CI/CD 流程配置
- 日志收集配置 (Loki + Grafana)
- 健康检查配置
- 环境变量管理

**交付物**:
- `docker-compose.yml` - 本地开发环境
- `docker-compose.prod.yml` - 生产环境
- `.env.example` - 环境变量模板
- `.github/workflows/ci.yml` - CI/CD 流程

---

### 4. QA Agent (质量保障 Agent)

**文档**: `qa_agent.md`

**职责**:
- pytest 配置
- 测试 fixtures 定义
- 单元测试编写
- 集成测试编写
- E2E 测试编写
- 性能测试配置

**交付物**:
- `pytest.ini` - pytest 配置
- `tests/conftest.py` - 测试 fixtures
- `tests/unit/` - 单元测试
- `tests/integration/` - 集成测试
- `tests/e2e/` - E2E 测试
- `tests/performance/` - 性能测试

---

### 5. Fund Svc Agent (基金服务 Agent)

**文档**: `fund_svc_agent.md`

**职责**:
- 创建 Flask 应用
- 迁移基金数据层
- 开发基金 API 端点
- 配置基金净值同步任务

**交付物**:
- `services/fund/app.py` - Flask 应用
- `services/fund/data/` - 数据层
- `services/fund/routes/` - API 路由
- `services/fund/tests/` - 测试

---

### 6. Stock Svc Agent (股票服务 Agent)

**文档**: `stock_svc_agent.md`

**职责**:
- 创建 Flask 应用
- 迁移股票和 LSTM 相关代码
- 开发股票和 LSTM API 端点
- 配置 LSTM 自动训练任务

**交付物**:
- `services/stock/app.py` - Flask 应用
- `services/stock/analysis/` - LSTM 模块
- `services/stock/routes/` - API 路由
- `services/stock/tests/` - 测试

---

### 7. News Svc Agent (新闻服务 Agent)

**文档**: `news_svc_agent.md`

**职责**:
- 创建 Flask 应用
- 迁移新闻爬虫和仓储代码
- 开发新闻相关 API 端点
- 新闻爬取完成后发布事件

**交付物**:
- `services/news/app.py` - Flask 应用
- `services/news/data/` - 数据层
- `services/news/routes/` - API 路由
- `services/news/tests/` - 测试

---

### 8. Market Svc Agent (市场服务 Agent)

**文档**: `market_svc_agent.md`

**职责**:
- 创建 Flask 应用
- 迁移市场数据爬虫和仓储代码
- 开发市场相关 API 端点
- **特点**: 零跨服务依赖，最容易独立部署

**交付物**:
- `services/market/app.py` - Flask 应用
- `services/market/data/` - 数据层
- `services/market/routes/` - API 路由
- `services/market/tests/` - 测试

---

### 9. Fund-Intel Agent (基金智能服务 Agent)

**文档**: `fund_intel_agent.md`

**职责**:
- 创建 Flask 应用
- 迁移业务模块代码
- 调用 Fund、News、LLM 服务
- 开发基金智能相关 API 端点
- 订阅新闻爬取事件

**交付物**:
- `services/fund-intel/app.py` - Flask 应用
- `services/fund-intel/clients/` - 服务客户端
- `services/fund-intel/modules/` - 业务模块
- `services/fund-intel/routes/` - API 路由
- `services/fund-intel/tests/` - 测试

---

### 10. LLM Svc Agent (大模型服务 Agent)

**文档**: `llm_svc_agent.md`

**职责**:
- 创建无状态 Flask 应用
- 迁移 DeepSeek/MiniMax 客户端
- 开发 LLM API 端点
- API Key 隔离
- 速率限制配置

**交付物**:
- `services/llm/app.py` - Flask 应用
- `services/llm/llm/` - LLM 客户端
- `services/llm/routes/` - API 路由
- `services/llm/tests/` - 测试

---

## 快速开始

### 1. 阅读架构文档

```bash
# 查看完整的架构设计
cat docs/AGENT_ARCHITECTURE.md

# 查看微服务拆分方案
cat docs/MICROSERVICES.md
```

### 2. 选择 Agent 角色

根据你的职责选择对应的 Agent 文档：

- 如果是**主协调者**，阅读 `coordinator_agent.md`
- 如果是**基础设施开发者**，阅读 `infra_agent.md`
- 如果是**运维工程师**，阅读 `devops_agent.md`
- 如果是**测试工程师**，阅读 `qa_agent.md`
- 如果是**基金服务开发者**，阅读 `fund_svc_agent.md`
- 如果是**股票服务开发者**，阅读 `stock_svc_agent.md`
- 如果是**新闻服务开发者**，阅读 `news_svc_agent.md`
- 如果是**市场服务开发者**，阅读 `market_svc_agent.md`
- 如果是**基金智能服务开发者**，阅读 `fund_intel_agent.md`
- 如果是**LLM 服务开发者**，阅读 `llm_svc_agent.md`

### 3. 开始任务

按照 Agent 文档中的任务清单开始执行：

```bash
# 例如：Infra Agent 开始任务
cat agents/infra_agent.md
```

---

## 协作流程

### 三层协作架构

```
第1层：协调层 (Coordination)
  Coordinator Agent - 整体架构、任务分发

第2层：支撑层 (Support)
  Infra Agent + QA Agent + DevOps Agent - 基础设施、质量、部署

第3层：业务层 (Business Services)
  Fund Svc + Stock Svc + News Svc + Market Svc + Fund-Intel Svc + LLM Svc
```

### 三阶段开发流程

**阶段1：基础设施 (1-2周)**
- Coordinator: 制定微服务拆分方案
- Infra Agent: Redis + Traefik + MySQL连接池改造
- DevOps Agent: Docker Compose 配置
- QA Agent: 定义测试框架

**阶段2：服务拆分 (3-10周)**
- Phase 2.1: LLM Service (3-4周)
- Phase 2.2: News + Market Service (5-6周)
- Phase 2.3: Fund-Intel Service (7-8周)
- Phase 2.4: Stock + Fund Service (9-12周)

**阶段3：集成部署 (13-14周)**
- DevOps Agent: 生产环境部署
- QA Agent: 集成测试 + E2E 测试
- Coordinator: 最终验收 + 文档整理

---

## 服务端口分配

| 服务 | 端口 |
|------|------|
| Stock Service | 8001 |
| Fund Service | 8002 |
| News Service | 8003 |
| Market Service | 8004 |
| Fund-Intel Service | 8005 |
| LLM Service | 8006 |
| API Gateway (Traefik) | 80/8080 |
| Redis | 6379 |
| MySQL | 3306 |

---

## 目录结构

```
trade/
├── services/                    # 6个微服务
│   ├── stock/
│   ├── fund/
│   ├── news/
│   ├── market/
│   ├── fund-intel/
│   └── llm/
├── shared/                      # 共享代码
│   ├── db.py                    # 数据库连接池
│   ├── cache.py                 # Redis 缓存
│   ├── response.py              # 统一响应格式
│   └── messaging.py             # 消息队列封装
├── scheduler/                   # 独立定时任务
│   └── app.py
├── traefik/                     # API 网关配置
│   └── traefik.yml
├── agents/                      # Agent 文档目录
│   ├── README.md                # 本文件
│   ├── coordinator_agent.md
│   ├── infra_agent.md
│   ├── devops_agent.md
│   ├── qa_agent.md
│   ├── fund_svc_agent.md
│   └── llm_svc_agent.md
├── docs/                        # 文档
│   ├── AGENT_ARCHITECTURE.md    # 完整架构文档
│   ├── MICROSERVICES.md         # 微服务拆分方案
│   └── ...
├── tests/                       # 测试用例
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docker-compose.yml           # Docker Compose 配置
```

---

## 下一步

1. **阅读完整架构文档**: `docs/AGENT_ARCHITECTURE.md`
2. **选择你的 Agent 角色**
3. **阅读对应的 Agent 文档**
4. **按照任务清单开始执行**

---

## 联系与支持

如有问题，请联系 Coordinator Agent 或查阅以下文档：

- `docs/AGENT_ARCHITECTURE.md` - 完整架构设计
- `docs/MICROSERVICES.md` - 微服务拆分方案
- `docs/COMPLETE_ARCHITECTURE.md` - 详细架构文档

---

**准备好开始了吗？选择你的 Agent 角色，开始行动！**