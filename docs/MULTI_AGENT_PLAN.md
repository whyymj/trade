# FundProphet 多 Agent 协作开发方案

## 一、Agent 角色

| 角色 | 名称 | 职责 |
|------|------|------|
| 整体负责人 | @architect-agent | 整体把控、模块对接、集成测试 |
| 数据层 | @data-agent | 数据库、缓存 |
| 数据仓储 | @repo-agent | 基金数据 CRUD |
| 分析指标 | @metrics-agent | 业绩指标计算 |
| LSTM 预测 | @lstm-agent | 净值预测 |
| API 层 | @api-agent | Flask 接口 |
| 前端 | @frontend-agent | Vue3 页面 |

---

## 二、模块划分

根据架构设计，将应用拆分为 6 个独立模块：

| # | 模块 | 依赖 | 产出 |
|---|------|------|------|
| 1 | 数据层 | 无 | schema.py, cache.py |
| 2 | 数据仓储 | 数据层 | fund_repo.py, fund_fetcher.py |
| 3 | 分析指标 | 数据仓储 | fund_metrics.py |
| 4 | LSTM 预测 | 数据仓储 | fund_lstm.py |
| 5 | API 层 | 所有模块 | api.py |
| 6 | 前端 | API 层 | Vue 页面 |

## 二、开发顺序

```
数据层 → 数据仓储 → 分析指标 → LSTM → API → 前端
```

## 三、Agent 任务定义

### Agent 1: 数据层

```
开发 data/ 目录基础设施：
1. data/mysql.py - MySQL 连接
2. data/schema.py - 基金表结构 (fund_meta, fund_nav, index_data, fund_prediction, fund_model)
3. data/cache.py - 内存缓存 (SimpleCache 类，支持 TTL、线程安全)

编写 tests/test_data/ 单元测试
```

### Agent 2: 数据仓储

```
开发数据操作模块：
1. data/fund_repo.py - 基金 CRUD (add_fund, get_fund_list, get_fund_nav 等)
2. data/fund_fetcher.py - 天天基金网数据抓取
3. data/index_repo.py - 指数数据

依赖 data/mysql.py, data/cache.py

编写 tests/test_repo/ 单元测试
```

### Agent 3: 分析指标

```
开发分析模块：
1. analysis/fund_metrics.py - 指标计算
   - calculate_return (1m/3m/6m/1y)
   - calculate_annual_return
   - calculate_volatility
   - calculate_sharpe_ratio
   - calculate_max_drawdown
   - calculate_win_rate
2. analysis/fund_benchmark.py - 基准对比

依赖 data/fund_repo.py

编写 tests/test_metrics/ 单元测试
```

### Agent 4: LSTM 预测

```
开发预测模块：
1. analysis/fund_lstm.py - LSTM 预测
   - train() 训练
   - predict() 预测
   - 输出: direction, magnitude, prob_up, magnitude_5

依赖 data/fund_repo.py

编写 tests/test_lstm/ 单元测试
```

### Agent 5: API 层

```
开发 Web 接口：
1. server/routes/api.py - 全部 API 端点
   - 基金列表/净值/分析
   - 预测接口
   - LLM 分析接口
   - 数据同步接口

依赖 data/, analysis/, analysis/llm/

编写 tests/test_api/ 单元测试
```

### Agent 6: 前端

```
开发 Vue3 前端：
1. frontend/src/views/FundHome.vue - 基金列表
2. frontend/src/views/FundDetail.vue - 基金详情
3. frontend/src/views/FundPredict.vue - 预测中心
4. frontend/src/styles/cute.css - 卡通样式

依赖 server/routes/api.py

编写组件测试
```

## 四、Agent 角色定义

### @architect-agent（整体负责人）

**职责**：
1. 制定整体开发计划
2. 定义模块接口规范
3. 协调各 Agent 任务分配
4. 验收各模块产出
5. 处理模块间依赖问题
6. 最终集成测试
7. 整体代码审查

**决策权**：
- 模块接口变更审批
- 依赖冲突解决
- 优先级排序
- 交付标准制定

### @data-agent（数据层）

**职责**：数据库连接、表结构、缓存

**交付物**：
- data/mysql.py
- data/schema.py
- data/cache.py
- tests/test_data/

**验收标准**：
- [ ] MySQL 连接正常
- [ ] 5 张表创建成功
- [ ] 缓存 get/set/clear 正常
- [ ] 单元测试通过

### @repo-agent（数据仓储）

**职责**：基金/指数数据 CRUD、数据抓取

**交付物**：
- data/fund_repo.py
- data/fund_fetcher.py
- data/index_repo.py
- tests/test_repo/

**验收标准**：
- [ ] 基金 CRUD 正常
- [ ] 天天基金网数据抓取成功
- [ ] 单元测试通过

### @metrics-agent（分析指标）

**职责**：业绩指标计算、基准对比

**交付物**：
- analysis/fund_metrics.py
- analysis/fund_benchmark.py
- tests/test_metrics/

**验收标准**：
- [ ] 各项指标计算正确
- [ ] 基准对比功能正常
- [ ] 单元测试通过

### @lstm-agent（LSTM 预测）

**职责**：基金净值预测

**交付物**：
- analysis/fund_lstm.py
- tests/test_lstm/

**验收标准**：
- [ ] 模型训练成功
- [ ] 预测输出格式正确
- [ ] 单元测试通过

### @api-agent（API 层）

**职责**：Flask API 接口开发

**交付物**：
- server/routes/api.py
- tests/test_api/

**验收标准**：
- [ ] 所有 API 端点正常
- [ ] 接口文档完整
- [ ] 单元测试通过

### @frontend-agent（前端）

**职责**：Vue3 前端页面开发

**交付物**：
- frontend/src/views/*.vue
- frontend/src/styles/cute.css

**验收标准**：
- [ ] 页面展示正常
- [ ] API 调用正常
- [ ] 卡通风格符合设计

---

## 五、执行方式

### @architect-agent 的工作流程

```
1. 启动 @data-agent
   ↓ 验收通过
2. 启动 @repo-agent
   ↓ 验收通过  
3. 启动 @metrics-agent 和 @lstm-agent（并行）
   ↓ 验收通过
4. 启动 @api-agent
   ↓ 验收通过
5. 启动 @frontend-agent
   ↓ 验收通过
6. 最终集成测试
7. 代码审查与优化
```

### 具体执行命令

```bash
# 1. @architect-agent 启动数据层
@architect-agent: 开发 data/ 模块

# 2. @architect-agent 启动数据仓储
@architect-agent: 开发 data/ 仓储层

# 3. @architect-agent 启动分析指标
@architect-agent: 开发 analysis/ 指标模块

# 4. @architect-agent 启动 LSTM
@architect-agent: 开发 analysis/ LSTM 模块

# 5. @architect-agent 启动 API
@architect-agent: 开发 server/ API 模块

# 6. @architect-agent 启动前端
@architect-agent: 开发 frontend/ 前端

# 7. @architect-agent 最终集成
@architect-agent: 集成测试与优化
```

---

## 六、沟通机制

### @architect-agent 与各 Agent 的交互

```
@architect-agent
    │
    ├─→ @data-agent: "开发 schema.py, cache.py"
    │       │
    │       └─→ 产出 + 测试结果
    │
    ├─→ @repo-agent: "开发 fund_repo.py，依赖 data/"
    │       │
    │       └─→ 产出 + 测试结果
    │
    ├─→ @metrics-agent: "开发 fund_metrics.py"
    │       │
    │       └─→ 产出 + 测试结果
    │
    ├─→ @lstm-agent: "开发 fund_lstm.py"
    │       │
    │       └─→ 产出 + 测试结果
    │
    ├─→ @api-agent: "开发 API，整合所有模块"
    │       │
    │       └─→ 产出 + 测试结果
    │
    └─→ @frontend-agent: "开发前端页面"
            │
            └─→ 产出 + 测试结果
```

### 任务下发模板

```
@architect-agent 下发任务给 @xxx-agent：

## 任务：开发 [模块名称]

### 背景
[简要说明为什么开发这个模块]

### 交付物
1. [文件1]
2. [文件2]
3. [测试文件]

### 依赖
- [依赖的模块]

### 验收标准
- [ ] 标准1
- [ ] 标准2

### 参考文档
- docs/ARCHITECTURE.md
- docs/MULTI_AGENT_PLAN.md
```

## 五、测试验证

每个 Agent 完成后，运行对应测试：

```bash
# 数据层测试
pytest tests/test_data/ -v

# 数据仓储测试
pytest tests/test_repo/ -v

# 分析指标测试
pytest tests/test_metrics/ -v

# LSTM 测试
pytest tests/test_lstm/ -v

# API 测试
pytest tests/test_api/ -v
```

## 六、立即开始

我现在可以按顺序启动各个 Agent 来开发。是否现在开始？

如果开始，我将依次启动：
1. **@data-agent** - 开发数据层基础设施
2. **@repo-agent** - 开发数据仓储
3. **@metrics-agent** - 开发分析指标
4. **@lstm-agent** - 开发 LSTM 预测
5. **@api-agent** - 开发 API 接口
6. **@frontend-agent** - 开发前端页面
7. **@final-agent** - 最终集成与测试
