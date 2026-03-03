# FundProphet 基金分析系统 - 架构设计文档

## 一、系统概述

### 1.1 项目定位

轻量级本地基金分析系统，支持基金净值查询、业绩分析、LSTM 净值预测、大模型智能分析。

### 1.2 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端 | Python Flask | 单进程开发服务器 |
| 前端 | Vue3 + Vite + ECharts | 卡通风格界面 |
| 数据库 | MySQL | 基金数据存储 |
| 机器学习 | PyTorch | 简化版 LSTM 预测 |
| 大模型 | MiniMax M2.5 | 智能分析（已配置） |
| 缓存 | 内存缓存 | 无 Redis 依赖 |

### 1.3 运行方式

```bash
# 启动后端
python server.py

# 启动前端
cd frontend && pnpm run dev
```

---

## 二、目录结构

```
trade/
├── config.yaml                 # 配置文件
├── server.py                   # 启动入口
├── requirements.txt            # 依赖
│
├── data/                       # 数据层
│   ├── __init__.py
│   ├── mysql.py               # MySQL 连接
│   ├── schema.py              # 表结构
│   ├── cache.py               # 内存缓存
│   ├── fund_repo.py           # 基金数据仓储
│   ├── fund_fetcher.py        # 基金数据抓取
│   └── index_repo.py          # 指数数据仓储
│
├── analysis/                   # 分析模块
│   ├── __init__.py
│   ├── fund_data.py           # 基金数据处理
│   ├── fund_metrics.py        # 基金指标计算
│   ├── fund_benchmark.py      # 基准对比
│   ├── fund_lstm.py           # LSTM 基金预测
│   ├── fund_factor.py         # 多因子筛选
│   ├── lstm_model.py          # LSTM 框架
│   ├── factor_library.py      # 因子库
│   ├── technical.py           # 技术指标
│   ├── llm/                   # ★ 大模型分析
│   │   ├── __init__.py
│   │   └── client.py          # LLM 客户端
│   └── utils.py               # 分析工具
│
├── server/                     # Web 层
│   ├── __init__.py
│   ├── app.py                 # Flask 应用
│   ├── utils.py               # 通用工具
│   ├── logging_config.py      # 日志
│   ├── scheduler.py           # 定时任务
│   └── routes/
│       ├── __init__.py
│       └── api.py             # API 路由
│
├── frontend/                   # 前端
│   ├── src/
│   │   ├── api/               # API 请求
│   │   ├── components/        # 组件
│   │   ├── views/             # 页面
│   │   ├── stores/            # 状态管理
│   │   ├── styles/            # 样式（卡通风格）
│   │   ├── router/            # 路由
│   │   ├── App.vue
│   │   └── main.js
│   └── package.json
│
└── docs/                      # 文档
    └── ARCHITECTURE.md        # 本文档
```

---

## 三、模块设计

### 3.1 数据层 `data/`

#### 3.1.1 职责

- 数据库连接管理
- 基金/指数数据 CRUD
- 数据抓取（天天基金网）
- 内存缓存

#### 3.1.2 核心文件

| 文件 | 职责 |
|------|------|
| `mysql.py` | MySQL 连接池 |
| `schema.py` | 表结构定义 |
| `cache.py` | 内存缓存 |
| `fund_repo.py` | 基金数据操作 |
| `fund_fetcher.py` | 数据抓取 |
| `index_repo.py` | 指数数据操作 |

#### 3.1.3 数据表结构

```sql
-- 基金基本信息
CREATE TABLE fund_meta (
    fund_code VARCHAR(10) PRIMARY KEY,
    fund_name VARCHAR(128),
    fund_type VARCHAR(32),
    manager VARCHAR(128),
    establishment_date DATE,
    fund_scale DECIMAL(20,2),
    watchlist TINYINT(1) DEFAULT 0,
    ...
);

-- 基金净值
CREATE TABLE fund_nav (
    fund_code VARCHAR(10),
    nav_date DATE,
    unit_nav DECIMAL(10,4),
    accum_nav DECIMAL(10,4),
    daily_return DECIMAL(10,4),
    UNIQUE(fund_code, nav_date),
    INDEX idx_code_date(fund_code, nav_date)
);

-- 指数数据
CREATE TABLE index_data (
    index_code VARCHAR(20),
    trade_date DATE,
    close_price DECIMAL(10,2),
    daily_return DECIMAL(10,4),
    UNIQUE(index_code, trade_date)
);

-- 预测记录
CREATE TABLE fund_prediction (
    fund_code VARCHAR(10),
    predict_date DATE,
    direction TINYINT,
    magnitude DECIMAL(12,6),
    prob_up DECIMAL(8,4),
    UNIQUE(fund_code, predict_date)
);

-- 模型存储
CREATE TABLE fund_model (
    fund_code VARCHAR(10) PRIMARY KEY,
    model_data LONGBLOB
);
```

### 3.2 分析层 `analysis/`

#### 3.2.1 职责

- 基金业绩指标计算
- 基准对比分析
- LSTM 净值预测
- 多因子筛选
- 大模型智能分析

#### 3.2.2 模块关系

```
┌─────────────────────────────────────────────────────────────┐
│                     分析层模块关系图                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  data/fund_repo ──────▶ fund_metrics ──────▶ fund_benchmark │
│        │                                              │      │
│        │                   │                          │      │
│        ▼                   ▼                          ▼      │
│  fund_data ──────▶ technical ──────▶ factor_library ──┘      │
│        │                                              │      │
│        │                   │                          │      │
│        ▼                   ▼                          ▼      │
│  fund_lstm ◀───────────────llm/client.py                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.3 核心模块

| 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `fund_data.py` | 原始净值 DataFrame | 处理后 DataFrame | 数据清洗 |
| `fund_metrics.py` | 净值序列 | 指标字典 | 收益率/夏普/回撤 |
| `fund_benchmark.py` | 基金净值 + 指数 | 对比报告 | 基准对比 |
| `fund_lstm.py` | 净值序列 | 预测结果 | LSTM 预测 |
| `fund_factor.py` | 多基金数据 | 筛选结果 | 多因子筛选 |
| `technical.py` | 净值序列 | 技术指标 | MACD/RSI |
| `factor_library.py` | 净值 DataFrame | 因子 DataFrame | 因子计算 |
| `llm/client.py` | 提示词 | 分析文本 | MiniMax API |

### 3.3 Web 层 `server/`

#### 3.3.1 职责

- Flask 应用创建
- API 路由定义
- 请求响应处理
- 定时任务调度

#### 3.3.2 核心文件

| 文件 | 职责 |
|------|------|
| `app.py` | Flask 应用工厂 |
| `routes/api.py` | API 路由定义 |
| `utils.py` | 通用工具函数 |
| `scheduler.py` | 定时任务 |

### 3.4 前端 `frontend/`

#### 3.4.1 页面结构

| 页面 | 路由 | 说明 |
|------|------|------|
| FundHome | `/` | 基金列表首页 |
| FundDetail | `/fund/:code` | 基金详情 |
| FundPredict | `/predict` | 预测中心 |
| Settings | `/settings` | 设置 |

#### 3.4.2 卡通风格

```css
:root {
  --primary: #6C9BFF;      /* 可爱蓝 */
  --secondary: #FF9ECD;    /* 糖果粉 */
  --bg-primary: #FFF9F0;   /* 奶白 */
  --success: #7ED957;
  --danger: #FF6B6B;
}
```

---

## 四、API 接口设计

### 4.1 接口列表

#### 基金列表

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/fund/list` | 基金列表 | 30分钟 |
| GET | `/api/fund/watchlist` | 关注列表 | 5分钟 |
| POST | `/api/fund/add` | 添加基金 | - |
| DELETE | `/api/fund/<code>` | 删除基金 | - |
| PUT | `/api/fund/<code>/watch` | 关注/取消 | - |

#### 基金净值

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/fund/nav/<code>` | 净值历史 | 30分钟 |
| GET | `/api/fund/nav/latest/<code>` | 最新净值 | 5分钟 |

#### 基金分析

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/fund/indicators/<code>` | 业绩指标 | 1小时 |
| GET | `/api/fund/benchmark/<code>` | 基准对比 | 1小时 |

#### 基金预测

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| POST | `/api/fund/predict` | 预测净值 | 5分钟 |
| GET | `/api/fund/prediction/<code>` | 预测结果 | 5分钟 |
| POST | `/api/fund/train` | 训练模型 | - |

#### 大模型分析

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/fund/llm-status` | LLM 状态 | - |
| GET | `/api/fund/analysis/profile/<code>` | 概况分析 | 24小时 |
| GET | `/api/fund/analysis/performance/<code>` | 业绩归因 | 24小时 |
| GET | `/api/fund/analysis/risk/<code>` | 风险评估 | 24小时 |
| GET | `/api/fund/advice/<code>` | 投资建议 | 24小时 |
| GET | `/api/fund/report/<code>` | 完整报告 | 24小时 |

#### 指数数据

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/index/list` | 指数列表 | 1天 |
| GET | `/api/index/data/<code>` | 指数数据 | 1小时 |

#### 数据同步

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/sync/funds` | 同步基金数据 |
| POST | `/api/sync/index` | 同步指数数据 |

### 4.2 响应格式

```json
// 成功
{
  "code": 0,
  "data": { ... }
}

// 错误
{
  "code": 404,
  "message": "基金不存在"
}
```

---

## 五、模块间通信

### 5.1 调用关系

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   前端       │────▶│   Flask      │────▶│   分析层     │
│   Vue3       │◀────│   API        │◀────│   analysis   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   数据层     │◀───▶│   LLM        │
                     │   data       │     │   MiniMax    │
                     └──────────────┘     └──────────────┘
```

### 5.2 数据结构

#### 基金信息

```python
{
    "fund_code": "001234",
    "fund_name": "某某混合基金",
    "fund_type": "混合型",
    "manager": "张三",
    "fund_scale": 50.0,
    "watchlist": True
}
```

#### 净值数据

```python
{
    "fund_code": "001234",
    "nav_date": "2024-01-15",
    "unit_nav": 1.2345,
    "accum_nav": 2.3456,
    "daily_return": 1.25
}
```

#### 业绩指标

```python
{
    "return_1m": 3.25,
    "return_3m": 8.50,
    "return_6m": 12.30,
    "return_1y": 15.20,
    "annual_return": 15.20,
    "volatility": 18.5,
    "sharpe_ratio": 0.82,
    "max_drawdown": -12.3,
    "win_rate": 55.0
}
```

#### 预测结果

```python
{
    "fund_code": "001234",
    "direction": 1,
    "direction_label": "涨",
    "magnitude": 0.025,
    "prob_up": 0.68,
    "magnitude_5": [0.01, 0.02, -0.01, 0.015, 0.005],
    "predict_date": "2024-01-15"
}
```

#### LLM 分析结果

```python
{
    "fund_profile": "该基金为混合型基金，投资风格...",
    "performance_attribution": "业绩主要来源于...",
    "risk_assessment": "风险等级中等，波动率...",
    "investment_advice": "建议适度配置..."
}
```

---

## 六、大模型集成

### 6.1 LLM 客户端

已实现 MiniMax M2.5 集成：

```python
from analysis.llm import get_client

client = get_client()
result = client.chat([{"role": "user", "content": "分析基金001234"}])
```

### 6.2 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MINIMAX_API_KEY` | MiniMax API Key | 必填 |
| `MINIMAX_MODEL` | 模型名称 | MiniMax-M2.5 |
| `MINIMAX_BASE_URL` | API 地址 | https://api.minimax.chat/v1 |

### 6.3 分析场景

| 场景 | 输入数据 | 输出 |
|------|----------|------|
| 基金概况 | 基金类型/经理/规模 | 投资风格分析 |
| 业绩归因 | 收益率/波动率/基准 | 收益来源分析 |
| 风险评估 | 回撤/胜率/波动率 | 风险等级评估 |
| 投资建议 | 指标+预测数据 | 买入/持有建议 |
| 智能报告 | 全部数据 | 完整分析报告 |

---

## 七、性能优化

### 7.1 缓存策略

| 数据类型 | TTL | 说明 |
|----------|-----|------|
| 基金列表 | 30分钟 | 变化较少 |
| 净值数据 | 30分钟 | 日更 |
| 最新净值 | 5分钟 | 高频访问 |
| 业绩指标 | 1小时 | 计算复杂 |
| 预测结果 | 5分钟 | 实时性要求 |
| LLM 分析 | 24小时 | 分析结果稳定 |

### 7.2 查询限制

```python
MAX_NAV_RECORDS = 5000    # 净值最多查5000条
MAX_FUND_LIST = 100       # 列表最多100条
MAX_PREDICT_BATCH = 10    # 批量预测最多10个
```

### 7.3 LSTM 简化

```python
TRAIN_CONFIG = {
    'epochs': 10,          # 开发环境
    'hidden_size': 32,
    'batch_size': 64,
    'seq_length': 20,
}
```

---

## 八、实施步骤

### Phase 1：数据库重构（1天）

- [ ] 重写 `data/schema.py`
- [ ] 执行 SQL 创建新表
- [ ] 删除旧股票表

### Phase 2：数据层（2天）

- [ ] 实现 `data/cache.py`
- [ ] 实现 `data/fund_repo.py`
- [ ] 实现 `data/fund_fetcher.py`

### Phase 3：分析模块（2天）

- [ ] 完善 `analysis/fund_metrics.py`
- [ ] 实现 `analysis/fund_lstm.py`
- [ ] 实现 `analysis/fund_benchmark.py`

### Phase 4：API（2天）

- [ ] 重写 `server/routes/api.py`
- [ ] 实现 LLM 分析接口
- [ ] 添加缓存

### Phase 5：LLM 集成（1天）

- [ ] 完成 `analysis/llm/` 模块
- [ ] 测试 MiniMax 连接

### Phase 6：前端（4天）

- [ ] 新建 Vue 页面
- [ ] 实现卡通风格
- [ ] 前后端联调

### Phase 7：测试优化（1天）

- [ ] 功能测试
- [ ] 性能优化
- [ ] Bug 修复

---

## 九、工作量估算

| 阶段 | 天数 |
|------|------|
| 数据库重构 | 1 |
| 数据层 | 2 |
| 分析模块 | 2 |
| API | 2 |
| LLM 集成 | 1 |
| 前端 | 4 |
| 测试优化 | 1 |
| **总计** | **13天** |

---

## 十、模块独立性设计

### 10.1 设计原则

| 原则 | 说明 |
|------|------|
| **接口隔离** | 模块间通过接口通信，不直接依赖具体实现 |
| **依赖倒置** | 上层模块依赖抽象接口，不依赖下层具体类 |
| **单一职责** | 每个模块只负责一项功能 |
| **可替换性** | 同一接口可替换不同实现（如 LLM 客户端） |

### 10.2 模块设计原则

| 原则 | 说明 |
|------|------|
| **单向依赖** | data → analysis → server，禁止反向依赖 |
| **配置外置** | 硬编码提取到 config.yaml |
| **统一异常** | 自定义异常类，区分业务/系统异常 |

### 10.3 LLM 模块使用

```python
# 使用 LLM 模块
from analysis.llm import get_client, is_available

# 检查可用性
if is_available():
    client = get_client()
    result = client.chat([{"role": "user", "content": "分析基金001234"}])
```

### 10.4 模块替换（如未来需要）

如需替换 LLM 后端，只需修改 `analysis/llm/client.py`，业务代码无需改动。

---

## 十一、实施注意事项

### 11.1 代码组织

| 规则 | 说明 |
|------|------|
| **禁止循环依赖** | data → analysis → data 禁止 |
| **接口先于实现** | 先定义接口，再写实现 |
| **配置外置** | 硬编码配置需提取到 config.yaml |
| **异常统一** | 自定义异常类，区分业务异常和系统异常 |

### 11.2 文件命名规范

```
# 模块目录
data/
├── interfaces.py      # 接口定义
├── repo.py            # 默认实现
├── mysql_repo.py      # MySQL 实现（可选）
├── cache.py           # 缓存实现
└── ...

analysis/
├── interfaces.py     # 接口定义
├── metrics.py        # 默认实现
├── fund_metrics.py   # 基金指标（可选）
└── ...
```

### 11.3 模块集成顺序

```
Step 1: 实现数据层 (fund_repo.py, cache.py)
    │
    ▼
Step 2: 实现分析层 (fund_metrics.py)
    │
    ▼
Step 3: 集成测试 (数据层 → 分析层)
    │
    ▼
Step 4: 实现 API 层
    │
    ▼
Step 5: 实现 LLM 集成 (已完成)
    │
    ▼
Step 6: 前端集成
```

### 11.4 测试策略

```python
# 单元测试 - 测试单个模块
# tests/test_fund_metrics.py
def test_calculate_return():
    from analysis.fund_metrics import calculate_return
    import pandas as pd
    nav = pd.Series([1.0, 1.1, 1.2])
    assert calculate_return(nav, 30) > 0

# 接口测试 - 测试接口契约
# tests/test_interfaces.py
def test_fund_data_port():
    from data.interfaces import FundDataPort
    # 验证接口方法存在
    assert hasattr(FundDataPort, 'get_fund_list')
    assert hasattr(FundDataPort, 'get_fund_nav')

# 集成测试 - 测试模块间通信
# tests/test_integration.py
def test_metrics_with_real_data():
    from data.fund_repo import get_fund_nav
    from analysis.fund_metrics import calculate_return
    
    nav_df = get_fund_nav('001234', days=365)
    nav_series = nav_df.set_index('nav_date')['unit_nav']
    result = calculate_return(nav_series)
    assert isinstance(result, (int, float))
```

### 11.5 常见错误与避免

| 错误 | 避免方法 |
|------|----------|
| 直接在业务代码 new 具体类 | 使用依赖注入 |
| 硬编码 API Key | 提取到环境变量/config.yaml |
| 模块间直接 import | 通过接口中转 |
| 忘记接口变更同步 | 接口文档与代码同步 |
| 单个文件过大 | 按职责拆分模块 |

### 11.6 重构检查清单

```markdown
- [ ] 确认修改的模块有对应接口
- [ ] 确认修改不破坏现有接口
- [ ] 确认修改不影响依赖该模块的其他模块
- [ ] 运行单元测试
- [ ] 运行集成测试
- [ ] 更新接口文档（如有变化）
```

---

## 十二、配置管理

### 12.1 配置文件结构

```yaml
# config.yaml
app:
  name: FundProphet
  debug: true

database:
  host: 127.0.0.1
  port: 3306
  user: root
  password: xxx
  database: fund_db

cache:
  enabled: true
  default_ttl: 300

llm:
  enabled: true
  api_key: ${MINIMAX_API_KEY}

fund:
  max_batch: 10
  default_days: 365
  update_hour: 16
  update_minute: 30

lstm:
  epochs: 10
  hidden_size: 32
  batch_size: 64
  seq_length: 20

frontend:
  host: localhost
  port: 5173

backend:
  host: 0.0.0.0
  port: 5050
```

### 12.2 环境变量

```bash
# .env 文件（不提交到版本控制）
MINIMAX_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
MYSQL_PASSWORD=xxx
```

---

## 十三、目录结构（最终版）

```
trade/
├── .env                         # 环境变量（不提交）
├── config.yaml                  # 配置文件
├── server.py                    # 启动入口
├── requirements.txt             # Python 依赖
│
├── data/                       # 数据层
│   ├── __init__.py
│   ├── mysql.py               # MySQL 连接
│   ├── schema.py              # 表结构
│   ├── cache.py               # 缓存实现
│   ├── fund_repo.py           # 基金仓储
│   ├── fund_fetcher.py        # 数据抓取
│   └── index_repo.py          # 指数仓储
│
├── analysis/                   # 分析层
│   ├── __init__.py
│   ├── fund_data.py           # 数据处理
│   ├── fund_metrics.py        # 指标计算
│   ├── fund_benchmark.py      # 基准对比
│   ├── fund_lstm.py           # LSTM 预测
│   ├── fund_factor.py         # 多因子筛选
│   ├── technical.py           # 技术指标
│   ├── factor_library.py      # 因子库
│   ├── llm/                  # ★ 大模型模块
│   │   ├── __init__.py
│   │   └── client.py         # MiniMax 客户端
│   └── utils.py              # 分析工具
│
├── server/                     # Web 层
│   ├── __init__.py
│   ├── app.py                # Flask 应用
│   ├── utils.py              # 通用工具
│   ├── logging_config.py     # 日志配置
│   ├── scheduler.py          # 定时任务
│   └── routes/
│       ├── __init__.py
│       └── api.py            # API 路由
│
├── frontend/                   # 前端
│   ├── src/
│   │   ├── api/             # API 请求
│   │   ├── components/      # 组件
│   │   ├── views/          # 页面
│   │   ├── stores/         # 状态管理
│   │   ├── styles/         # 卡通样式
│   │   ├── router/         # 路由
│   │   ├── App.vue
│   │   └── main.js
│   └── package.json
│
├── tests/                      # 测试（后续添加）
│   ├── test_data/
│   └── test_analysis/
│
└── docs/                      # 文档
    └── ARCHITECTURE.md       # 本文档
```
