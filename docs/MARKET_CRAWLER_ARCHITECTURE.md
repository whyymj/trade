# 市场行情爬虫模块 - 架构设计文档

## 一、概述

本文档描述基金分析系统中市场行情爬虫模块的架构设计，用于爬取宏观经济、资金流向、市场情绪等数据，为基金走势预测提供特征输入。

### 1.1 设计目标

| 目标 | 说明 |
|------|------|
| 数据完整 | 覆盖宏观、资金、情绪三大维度 |
| 高可用 | 多数据源自动切换 |
| 可扩展 | 便于新增爬虫源 |
| 实时性 | 支持每日增量更新 |

---

## 二、数据体系

### 2.1 数据分层

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据源层 (Source)                               │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│  第一梯队(权威)  │  第二梯队(媒体)  │       第三梯队(机构/工具)        │
│  国家统计局      │  财联社          │  akshare (整合数据)             │
│  中国人民银行    │  华尔街见闻      │  东方财富Choice                 │
│  财政部/发改委   │  证券时报        │  券商研究报告                   │
└──────────────────┴──────────────────┴──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        爬虫层 (Crawler)                                  │
├──────────────┬──────────────┬──────────────┬─────────────────────────────┤
│  宏观数据     │  资金流向    │  市场情绪     │  全球宏观                  │
│  爬虫模块     │  爬虫模块    │  爬虫模块    │  爬虫模块                 │
└──────────────┴──────────────┴──────────────┴─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        存储层 (Storage)                                  │
├──────────────┬──────────────┬──────────────┬─────────────────────────────┤
│  macro_data  │  money_flow  │  sentiment   │  global_macro              │
│  宏观经济    │  资金流向    │  市场情绪    │  全球宏观                  │
└──────────────┴──────────────┴──────────────┴─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        特征层 (Features)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  市场特征工程：宏观因子、资金因子、情绪因子 → 输入 LSTM 模型            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据分类

| 维度 | 数据项 | 更新频率 | 数据源 |
|------|--------|----------|--------|
| **宏观经济** | GDP、CPI、PPI、PMI、社融、利率、信贷 | 每月/每周 | akshare |
| **资金流向** | 北向资金、主力资金、融资融券、ETF申购 | 每日 | akshare |
| **市场情绪** | 两市成交额、涨跌停数量、市场换手率、基金发行规模 | 每日 | akshare |
| **全球宏观** | 美元指数、人民币汇率、美债收益率、布伦特原油 | 每日 | akshare |

---

## 三、数据库设计

### 3.1 表结构

#### 3.1.1 宏观经济数据表 (macro_data)

```sql
CREATE TABLE macro_data (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    indicator VARCHAR(32) NOT NULL COMMENT '指标代码: gdp/cpi/ppi/pmi/softr/m2',
    period VARCHAR(16) NOT NULL COMMENT '周期: 2024Q1 / 2024-01',
    value DECIMAL(20,4) COMMENT '数值',
    unit VARCHAR(16) COMMENT '单位: 亿元 / %',
    source VARCHAR(32) COMMENT '数据源: akshare',
    publish_date DATE COMMENT '发布时间',
    trade_date DATE COMMENT '交易日期(月末)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_indicator_period (indicator, period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 3.1.2 资金流向表 (money_flow)

```sql
CREATE TABLE money_flow (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL COMMENT '交易日期',
    north_money DECIMAL(20,2) COMMENT '北向资金(亿)',
    north_buy DECIMAL(20,2) COMMENT '北向买入额(亿)',
    north_sell DECIMAL(20,2) COMMENT '北向卖出额(亿)',
    main_money DECIMAL(20,2) COMMENT '主力资金(亿)',
    margin_balance DECIMAL(20,2) COMMENT '融资余额(亿)',
    margin_buy DECIMAL(20,2) COMMENT '融资买入额(亿)',
    margin_repay DECIMAL(20,2) COMMENT '融资偿还额(亿)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trade_date (trade_date),
    KEY idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 3.1.3 市场情绪表 (market_sentiment)

```sql
CREATE TABLE market_sentiment (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL COMMENT '交易日期',
    volume DECIMAL(20,2) COMMENT '两市成交额(亿)',
    up_count INT COMMENT '涨停数',
    down_count INT COMMENT '跌停数',
    turnover_rate DECIMAL(10,4) COMMENT '市场换手率(%)',
    advance_count INT COMMENT '上涨家数',
    decline_count INT COMMENT '下跌家数',
    new_fund_count INT COMMENT '新基金发行数(本周)',
    new_fund_scale DECIMAL(20,2) COMMENT '新基金发行规模(亿)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trade_date (trade_date),
    KEY idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 3.1.4 全球宏观数据表 (global_macro)

```sql
CREATE TABLE global_macro (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL COMMENT '交易日期',
    symbol VARCHAR(16) NOT NULL COMMENT '品种: USDX/USDCNY/CL.BRENT',
    open_price DECIMAL(14,4) COMMENT '开盘价',
    high_price DECIMAL(14,4) COMMENT '最高价',
    low_price DECIMAL(14,4) COMMENT '最低价',
    close_price DECIMAL(14,4) COMMENT '收盘价',
    change_pct DECIMAL(10,4) COMMENT '涨跌幅(%)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_symbol_date (symbol, trade_date),
    KEY idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 四、模块设计

### 4.1 目录结构

```
data/
├── __init__.py
├── mysql.py                # MySQL 连接
├── schema.py               # 表结构
├── cache.py                # 内存缓存
├── fund_repo.py            # 基金数据仓储
├── fund_fetcher.py         # 基金数据抓取
├── index_repo.py           # 指数数据仓储
│
├── market_repo.py          # ★ 市场数据仓储
├── market_crawler.py       # ★ 市场数据爬虫
│   ├── fetch_macro()       # 宏观经济抓取
│   ├── fetch_money_flow()  # 资金流向抓取
│   ├── fetch_sentiment()   # 市场情绪抓取
│   └── fetch_global()      # 全球宏观抓取
```

### 4.2 核心模块

#### 4.2.1 市场数据爬虫 (market_crawler.py)

| 函数 | 说明 | 返回类型 |
|------|------|----------|
| `fetch_macro_data()` | 获取宏观数据 | DataFrame |
| `fetch_money_flow()` | 获取资金流向 | DataFrame |
| `fetch_market_sentiment()` | 获取市场情绪 | DataFrame |
| `fetch_global_macro()` | 获取全球宏观 | DataFrame |
| `sync_all()` | 同步全部数据 | dict |

#### 4.2.2 市场数据仓储 (market_repo.py)

| 函数 | 说明 | 返回类型 |
|------|------|----------|
| `get_macro_data(indicator, days)` | 获取宏观数据 | DataFrame |
| `get_money_flow(days)` | 获取资金流向 | DataFrame |
| `get_sentiment(days)` | 获取市场情绪 | DataFrame |
| `get_global_macro(symbol, days)` | 获取全球宏观 | DataFrame |
| `get_market_features(days)` | 获取全部市场特征 | DataFrame |

---

## 五、API 接口设计

### 5.1 接口列表

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/market/macro` | 宏观经济数据 | 1天 |
| GET | `/api/market/money-flow` | 资金流向 | 1小时 |
| GET | `/api/market/sentiment` | 市场情绪 | 1小时 |
| GET | `/api/market/global` | 全球宏观 | 1小时 |
| GET | `/api/market/features` | 市场特征(合并) | 1小时 |
| POST | `/api/market/sync` | 手动同步数据 | - |

### 5.2 响应格式

```json
// GET /api/market/money-flow
{
  "code": 0,
  "data": {
    "list": [
      {
        "trade_date": "2024-01-15",
        "north_money": 25.63,
        "main_money": -120.5,
        "margin_balance": 15420.3
      }
    ],
    "update_time": "2024-01-15 16:00:00"
  }
}
```

---

## 六、LSTM 集成

### 6.1 特征准备

```python
# analysis/fund_lstm.py 增强

def prepare_features(fund_code: str, days: int = 60) -> pd.DataFrame:
    """
    准备模型输入特征：基金净值 + 市场特征
    """
    # 1. 基金净值序列
    nav_df = get_fund_nav(fund_code, days=days)
    
    # 2. 市场特征
    market_features = get_market_features(days=days)
    
    # 3. 合并
    features = merge_nav_with_market(nav_df, market_features)
    
    return features
```

### 6.2 特征工程

| 特征类型 | 指标 | 说明 |
|----------|------|------|
| 资金因子 | north_money_ma5 | 北向资金5日均值 |
| 资金因子 | main_money_ma5 | 主力资金5日均值 |
| 资金因子 | margin_ratio | 融资买入/偿还比 |
| 情绪因子 | volume_ma5 | 成交额5日均值 |
| 情绪因子 | up_ratio | 涨停占比 |
| 情绪因子 | turnover_ma5 | 换手率5日均值 |
| 宏观因子 | gdp_trend | GDP趋势 |
| 宏观因子 | m2_yoy | M2同比增速 |

---

## 七、定时任务

### 7.1 更新策略

| 数据类型 | 更新时机 | 策略 |
|----------|----------|------|
| 宏观经济 | 每月15日 / 每周一 | 全量更新 |
| 资金流向 | 每日16:00后 | 增量更新 |
| 市场情绪 | 每日15:30后 | 增量更新 |
| 全球宏观 | 每日23:00 | 全量更新 |

### 7.2 调度配置

```yaml
# config.yaml
scheduler:
  market:
    macro:
      enabled: true
      cron: "0 8 15 * *"  # 每月15日
    money_flow:
      enabled: true
      cron: "0 16 * * 1-5" # 每日16:00
    sentiment:
      enabled: true
      cron: "30 15 * * 1-5" # 每日15:30
    global:
      enabled: true
      cron: "0 23 * * *"  # 每日23:00
```

---

## 八、实施计划

### Phase 1: 数据库 (0.5天)

- [ ] 在 schema.py 新增表
- [ ] 执行 SQL 创建表

### Phase 2: 爬虫模块 (1.5天)

- [ ] 实现 data/market_crawler.py
- [ ] 实现 akshare 数据获取
- [ ] 多数据源容错

### Phase 3: 仓储层 (0.5天)

- [ ] 实现 data/market_repo.py

### Phase 4: API (0.5天)

- [ ] 新增 /api/market/* 路由

### Phase 5: 集成测试 (1天)

- [ ] 数据拉取测试
- [ ] LSTM 特征集成测试

---

## 九、工作量估算

| 阶段 | 天数 |
|------|------|
| 数据库设计 | 0.5 |
| 爬虫模块 | 1.5 |
| 仓储层 | 0.5 |
| API | 0.5 |
| 集成测试 | 1 |
| **总计** | **4天** |
