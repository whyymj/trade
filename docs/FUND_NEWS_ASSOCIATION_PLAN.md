# 基金新闻关联分析计划

## 背景

当前系统存在以下问题：
1. 基金没有行业字段，无法按行业分类
2. 新闻分类较少（general/全球/行业），粒度不够细
3. 基金与新闻之间没有关联，无法针对特定基金分析相关新闻

## 目标

实现基金与新闻的智能关联，支持按行业维度分析基金相关新闻

---

## 整体架构

```
┌─────────────┐    行业分析    ┌──────────────────┐
│  基金重仓股  │ ──────────→  │   基金行业标签    │
└─────────────┘               └──────────────────┘
                                    ↓
┌─────────────┐    LLM分类     ┌──────────────────┐
│    新闻     │ ──────────→  │  新闻行业标签     │
└─────────────┘               └──────────────────┘
                                    ↓
                            ┌──────────────────┐
                            │  关联匹配算法    │
                            └──────────────────┘
                                    ↓
                            ┌──────────────────┐
                            │  相关新闻推荐    │
                            └──────────────────┘
```

---

## 详细设计

### Phase 1: 基金行业分析

#### 1.1 基金行业数据获取

**数据来源**：
- 基金季报/年报中的重仓股
- 基金持仓股票的行业分布

**行业分类标准**：

| 行业 | 关键词 |
|------|--------|
| 新能源 | 光伏、锂电、储能，光伏、电动车、锂电池、动力电池 |
| 半导体 | 芯片、集成电路、半导体、AI芯片、算力、GPU |
| 医药 | 创新药、医疗器械、中药、疫苗、CRO、医疗服务 |
| 消费 | 白酒、食品饮料、免税、零售、消费电子 |
| 金融 | 银行、保险、券商、地产、信托 |
| 军工 | 航空航天、船舶、导弹、无人机、军工电子 |
| TMT | 互联网，软件、云服务、数字经济 |
| 基建 | 建筑、建材、工程机械、水泥 |
| 农业 | 种植、养殖、农药、化肥 |
| 化工 | 石化、化工新材料、化学制药 |

#### 1.2 基金行业标签表结构

```sql
-- 基金行业表
CREATE TABLE IF NOT EXISTS fund_industry (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fund_code VARCHAR(20) NOT NULL,
    industry VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) DEFAULT 0,
    source VARCHAR(20) DEFAULT 'llm',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_fund_industry (fund_code, industry),
    INDEX idx_fund_code (fund_code),
    INDEX idx_industry (industry)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 1.3 新增API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/fund/analyze-industry/:code | 分析基金行业 |
| GET | /api/fund/industries/:code | 获取基金行业标签 |

---

### Phase 2: 新闻行业分类增强

#### 2.1 LLM新闻行业分类

**分类Prompt**：

```python
prompt = """
你是一个专业的财经分析师。请根据以下新闻标题，判断其所属的行业类别。

分类选项：
- 新能源(光伏、锂电、储能，光伏、电动车)
- 半导体(芯片、集成电路、AI算力、GPU)
- 医药(创新药、医疗器械、中药、疫苗)
- 消费(白酒、食品饮料、免税、零售)
- 金融(银行、保险、券商、地产)
- 军工(航空航天、船舶、导弹)
- TMT(互联网，软件，云服务)
- 宏观(政策，经济数据、GDP、CPI、PMI)
- 全球(美联储、国际局势，地缘政治)
- 其他

新闻标题：{title}

请以JSON格式输出：
{{"industry": "行业名", "confidence": 0.95, "keywords": ["关键词1", "关键词2"]}}
"""
```

#### 2.2 新闻表结构增强

```sql
-- 新闻表增加行业字段
ALTER TABLE news_data 
ADD COLUMN industry VARCHAR(50) DEFAULT 'general',
ADD COLUMN industry_confidence DECIMAL(5,2) DEFAULT 0,
ADD COLUMN keywords JSON,
ADD INDEX idx_industry (industry);

-- 新闻行业分析结果表
CREATE TABLE IF NOT EXISTS news_industry_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    news_id VARCHAR(100) NOT NULL,
    industry VARCHAR(50),
    confidence DECIMAL(5,2),
    keywords JSON,
    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_news_id (news_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 2.3 新增API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/news/analyze-industry | 批量分析新闻行业 |
| GET | /api/news/by-industry | 按行业获取新闻 |

---

### Phase 3: 基金新闻关联

#### 3.1 关联算法

```
1. 获取基金的行业标签（按置信度排序）
2. 查询近期新闻（默认7天）
3. 匹配相同行业的新闻
4. 计算关联度 = 行业匹配 * 时间衰减 * 关键词匹配
5. 按关联度排序返回
```

#### 3.2 新增API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/fund/news/:code | 获取基金相关新闻 |

---

### Phase 4: 前端展示

#### 4.1 基金详情页增强

**位置**：基金详情页 - 基本信息区域下方

**UI设计**：
```
┌─────────────────────────────────────────┐
│  行业标签                                    │
│  ┌──────┐ ┌──────┐ ┌──────┐            │
│  │新能源 │ │电力设备│ │ 储能  │            │
│  │ 85%  │ │ 62%  │ │ 45%  │            │
│  └──────┘ └──────┘ └──────┘            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  相关新闻                                    │
│  ┌─────────────────────────────────────┐│
│  │ 📰 光伏行业协会：今年装机量预计增长30%  ││
│  │     新能源 | 2026-03-04 | 置信度95%  ││
│  ├─────────────────────────────────────┤│
│  │ 📰 宁德时代发布新型储能电池            ││
│  │     新能源 | 2026-03-03 | 置信度88%  ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

#### 4.2 新闻列表页增强

**新增筛选条件**：
- 按行业筛选（多选）
- 按关键词搜索

#### 4.3 新闻详情页增强

**新增显示**：
- 行业标签
- 相关行业基金推荐

---

## 实施顺序

```
Phase 1 (第1天)
    ↓
Phase 2 (第2天)
    ↓
Phase 3 (第3天)
    ↓
Phase 4 (第4天)
```

---

## 测试计划

| 模块 | 测试用例 |
|------|---------|
| 基金行业分析 | 测试行业识别准确率 |
| 新闻行业分类 | 测试分类准确率 |
| 新闻关联 | 测试关联匹配度 |
| API测试 | 测试各接口返回 |
| 前端测试 | 测试页面展示 |

---

## 预计效果

- 基金行业覆盖率：>90%
- 新闻行业分类准确率：>80%
- 关联匹配准确率：>70%

---

# 附录：新闻数据源

## 国内数据源

### 综合财经
| 来源 | 接口 | 更新频率 | 数量/天 |
|------|------|---------|---------|
| 东方财富 | akshare.stock_news_em | 实时 | 60条 |
| 财联社 | cls.cn API | 实时 | 50条 |
| 华尔街见闻 | goldboot.cn API | 实时 | 30条 |
| 新浪财经 | sina.com.cn | 实时 | 50条 |
| 凤凰网财经 | ifeng.com | 实时 | 30条 |

### 垂直领域
| 来源 | 类型 | 数量/天 |
|------|------|---------|
| 同花顺 | 财经 | 40条 |
| 大智慧 | 财经 | 30条 |
| 雪球 | 投资社区 | 50条 |
| 东财股吧 | 社区 | 100条 |

---

## 国外数据源

### 国际主流媒体

| 来源 | 网站 | 类型 | 更新频率 | 数量/天 |
|------|------|------|---------|---------|
| Reuters | reuters.com | 财经、国际 | 实时 | 100+ |
| Bloomberg | bloomberg.com | 财经、金融 | 实时 | 80+ |
| CNBC | cnbc.com | 财经、商业 | 实时 | 60+ |
| Financial Times | ft.com | 财经、金融 | 实时 | 50+ |
| Wall Street Journal | wsj.com | 财经、金融 | 实时 | 50+ |
| MarketWatch | marketwatch.com | 股市、投资 | 实时 | 40+ |
| Yahoo Finance | finance.yahoo.com | 股市、财经 | 实时 | 60+ |
| Barron's | barrons.com | 投资、评论 | 日更 | 20+ |

### 美股相关

| 来源 | 网站 | 类型 | 数量/天 |
|------|------|------|---------|
| Seeking Alpha | seekingalpha.com | 投资分析 | 80+ |
| The Motley Fool | fool.com | 投资建议 | 30+ |
| Investopedia | investopedia.com | 投资教育 | 20+ |
| Barchart | barchart.com | 行情数据 | 50+ |
| Finviz | finviz.com | 股票筛选 | 30+ |

### 加密货币/新兴市场

| 来源 | 网站 | 类型 | 数量/天 |
|------|------|------|---------|
| CoinDesk | coindesk.com | 加密货币 | 50+ |
| CoinTelegraph | cointelegraph.com | 加密货币 | 50+ |
| CryptoSlate | cryptoslate.com | 加密货币 | 40+ |

### 宏观经济

| 来源 | 网站 | 类型 | 数量/天 |
|------|------|------|---------|
| Trading Economics | tradingeconomics.com | 宏观数据 | 30+ |
| Investing.com | investing.com | 宏观、财经 | 80+ |
| FocusEconomics | focus-economics.com | 宏观预测 | 10+ |

### API接口

| 来源 | API类型 | 说明 |
|------|---------|------|
| NewsAPI.org | 付费/免费 | 全球新闻API |
| Bing News Search | 付费 | 微软新闻搜索 |
| GDELT | 免费 | 全球新闻数据库 |
| Alpha Vantage | 免费/付费 | 财经新闻API |

---

## 新闻抓取策略

### 优先级

1. **高优先级**（国内主流）：
   - 东方财富
   - 财联社
   - 新浪财经

2. **中优先级**（国际主流）：
   - Reuters
   - Bloomberg
   - CNBC

3. **低优先级**（垂直领域）：
   - 加密货币
   - 投资社区

### 抓取频率

| 类型 | 频率 |
|------|------|
| 国内实时新闻 | 每30分钟 |
| 国际新闻 | 每小时 |
| 深度分析 | 每日 |
| 宏观数据 | 每日 |

### 数据处理

1. **语言处理**
   - 英文新闻 → 翻译 + 分类
   - 中文新闻 → 直接分类

2. **去重**
   - URL去重
   - 标题相似度去重（>90%）

3. **分类增强**
   - 关键词匹配
   - LLM二次分类
