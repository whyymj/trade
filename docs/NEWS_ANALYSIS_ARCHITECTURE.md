# 新闻分析系统 - 架构设计文档

## 一、概述

本文档描述新闻分析系统的架构设计，实现以下流程：
1. 爬取财经新闻
2. 使用 **MiniMax LLM** 提取关键信息
3. 将汇总信息发送给 **DeepSeek LLM** 做最终分析

### 1.1 系统流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  新闻爬虫   │───▶│  信息提取   │───▶│  汇总整理   │───▶│  深度分析   │
│  Crawler   │    │  (MiniMax)  │    │  (合并要点) │    │  (DeepSeek) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 二、数据设计

### 2.1 新闻数据表

```sql
CREATE TABLE news_data (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(256) NOT NULL,
    content TEXT,
    source VARCHAR(32) COMMENT '来源: 财联社/华尔街见闻/新华社',
    url VARCHAR(512),
    published_at DATETIME,
    category VARCHAR(32) COMMENT '分类: 宏观/行业/公司/全球',
    sentiment VARCHAR(16) COMMENT '情绪: positive/negative/neutral',
    importance TINYINT DEFAULT 0 COMMENT '重要性: 1-5',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_url (url),
    KEY idx_category (category),
    KEY idx_published_at (published_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 2.2 分析结果表

```sql
CREATE TABLE news_analysis (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    analysis_date DATE NOT NULL,
    news_count INT COMMENT '分析的新闻数量',
    summary TEXT COMMENT 'MiniMax 提取的要点',
    deep_analysis TEXT COMMENT 'DeepSeek 最终分析',
    market_impact VARCHAR(32) COMMENT '市场影响: bullish/bearish/neutral',
    key_events JSON COMMENT '关键事件列表',
    investment_advice TEXT COMMENT '投资建议',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_date (analysis_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 三、模块设计

### 3.1 目录结构

```
analysis/
├── llm/
│   ├── __init__.py
│   ├── client.py           # MiniMax 客户端
│   ├── deepseek.py        # ★ 新增：DeepSeek 客户端
│   └── news_analyzer.py   # ★ 新增：新闻分析器
│
data/
├── news_crawler.py         # ★ 新增：新闻爬虫
├── news_repo.py            # ★ 新增：新闻仓储
└── ...
```

### 3.2 新闻爬虫 (news_crawler.py)

| 函数 | 说明 | 返回类型 |
|------|------|----------|
| `fetch_cailian` | 爬取财联社快讯 | List[News] |
| `fetch_wallstreet` | 爬取华尔街见闻 | List[News] |
| `fetch_xinhua` | 爬取新华社 | List[News] |
| `fetch_all()` | 聚合全部新闻 | List[News] |

### 3.3 LLM 客户端

#### 3.3.1 MiniMax 客户端 (已有)

```python
from analysis.llm import get_client  # 获取 MiniMax 客户端
```

#### 3.3.2 DeepSeek 客户端 (新增)

```python
# analysis/llm/deepseek.py

class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        
    def chat(self, messages: list, **kwargs) -> str:
        """DeepSeek 对话请求"""
        # 实现...
```

### 3.4 双 LLM 分析流程

```python
# analysis/llm/news_analyzer.py

class NewsAnalyzer:
    """双 LLM 新闻分析器"""
    
    def __init__(self):
        self.minimax = get_client()
        self.deepseek = DeepSeekClient()
    
    def analyze(self, news_list: List[News]) -> AnalysisResult:
        """
        1. MiniMax 提取关键信息
        2. DeepSeek 综合分析
        """
        # Step 1: MiniMax 信息提取
        summary = self._extract_with_minimax(news_list)
        
        # Step 2: DeepSeek 深度分析
        final_analysis = self._analyze_with_deepseek(summary)
        
        return final_analysis
    
    def _extract_with_minimax(self, news_list: List[News]) -> str:
        """MiniMax 提取要点"""
        prompt = f"""请从以下新闻中提取关键信息：
        - 政策变化
        - 经济数据
        - 重大事件
        - 行业动态
        
        新闻列表：
        {self._format_news(news_list)}
        
        请简洁列出要点，每条不超过20字。"""
        
        return self.minimax.chat([{"role": "user", "content": prompt}])
    
    def _analyze_with_deepseek(self, summary: str) -> str:
        """DeepSeek 深度分析"""
        prompt = f"""你是一位专业的财经分析师。请根据以下新闻要点：
        1. 分析对A股市场的影响
        2. 判断市场情绪（看涨/看跌/中性）
        3. 给出投资建议
        
        新闻要点：
        {summary}
        
        请给出详细分析报告。"""
        
        return self.deepseek.chat([{"role": "user", "content": prompt}])
```

---

## 四、API 接口设计

### 4.1 接口列表

| 方法 | 路径 | 说明 | 缓存 |
|------|------|------|------|
| GET | `/api/news/list` | 新闻列表 | 5分钟 |
| GET | `/api/news/latest` | 最新新闻 | 5分钟 |
| POST | `/api/news/sync` | 手动同步新闻 | - |
| POST | `/api/news/analyze` | 分析新闻 | - |
| GET | `/api/news/analysis/latest` | 最新分析结果 | 1小时 |

### 4.2 请求示例

```bash
# 分析新闻
curl -X POST /api/news/analyze \
  -d '{"days": 1, "categories": ["宏观", "全球"]}'
```

### 4.3 响应格式

```json
{
  "code": 0,
  "data": {
    "news_count": 15,
    "summary": "要点1：央行降准...\n要点2：美伊冲突...",
    "deep_analysis": "## 市场分析\n\n1. 宏观影响...",
    "market_impact": "bearish",
    "key_events": [
      {"title": "央行降准", "impact": "positive"},
      {"title": "美伊冲突", "impact": "negative"}
    ],
    "investment_advice": "建议减仓避险",
    "analyzed_at": "2024-01-15 16:30:00"
  }
}
```

---

## 五、Prompt 设计

### 5.1 MiniMax 信息提取 Prompt

```
你是一位财经信息提取助手。请从新闻列表中提取关键信息。

要求：
1. 每条新闻提取：时间、事件、影响
2. 按重要性排序
3. 过滤噪音信息
4. 合并同类事件

输出格式：
- 政策：xxx
- 数据：xxx
- 事件：xxx
- 行业：xxx
```

### 5.2 DeepSeek 分析 Prompt

```
作为资深财经分析师，请根据以下新闻要点进行深度分析：

1. 【宏观判断】对国内宏观经济的影响
2. 【资金面】对市场流动性的影响
3. 【情绪面】投资者情绪变化
4. 【行业影响】对哪些行业有利/不利
5. 【操作建议】仓位建议

要求：
- 分析逻辑清晰
- 结论明确
- 建议具体
```

---

## 六、定时任务

### 6.1 执行策略

| 任务 | 时机 | 说明 |
|------|------|------|
| 新闻爬取 | 每日 8:00 / 12:00 / 16:00 / 20:00 | 分时段抓取 |
| 新闻分析 | 每日 16:30 / 20:30 | 收盘后分析 |

### 6.2 调度配置

```yaml
scheduler:
  news:
    crawl:
      enabled: true
      cron: "0 8,12,16,20 * * *"  # 每日4次
    analyze:
      enabled: true
      cron: "30 16,20 * * 1-5"     # 收盘后分析
```

---

## 七、实施计划

### Phase 1: 新闻爬虫 (1天)

- [ ] 实现 news_crawler.py
- [ ] 接入财联社/华尔街见闻 API
- [ ] 测试爬取功能

### Phase 2: 数据存储 (0.5天)

- [ ] 新增数据库表
- [ ] 实现 news_repo.py

### Phase 3: DeepSeek 客户端 (0.5天)

- [ ] 实现 deepseek.py
- [ ] 测试 API 连接

### Phase 4: 分析流程 (1天)

- [ ] 实现 news_analyzer.py
- [ ] 设计 Prompt
- [ ] 调试双 LLM 流程

### Phase 5: API 接口 (0.5天)

- [ ] 新增 /api/news/* 路由
- [ ] 前端展示

### Phase 6: 定时任务 (0.5天)

- [ ] 配置调度
- [ ] 线上测试

---

## 八、工作量估算

| 阶段 | 天数 |
|------|------|
| 新闻爬虫 | 1 |
| 数据存储 | 0.5 |
| DeepSeek 客户端 | 0.5 |
| 分析流程 | 1 |
| API 接口 | 0.5 |
| 定时任务 | 0.5 |
| **总计** | **4天** |

---

## 九、配置

```yaml
# .env
MINIMAX_API_KEY=sk-cp-xxx
DEEPSEEK_API_KEY=sk-xxx

# config.yaml
llm:
  minimax:
    enabled: true
    model: MiniMax-M2.5
  deepseek:
    enabled: true
    model: deepseek-chat
```
