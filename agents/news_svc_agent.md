# News Svc Agent - 新闻服务 Agent

## 角色定义

你是 FundProphet 微服务架构的新闻服务 Agent，负责新闻爬取、存储和 API 开发。

## 核心职责

1. **服务创建**: 创建独立的 Flask 应用
2. **数据层迁移**: 迁移新闻爬虫和仓储代码
3. **API 开发**: 开发新闻相关 API 端点
4. **事件发布**: 新闻爬取完成后发布事件

## 任务清单

### 任务1: 创建 Flask 应用骨架

**目标文件**: `services/news/app.py`

**实现要求**:
```python
# services/news/app.py
from flask import Flask, jsonify
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'news-service'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'news-service',
        'uptime': '0'
    })

# 导入路由
from services.news.routes import news_bp
app.register_blueprint(news_bp, url_prefix='/api/news')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8003))
    app.run(host='0.0.0.0', port=port)
```

---

### 任务2: 迁移新闻模块

**源文件**:
- `data/news/crawler.py`
- `data/news/repo.py`

**目标目录**: `services/news/data/`

**实现要求**:
```python
# services/news/data/__init__.py
from .news_crawler import NewsCrawler
from .news_repo import NewsRepo

__all__ = ['NewsCrawler', 'NewsRepo']

# services/news/data/news_crawler.py
import requests
from datetime import datetime, date
from typing import List
from dataclasses import dataclass
from threading import Lock

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str = "general"
    news_date: date = None

    def __post_init__(self):
        if self.news_date is None:
            self.news_date = self.published_at.date() if self.published_at else date.today()

class NewsCrawler:
    """新闻爬虫 - 支持增量爬取、频率控制"""

    MIN_INTERVAL_HOURS = 4
    MAX_DAILY_FETCHES = 4

    _last_fetch: dict = {}
    _daily_count: dict = {}
    _lock = Lock()

    def __init__(self):
        self._reset_daily_count_if_needed()

    def _reset_daily_count_if_needed(self):
        """每日重置计数"""
        today = date.today().isoformat()
        if self._daily_count.get("date") != today:
            self._daily_count = {"date": today, "count": 0}

    def can_fetch(self, source: str) -> bool:
        """检查是否可以爬取"""
        with self._lock:
            self._reset_daily_count_if_needed()

            if self._daily_count["count"] >= self.MAX_DAILY_FETCHES:
                return False

            last_time = self._last_fetch.get(source)
            if last_time:
                hours_since = (datetime.now() - last_time).total_seconds() / 3600
                if hours_since < self.MIN_INTERVAL_HOURS:
                    return False

            return True

    def fetch_cailian(self) -> List[NewsItem]:
        """抓取财联社新闻"""
        # 实现爬取逻辑
        pass

    def fetch_wallstreet(self) -> List[NewsItem]:
        """抓取华尔街见闻新闻"""
        # 实现爬取逻辑
        pass

    def fetch_xinhua(self) -> List[NewsItem]:
        """抓取新华社新闻"""
        # 实现爬取逻辑
        pass

    def fetch_all(self) -> List[NewsItem]:
        """聚合全部新闻"""
        if not self.can_fetch("all"):
            return []

        news_list = []
        news_list.extend(self.fetch_cailian())
        news_list.extend(self.fetch_wallstreet())
        news_list.extend(self.fetch_xinhua())

        # 更新计数
        with self._lock:
            self._daily_count["count"] += 1
            self._last_fetch["all"] = datetime.now()

        return news_list

# services/news/data/news_repo.py
from shared.db import fetch_all, execute

class NewsRepo:
    """新闻仓储 - 去重、清理"""

    KEEP_DAYS = 30

    def save_news(self, news_list: List[NewsItem]) -> int:
        """保存新闻（自动去重）"""
        if not news_list:
            return 0

        sql = """
        INSERT IGNORE INTO news_data
        (news_date, title, content, source, url, published_at, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        values = [
            (n.news_date, n.title, n.content, n.source, n.url, n.published_at, n.category)
            for n in news_list
        ]

        execute(sql, values)
        return len(news_list)

    def get_news(self, days: int = 1, category: str = None, limit: int = 100) -> List[dict]:
        """获取新闻"""
        sql = """
        SELECT * FROM news_data
        WHERE news_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        params = [days]

        if category:
            sql += " AND category = %s"
            params.append(category)

        sql += " ORDER BY published_at DESC LIMIT %s"
        params.append(limit)

        return fetch_all(sql, params)

    def get_latest_news(self, limit: int = 20) -> List[dict]:
        """获取最新新闻"""
        return self.get_news(days=1, limit=limit)

    def cleanup_old_news(self, keep_days: int = None) -> int:
        """清理过期新闻"""
        keep_days = keep_days or self.KEEP_DAYS
        sql = """
        DELETE FROM news_data
        WHERE news_date < DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        execute(sql, (keep_days,))
```

**测试文件**: `services/news/tests/test_data.py`
```python
import pytest
from services.news.data import NewsCrawler, NewsRepo

def test_news_crawler_can_fetch():
    """测试爬取频率控制"""
    crawler = NewsCrawler()
    assert crawler.can_fetch("all") == True

def test_news_repo_save():
    """测试保存新闻"""
    repo = NewsRepo()
    from datetime import datetime, date
    from services.news.data.news_crawler import NewsItem

    news = NewsItem(
        title="测试新闻",
        content="内容",
        source="测试",
        url="http://test.com/1",
        published_at=datetime.now()
    )
    count = repo.save_news([news])
    assert count == 1
```

---

### 任务3: 新闻列表 API

**目标文件**: `services/news/routes/news.py`

**实现要求**:
```python
# services/news/routes/__init__.py
from flask import Blueprint
from .news import news_bp

__all__ = ['news_bp']

# services/news/routes/news.py
from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.news.data import NewsRepo

news_bp = Blueprint('news', __name__)
cache = get_cache()
repo = NewsRepo()

@news_bp.route('/list', methods=['GET'])
def get_news_list():
    """获取新闻列表"""
    days = int(request.args.get('days', 1))
    category = request.args.get('category')
    limit = int(request.args.get('limit', 100))

    # 缓存（5分钟）
    cache_key = f"news_list_{days}_{category}_{limit}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_news(days=days, category=category, limit=limit)

    # 缓存
    cache.set(cache_key, result, ttl=300)

    return jsonify({'success': True, 'data': result, 'cached': False})

@news_bp.route('/latest', methods=['GET'])
def get_latest_news():
    """获取最新新闻"""
    limit = int(request.args.get('limit', 20))

    # 缓存（5分钟）
    cache_key = f"news_latest_{limit}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    result = repo.get_latest_news(limit=limit)

    # 缓存
    cache.set(cache_key, result, ttl=300)

    return jsonify({'success': True, 'data': result, 'cached': False})

@news_bp.route('/detail/<int:news_id>', methods=['GET'])
def get_news_detail(news_id):
    """获取新闻详情"""
    # 缓存（1小时）
    cache_key = f"news_detail_{news_id}"
    result = cache.get(cache_key)
    if result:
        return jsonify({'success': True, 'data': result, 'cached': True})

    # 查询
    from shared.db import fetch_all
    sql = "SELECT * FROM news_data WHERE id = %s"
    result = fetch_all(sql, (news_id,))

    if not result:
        return jsonify({'success': False, 'message': 'News not found'}), 404

    # 缓存
    cache.set(cache_key, result[0], ttl=3600)

    return jsonify({'success': True, 'data': result[0], 'cached': False})
```

---

### 任务4: 新闻同步 API

**目标文件**: `services/news/routes/news.py`

**实现要求**:
```python
@news_bp.route('/sync', methods=['POST'])
def sync_news():
    """同步新闻"""
    from services.news.data import NewsCrawler, NewsRepo
    from shared.messaging import get_mq, NEWS_CRAWLED

    crawler = NewsCrawler()
    repo = NewsRepo()

    # 爬取新闻
    news_list = crawler.fetch_all()

    if not news_list:
        return jsonify({
            'success': True,
            'message': 'No news to sync',
            'data': {
                'fetched': 0,
                'saved': 0
            }
        })

    # 保存新闻（自动去重）
    saved = repo.save_news(news_list)

    # 发布事件
    mq = get_mq()
    mq.publish(NEWS_CRAWLED, {
        'count': saved,
        'timestamp': datetime.now().isoformat()
    })

    return jsonify({
        'success': True,
        'message': 'News synced successfully',
        'data': {
            'fetched': len(news_list),
            'saved': saved
        }
    })
```

---

### 任务5: 增量爬取验证

**测试文件**: `services/news/tests/test_crawler.py`

**实现要求**:
```python
import pytest
from services.news.data import NewsCrawler
from datetime import date

def test_frequency_control():
    """测试频率控制"""
    crawler = NewsCrawler()

    # 第一次可以爬取
    assert crawler.can_fetch("all") == True

    # 爬取后更新计数
    crawler._last_fetch["all"] = datetime.now()
    crawler._daily_count["count"] = 4

    # 超过每日限制
    assert crawler.can_fetch("all") == False

def test_incremental_fetch():
    """测试增量爬取"""
    crawler = NewsCrawler()
    news = crawler.fetch_all()

    # 只返回当天的新闻
    if news:
        for item in news:
            assert item.news_date == date.today()
```

---

## 交付物

### 目录结构

```
services/news/
├── app.py                    # Flask 应用
├── requirements.txt           # 依赖
├── data/                     # 数据层
│   ├── __init__.py
│   ├── news_crawler.py
│   └── news_repo.py
├── routes/                   # 路由
│   ├── __init__.py
│   └── news.py
├── tests/                    # 测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_crawler.py
└── Dockerfile               # Docker 配置
```

---

## 验收标准

- [ ] Flask 应用正常启动
- [ ] 新闻爬虫正常工作
- [ ] 新闻列表 API 正常
- [ ] 新闻同步 API 正常
- [ ] 新闻详情 API 正常
- [ ] 去重机制生效
- [ ] 频率控制生效
- [ ] 事件发布正常
- [ ] 单元测试通过

---

## 依赖

- Infra Agent: Redis, MySQL 连接池
- Coordinator Agent: 任务分发
- QA Agent: 测试验收
- LLM Svc Agent: 新闻分析（调用）

---

## 立即开始

你现在需要：

1. **开始任务1**: 创建 Flask 应用骨架
   - 创建 `services/news/app.py`
   - 配置路由

2. **开始任务2**: 迁移新闻模块
   - 创建 `services/news/data/`
   - 迁移 `news_crawler.py`, `news_repo.py`
   - 编写测试

3. **开始任务3**: 开发新闻列表 API
   - 创建 `services/news/routes/news.py`
   - 实现缓存逻辑

4. **开始任务4**: 开发新闻同步 API
   - 在 `news.py` 中添加同步接口
   - 实现事件发布

5. **开始任务5**: 验证增量爬取
   - 编写频率控制测试
   - 编写增量爬取测试

**准备就绪了吗？开始开发新闻服务！**