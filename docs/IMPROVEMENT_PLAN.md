# FundProphet 项目改进计划

> 基于全面代码审查生成，按优先级分阶段实施。

---

## 问题总览

| 优先级 | 数量 | 类别 |
|--------|------|------|
| 🔴 高 | 5 项 | 安全漏洞、运行时 Bug、路由缺失 |
| 🟡 中 | 10 项 | 代码质量、架构规范、错误处理 |
| 🟢 低 | 8 项 | 性能优化、无障碍、文档、规范 |

---

## Phase 1：紧急修复（安全 & Bug）

预计工时：0.5 天

### 1.1 移除 config.yaml 中的明文密码

**问题**：`config.yaml` 硬编码了数据库密码，且未被 `.gitignore` 排除。

**文件**：`config.yaml`

**方案**：
```yaml
# 修改前
database:
  password: Belldandy_13579

# 修改后
database:
  password: ${DB_PASSWORD}
```
- 敏感配置统一读取 `.env` 环境变量
- 提供 `config.yaml.example` 作为模板（不含真实密码）
- `.gitignore` 添加 `config.yaml`（保留 `config.yaml.example`）

**验收**：`config.yaml` 中无明文密码；项目可正常启动。

---

### 1.2 修复 LLM 调用参数类型错误

**问题**：`news_classification` 和 `investment_advice` 模块传入字符串，`DeepSeekClient.chat()` 期望 `List[dict]`，运行时报错。

**文件**：
- `modules/news_classification/analyzer.py`
- `modules/investment_advice/__init__.py`

**方案**：
```python
# 修改前
result = self.llm.chat(prompt)

# 修改后
result = self.llm.chat([{"role": "user", "content": prompt}])
```

**验收**：新闻分类和投资建议功能可正常调用 LLM；对应测试通过。

---

### 1.3 修复测试代码 Bug

**问题**：
- `tests/test_fund.py:165,175` — `get_client()` 未定义，应为 `get_minimax_client()`
- `tests/test_fund_news_association.py` — Mock 目标错误，未 patch 到实际使用的 `fund_repo`

**文件**：
- `tests/test_fund.py`
- `tests/test_fund_news_association.py`

**验收**：`python -m pytest tests/test_fund.py tests/test_fund_news_association.py -v` 全部通过。

---

### 1.4 修复新闻列表路由未注册

**问题**：`news.py` 中 `get_news_list` 函数已实现但缺少 `@news_bp.route` 装饰器，`/api/news/list` 接口不可用。

**文件**：`server/routes/news.py`

**方案**：添加路由装饰器：
```python
@news_bp.route("/list", methods=["GET"])
def get_news_list():
    ...
```

**验收**：`GET /api/news/list?days=1` 返回正确数据。

---

### 1.5 修复 FundDetail XSS 风险

**问题**：`v-html="analysisRendered"` 直接渲染 LLM 返回内容，存在 XSS 风险。

**文件**：`frontend/src/views/FundDetail.vue`

**方案**：
```bash
cd frontend && npm install dompurify
```
```javascript
import DOMPurify from 'dompurify'
const analysisRendered = computed(() => DOMPurify.sanitize(rawHtml))
```

**验收**：包含 `<script>` 标签的 LLM 输出被过滤，正常 HTML 渲染不受影响。

---

## Phase 2：代码质量提升

预计工时：2 天

### 2.1 统一 API 响应格式

**问题**：三个路由文件使用不同的响应格式（`error`/`message`/`code`/`data` 混用）。

**文件**：
- `server/routes/api.py`
- `server/routes/news.py`
- `server/routes/market.py`
- 新增 `server/utils.py` 响应工具

**方案**：封装统一响应函数：
```python
# server/utils.py
def success_response(data=None, message="ok"):
    return jsonify({"code": 0, "data": data, "message": message})

def error_response(message, code=400, http_status=None):
    return jsonify({"code": code, "message": message}), (http_status or code)
```
所有路由统一使用这两个函数。

**验收**：所有 API 返回格式一致；现有测试通过。

---

### 2.2 消除后端路由代码重复

**问题**：
- 基金代码验证逻辑在 `api.py` 重复 10+ 次
- LLM 分析 5 个路由结构几乎一样
- 新闻序列化逻辑重复

**文件**：`server/routes/api.py`, `server/routes/news.py`

**方案**：
```python
# 提取装饰器
def require_fund_code(f):
    @wraps(f)
    def wrapper(code, *args, **kwargs):
        code = (code or "").strip()
        if not code:
            return error_response("基金代码不能为空")
        return f(code, *args, **kwargs)
    return wrapper

# 提取 LLM 分析通用函数
def _llm_analysis(code, analysis_type, prompt_builder):
    ...
```

**验收**：`api.py` 减少 ~200 行重复代码；所有路由功能不变。

---

### 2.3 替换 print 为 logging

**问题**：后端多处使用 `print` 输出错误信息，不利于生产环境日志管理。

**涉及文件**：
- `server/routes/api.py`
- `server/routes/news.py`
- `server/routes/market.py`
- `data/news/repo.py`
- `data/fund_repo.py`
- `server/app.py`

**方案**：
```python
import logging
logger = logging.getLogger(__name__)

# 修改前
print(f"[错误] {e}")

# 修改后
logger.error("操作失败: %s", e, exc_info=True)
```

**验收**：项目中无 `print` 用于错误/日志输出（允许启动提示信息保留 print）。

---

### 2.4 完善 .gitignore 并清理已跟踪的 .pyc

**问题**：`.gitignore` 缺少 `__pycache__/`、`*.pyc`、`.pytest_cache/` 等常见模式，git status 显示多个 `.pyc` 文件被跟踪。

**文件**：`.gitignore`

**方案**：
```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/

# Environment
.env
.env.*

# IDE
.vscode/
.idea/

# Build
frontend/dist/
*.egg-info/
```
并执行：
```bash
git rm -r --cached **/__pycache__
git rm --cached **/*.pyc
```

**验收**：`git status` 不再显示 `.pyc` 文件；新生成的 `.pyc` 不被跟踪。

---

### 2.5 前端统一错误处理

**问题**：大部分 API 调用失败只 `console.error`，部分使用 `alert()`。

**涉及文件**：
- `frontend/src/views/FundHome.vue`
- `frontend/src/views/FundDetail.vue`
- `frontend/src/views/NewsHome.vue`
- `frontend/src/views/NewsList.vue`

**方案**：
```javascript
// 统一使用 Element Plus 消息
import { ElMessage } from 'element-plus'

// 修改前
alert('操作失败')
console.error(err)

// 修改后
ElMessage.error('操作失败，请稍后重试')
console.error(err)
```

**验收**：所有 API 失败场景展示用户友好的提示信息；无 `alert()` 调用。

---

### 2.6 提取前端工具函数

**问题**：`getCategoryClass`、`formatTime`、`formatMoney`、`formatVolume` 等函数在多个组件中重复定义。

**方案**：新建 `frontend/src/utils/formatters.js`：
```javascript
export function formatTime(dateStr) { ... }
export function formatMoney(value) { ... }
export function formatVolume(value) { ... }
export function getCategoryClass(category) { ... }
export function getSentimentLabel(sentiment) { ... }
```
各组件统一 import 使用。

**验收**：所有格式化函数从单一文件导出；无重复定义。

---

### 2.7 修复 FundDetail 低效查询

**问题**：`FundDetail.vue` 通过 `getFundList({ page: 1, size: 1000 })` 获取完整列表后在前端过滤单个基金。

**文件**：
- `frontend/src/views/FundDetail.vue`
- `frontend/src/api/fund.js`

**方案**：
```javascript
// fund.js 新增
export function getFundDetail(code) {
  return request(`/api/fund/${code}`)
}
```
后端 `api.py` 的 `fund_detail` 路由改用直接查询，而非从列表中过滤。

**验收**：基金详情页加载只发起 1 次请求，而非拉取全量列表。

---

### 2.8 修复 Dockerfile 缺少 modules/

**问题**：Docker 构建时未复制 `modules/` 目录，部署后业务模块不可用。

**文件**：`Dockerfile`

**方案**：添加 `COPY modules/ ./modules/`

**验收**：Docker 镜像内包含 `modules/`；容器启动后相关 API 可访问。

---

### 2.9 完善 schema 创建失败处理

**问题**：`server/app.py` 中 `create_all_tables()` 失败被静默吞掉，数据库初始化问题不可见。

**文件**：`server/app.py`

**方案**：
```python
try:
    from data.schema import create_all_tables
    create_all_tables()
except Exception as e:
    logging.critical("数据库表初始化失败: %s", e)
    raise
```

**验收**：数据库连接失败时，应用启动报错（而非静默启动后 500）。

---

### 2.10 修复 NewsItem 缺少 id 字段

**问题**：`data/news/repo.py` 查询 news 时获取了 `id` 但未传入 `NewsItem`，导致 `news_ids` 始终为空，行业标签更新失效。

**文件**：`data/news/repo.py`

**方案**：`NewsItem` dataclass 添加 `id: Optional[int] = None` 字段，`get_news` 查询结果映射时赋值 `id`。

**验收**：`analyze_news` 后 `news_ids` 非空；`update_news_industry_tags` 正确执行。

---

## Phase 3：前端优化

预计工时：1.5 天

### 3.1 拆分 FundDetail 大组件

**问题**：`FundDetail.vue` 约 570 行，职责过多。

**方案**：拆分为子组件：
```
components/fund-detail/
├── FundHeader.vue          # 基金头部信息
├── HoldingsTable.vue       # 持仓明细表
├── NavChart.vue            # 净值走势图
├── PerformanceMetrics.vue  # 业绩指标
├── AIAnalysis.vue          # AI 分析
└── FundNewsSection.vue     # 相关新闻
```

**验收**：`FundDetail.vue` 降至 ~150 行；各子组件独立可维护。

---

### 3.2 修复搜索功能失效

**问题**：`FundHome.vue` 中 `searchKeyword` 定义了但未传递给 `loadFunds()`。

**文件**：
- `frontend/src/views/FundHome.vue`
- `frontend/src/api/fund.js`

**方案**：`getFundList` 支持 `keyword` 和 `industry_tag` 参数：
```javascript
export function getFundList(params = {}) {
  return request('/api/fund/list', { params })
}
```
`loadFunds` 传入搜索和筛选参数。

**验收**：输入搜索关键词后列表正确过滤。

---

### 3.3 添加加载状态

**问题**：`NewsHome.vue` 等组件发起多个并行请求但无加载指示。

**方案**：所有异步数据加载添加 loading 状态和骨架屏/spinner 展示。

**验收**：数据加载中有明确的视觉反馈。

---

### 3.4 ECharts 性能优化

**问题**：
- ECharts 实例在组件卸载时未 `dispose`，存在内存泄漏
- 未做 chunk 拆分，ECharts 打入主 bundle

**文件**：
- `frontend/src/views/FundDetail.vue`（及其他使用 ECharts 的组件）
- `frontend/vite.config.js`

**方案**：
```javascript
// 组件内
onUnmounted(() => {
  chartInstance?.dispose()
})

// vite.config.js
build: {
  rollupOptions: {
    output: {
      manualChunks: {
        echarts: ['echarts']
      }
    }
  }
}
```

**验收**：切换页面后 ECharts 实例被正确释放；构建产物中 echarts 为独立 chunk。

---

## Phase 4：工程规范

预计工时：1 天

### 4.1 添加健康检查接口

**文件**：`server/routes/api.py`

**方案**：
```python
@api_bp.route("/health", methods=["GET"])
def health_check():
    return success_response({"status": "healthy", "version": "1.0.0"})
```

**验收**：`GET /api/health` 返回 200。

---

### 4.2 补全 requirements.txt

**问题**：缺少 `requests`、`python-dotenv`、`apscheduler` 等实际使用的依赖。

**方案**：补全所有运行时依赖并固定版本。

**验收**：新环境 `pip install -r requirements.txt` 后项目可正常启动。

---

### 4.3 添加共享测试基础设施

**方案**：创建 `tests/conftest.py`：
```python
import pytest
from server.app import create_app

@pytest.fixture(scope="session")
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_llm():
    from unittest.mock import Mock
    client = Mock()
    client.chat.return_value = "测试回复"
    client.is_available.return_value = True
    return client
```
移除各测试文件中重复的 `sys.path` 和 `load_dotenv` 设置。

**验收**：所有测试文件可使用共享 fixture；`pytest -v` 全部通过。

---

### 4.4 补充缺失的测试

**当前缺失**：
- `modules/investment_advice/` — 无测试
- `data/fund_holdings.py` — 无专项测试
- `analysis/fund_analyzer.py` — 无测试

**方案**：每个模块至少补充 5 个核心用例。

**验收**：测试覆盖所有业务模块。

---

### 4.5 更新 AGENTS.md

**问题**：缺少 `modules/`、`data/fund_holdings.py`、`analysis/fund_analyzer.py` 等新增模块描述，测试统计过时。

**方案**：同步更新项目结构、路由列表、测试统计。

**验收**：AGENTS.md 与代码实际结构一致。

---

### 4.6 统一前端 API 调用方式

**问题**：`FundDetail.vue` 部分接口使用 `fetch` 而非 `api/fund.js` 封装。

**方案**：所有 API 调用统一通过 `api/*.js` 模块发起。

**验收**：前端代码中无直接 `fetch('/api/...')` 调用。

---

### 4.7 前端基础无障碍支持

**方案**：
- 图标按钮添加 `aria-label`
- Tab 组件添加 `role="tablist"` / `role="tab"`
- 搜索框添加 `aria-describedby`

**验收**：axe DevTools 无严重无障碍问题。

---

### 4.8 添加 config.yaml.example

**方案**：提供模板文件，标注必填项：
```yaml
database:
  host: 127.0.0.1
  port: 3306
  user: root
  password: <YOUR_DB_PASSWORD>  # 必填
  database: fund_db
```

**验收**：新开发者可参照模板快速配置。

---

## 实施计划

```
Phase 1 (0.5天)     Phase 2 (2天)        Phase 3 (1.5天)      Phase 4 (1天)
┌──────────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1.1 密码移除  │    │ 2.1 响应格式  │     │ 3.1 组件拆分  │     │ 4.1 健康检查  │
│ 1.2 LLM修复  │    │ 2.2 消除重复  │     │ 3.2 搜索修复  │     │ 4.2 依赖补全  │
│ 1.3 测试修复  │    │ 2.3 logging  │     │ 3.3 加载状态  │     │ 4.3 测试基础  │
│ 1.4 路由注册  │    │ 2.4 gitignore│     │ 3.4 ECharts   │     │ 4.4 补测试    │
│ 1.5 XSS修复  │    │ 2.5 错误处理  │     └──────────────┘     │ 4.5 更新文档  │
└──────────────┘    │ 2.6 工具函数  │                          │ 4.6 API统一   │
                    │ 2.7 详情查询  │                          │ 4.7 无障碍    │
                    │ 2.8 Dockerfile│                          │ 4.8 配置模板  │
                    │ 2.9 schema   │                          └──────────────┘
                    │ 2.10 NewsItem│
                    └──────────────┘

总计：约 5 天
```

## 验收标准

| 阶段 | 验收方式 |
|------|----------|
| Phase 1 | `pytest -v` 全部通过；安全扫描无高危 |
| Phase 2 | 代码审查通过；API 响应格式一致；无 print 日志 |
| Phase 3 | 前端功能回归通过；Lighthouse 性能评分提升 |
| Phase 4 | 新环境可一键部署；文档与代码同步 |
