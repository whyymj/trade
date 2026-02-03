# 股票数据服务 - 接口与功能说明

本文档描述本地数据服务器的接口、数据格式与配置，便于前端对接与后期拓展。

---

## 一、项目概述

- **职责**：提供股票日线数据接口供前端 ECharts 展示；支持通过前端添加股票、一键更新、全量同步、移除股票及配置管理。
- **技术**：Flask + server 工具包（配置、akshare 拉取）+ MySQL 存储（data.schema / data.stock_repo）。
- **配置**：`config.yaml`（日期范围、复权、股票列表、**mysql** 连接）；数据抓取与管理仅通过前端页面调用 API 完成。
- **数据存储**：MySQL。表 `stock_meta`（股票元信息，含 `first_trade_date` / `last_trade_date` 已有数据时间范围、`remark` 用户说明）、`stock_daily`（日线行情）；启动时自动建表。

---

## 二、配置文件 config.yaml

| 项 | 说明 | 示例 |
|----|------|------|
| `start_date` | 开始日期，YYYYMMDD；空则近一年 | `""` 或 `"20250101"` |
| `end_date` | 结束日期，YYYYMMDD；空则今天 | `""` 或 `"20260128"` |
| `adjust` | 复权：`qfq` 前复权 / `hfq` 后复权 / `""` 不复权 | `qfq` |
| `stocks` | 股票代码列表，用于「一键更新全部」与本地列表；支持 A 股 6 位、港股 5 位或 xxxxx.HK | `["600519", "09678.HK"]` |
| `mysql` | 数据库连接（可选）：`host`, `port`, `user`, `password`, `database`, `charset` | 见 config 示例 |

接口中「日期范围与复权」均由此配置决定（未配置时默认近一年、前复权）。

---

## 三、接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 index.html |
| GET | `/static/<path>` | 静态资源 |
| GET | `/api/list` | 获取数据库中股票列表（filename 为 symbol，供 /api/data 使用） |
| GET | `/api/data` | 按 file 参数（股票代码）从数据库获取日线图表 JSON |
| GET | `/api/data_range` | 按日期范围查询多只股票日线，分页（Query: symbols, start, end, page, size） |
| GET | `/api/fetch_data/<stock_code>` | 按股票代码获取数据（优先本地，无则拉取） |
| POST | `/api/update_all` | 增量更新全部：按库内每只股票最后交易日补全至今日 |
| POST | `/api/add_stock` | 抓取该股票近 5 年日线并加入 config.stocks |
| GET | `/api/config` | 获取当前 config.yaml 内容（数据管理用） |
| PUT/PATCH | `/api/config` | 更新 config（body: start_date, end_date, adjust, stocks, output_dir） |
| POST | `/api/sync_all` | 全量同步：清空 DB 后按 config.stocks 拉取并写入数据库 |
| PUT/PATCH | `/api/stock_remark` | 更新股票说明（body: { "symbol": "600519", "remark": "说明" }） |
| POST | `/api/remove_stock` | 从 config 移除股票并删除库中该股票数据（body: { "code": "600519" }） |

---

## 四、接口详情

### 1. GET /api/list

获取数据库中已有日线数据的股票列表。

**响应 200**

```json
{
  "files": [
    { "filename": "600519", "displayName": "贵州茅台", "remark": "用户说明" }
  ]
}
```

- `filename`：股票代码，用于请求 `/api/data?file=600519`。
- `displayName`：展示用名称（来自 stock_meta 或代码）。
- `remark`：用户手动输入的说明（可 PUT /api/stock_remark 更新）。

---

### 2. GET /api/data

按股票代码从数据库读取日线，转为图表用 JSON。

**请求**

- Query: `file` = 股票代码（与 list 返回的 `filename` 一致，如 `600519`）。

**响应 200**

见下文「图表数据格式」。

**错误**

- 400：缺少或非法 `file`。
- 404：暂无数据或拉取失败。
- 500：读取失败。

---

### 3. GET /api/data_range

按日期范围查询多只股票日线，分页返回。

**请求**

- Query: `symbols` = 股票代码，逗号分隔（必填）；`start`、`end` = 日期 YYYY-MM-DD（可选）；`page` = 页码，默认 1；`size` = 每页条数，默认 20，最大 500。

**响应 200**

```json
{
  "total": 1000,
  "page": 1,
  "page_size": 20,
  "data": [
    { "symbol": "600519", "trade_date": "2025-02-01", "open": 1680.0, "high": 1690.0, "low": 1675.0, "close": 1685.0, "volume": 1234567, "amount": 2080000000, "amplitude": 0.59, "change_pct": 0.3, "change_amt": 5.0, "turnover_rate": 0.98 }
  ]
}
```

**错误**

- 400：缺少或非法 `symbols`。
- 500：服务器错误。

---

### 4. GET /api/fetch_data/<stock_code>

按股票代码获取日线：优先从数据库读取，若无则用 akshare 拉取并写入 DB（A 股/港股自动区分）。日期范围与复权来自 config。

**路径参数**

- `stock_code`：A 股 6 位数字（如 `600519`）或港股 5 位数字/5位.HK（如 `09678` 或 `09678.HK`）。

**响应 200**

与 `/api/data` 同结构（图表数据格式）。

**错误**

- 400：`stock_code` 格式非法（非 A 股 6 位或港股 5 位/.HK）。
- 502：拉取失败或暂无数据。

---

### 5. POST /api/update_all

增量更新：按 config 中 `stocks` 列表，对每只股票只拉取「库内最后交易日」至今日的缺失日线并写入（不重复拉取已有区间）。

**请求**

- 无 body。

**响应 200**

```json
{
  "ok": true,
  "results": [
    { "symbol": "600519", "ok": true, "message": "已更新 244 条" },
    { "symbol": "000001", "ok": false, "message": "拉取失败或无数据" }
  ]
}
```

---

### 6. POST /api/add_stock

抓取该股票**近 5 年**日线 → 写入数据库（stock_meta + stock_daily）→ 将代码追加到 config 的 `stocks`。

**请求**

- Content-Type: `application/json` 或 form。
- Body JSON: `{ "code": "600519" }` 或 `{ "code": "09678.HK" }`；form: `code=600519`。

**响应 200**

```json
{
  "ok": true,
  "message": "已抓取并加入配置，共 244 条",
  "displayName": "贵州茅台"
}
```

**错误**

- 400：缺少 `code`。
- 502：`ok: false`，`message` 为错误说明（如「股票代码需为 A股6位数字 或 港股5位数字/5位.HK」「拉取数据失败或暂无数据」等）。

---

### 7. GET /api/config

获取当前 `config.yaml` 内容（日期范围、复权、股票列表等），供前端数据管理页展示与编辑。

**响应 200**：JSON 对象，键与 config 一致（如 `start_date`, `end_date`, `adjust`, `stocks`, `mysql` 等）。

---

### 8. PUT /api/config

更新 config。请求体可含 `start_date`, `end_date`, `adjust`, `stocks`, `output_dir` 等，与现有配置合并后写回 `config.yaml`。

**响应 200**：`{ "ok": true, "message": "已保存" }`；失败时 `ok: false`。

---

### 9. POST /api/sync_all

全量同步：先清空数据库中全部日线与元信息，再按 config 的 `stocks` 逐个拉取并写入。适用于「重置后按当前配置重建数据」。

**响应 200**：`{ "ok": true, "results": [ { "symbol": "...", "ok": true/false, "message": "..." }, ... ] }`。

---

### 10. POST /api/remove_stock

从 config 的 `stocks` 中移除指定代码，并删除数据库中该股票的全部日线与元信息。

**请求**：Body JSON `{ "code": "600519" }`。  
**响应 200**：`{ "ok": true, "message": "已移除" }`；若本就不在配置中则 `message`: "已不在配置中"。

---

## 五、图表数据格式（ECharts 用）

以下接口返回的「单只股票日线」均为同一结构，便于前端统一处理：

- `GET /api/data`
- `GET /api/fetch_data/<stock_code>`

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `dates` | string[] | 日期序列，如 `["2025-02-05", "2025-02-06"]` |
| `开盘` | number[] | 开盘价 |
| `收盘` | number[] | 收盘价 |
| `最高` | number[] | 最高价 |
| `最低` | number[] | 最低价 |
| `成交量` | number[] | 成交量 |
| `成交额` | number[] | 成交额 |
| 其他 | - | 日线其余列（如 `振幅`、`涨跌幅`、`换手率`）同样按列名成数组返回 |

前端可直接用 `dates` 作 x 轴，`开盘/收盘/最高/最低` 等作 series。

---

## 六、server 包工具函数（拓展参考）

逻辑集中在 `server/utils.py`，便于复用与单元测试：

| 函数 | 说明 |
|------|------|
| `get_data_dir()` | 数据目录（来自 config.output_dir） |
| `load_config()` | 读取 config.yaml |
| `is_a_share_stock(symbol)` / `is_hk_stock(symbol)` / `is_valid_stock_code(symbol)` | 判断 A 股 6 位 / 港股 5 位或 .HK |
| `get_stock_name(symbol)` | 股票代码 → 名称（akshare，A 股/港股分别用不同接口） |
| `fetch_hist_remote(symbol, start, end, adjust)` | 强制从网络拉取日线（A 股 stock_zh_a_hist，港股 stock_hk_hist） |
| `fetch_hist(symbol, start, end, adjust)` | 优先数据库，无则拉取并写入 DB（自动区分 A 股/港股） |
| `save_stock_db(symbol, df, name?)` | 写入 stock_meta + stock_daily |
| `df_to_chart_result(df)` | DataFrame → 图表 JSON |
| `get_date_range_from_config()` | 从 config 取 start/end/adjust |
| `update_daily_stocks()` | 增量更新：按 stock_meta.last_trade_date 只拉取缺失日期至今天（/api/update_all 使用） |
| `update_all_stocks()` | 按 config 全量拉取并覆盖（可选，当前前端用增量更新） |
| `add_stock_to_config(symbol)` | 将代码追加到 config.stocks |
| `add_stock_and_fetch(symbol)` | 抓取近 5 年 + 写入 DB + 写配置 |

新增接口时可直接调用上述函数，或在此基础上封装新能力（如按日期范围过滤、多股票合并等）。

---

## 七、后期拓展建议

1. **接口**
   - 按日期范围查询：如 `GET /api/data?file=xxx&start=2025-01-01&end=2025-06-01`，在返回前对 `dates` 及对应列做过滤。
   - 每日增量更新：增加 `POST /api/update_daily`，内部调用 `update_daily_stocks()`，供定时任务或前端「每日更新」按钮调用。
   - 列表分页或游标：若股票很多，可对 `/api/list` 加 `page`/`size` 或 `cursor`。

2. **配置**
   - 已有 `GET /api/config`、`PUT /api/config`；可对修改接口做鉴权或环境变量覆盖部分配置。

3. **数据**
   - 更多数据源：在 utils 中抽象「数据源接口」，除 akshare 外接入其他 API，由 config 或参数选择。

4. **安全与部署**
   - 对 POST 写操作（update_all、add_stock、sync_all、remove_stock、config）做频率限制或鉴权。
   - 生产环境使用 Gunicorn/uWSGI + Nginx，关闭 debug。

5. **前端**
   - 列表支持搜索、排序；展示每条数据的最后更新时间（可从 list 或单独接口返回 `last_trade_date`）。
   - 错误码统一为数值，便于前端分支处理。

文档与接口注释会随代码更新，拓展时请同步修改本文档。
