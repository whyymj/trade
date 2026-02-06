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
| GET | `/api/analyze` | 综合分析（时域/频域/ARIMA/复杂度/技术指标）。Query: symbol, start, end（YYYY-MM-DD）。返回 { summary, report_md, charts } |
| GET | `/api/analyze/export` | 同分析参数，返回 Markdown 文件附件（含 YAML 元数据、结构化摘要 JSON、完整报告），便于存档与 AI 解析 |
| GET | `/api/lstm/recommended-range` | 获取 LSTM 训练推荐日期范围。Query: years=1\|2（默认 1），use_config=1 时用 config 范围。返回 { start, end, hint } |
| POST | `/api/lstm/train` | 训练 LSTM 模型（存数据库，只保留最新 1 个版本）。Body: symbol, start, end, do_cv_tune, do_shap。返回 metrics、version_id、plot_path |
| POST | `/api/lstm/train-all` | 一键训练全部股票（以列表中已有数据的股票为准）。Body: start?, end?, years?, do_cv_tune?, do_shap?, do_plot?, fast_training?。返回 results、total、success_count、fail_count |
| GET | `/api/lstm/predict` | 使用当前版本模型预测；自动记录预测日志。Query: symbol |
| POST | `/api/lstm/predict-all` | 对当前全部股票执行预测；Body: use_fallback?, trigger_train_async? |
| GET | `/api/lstm/plot` | 从数据库返回某股票拟合曲线图（PNG）。Query: symbol（必填）, generate=1 按需生成 |
| GET | `/api/lstm/training-runs` | 从 MySQL 查询训练流水（参数、指标、验证结果）。Query: symbol, limit |
| GET | `/api/lstm/versions` | 列出模型版本与当前版本 ID |
| POST | `/api/lstm/rollback` | 回滚到指定版本。Body: version_id |
| POST | `/api/lstm/check-triggers` | 检查/执行训练触发（周五周度、月末完整、性能衰减）。Body: symbol, run |
| POST | `/api/lstm/update-accuracy` | 回填预测准确性（供性能衰减判断）。Body: symbol, as_of_date |
| GET | `/api/lstm/monitoring` | 监控状态汇总（版本、训练时间、准确性、性能衰减检测） |
| GET | `/api/lstm/performance-decay` | 执行一次性能衰减检测。Query: threshold, n_recent, log |
| POST | `/api/lstm/alerts` | 检查并返回告警；fire=true 时写日志并可选 webhook 通知。Body: fire |

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

## 六、LSTM 训练与预测

LSTM 模块使用过去 60 个交易日的收盘价、成交量与技术指标，预测未来 5 日价格方向（涨/跌）与涨跌幅。依赖 PyTorch、scikit-learn、shap（见 `requirements.txt`）；未安装时接口返回 503。

**版本与触发**：每次训练保存为带时间戳的版本（含训练时间、数据范围、验证分数），只保留最新 1 个版本。每次预测会记录到预测日志，便于后续回填准确性；当最近 20 个交易日平均预测误差 > 历史平均误差 × 1.5 时可触发重新训练。支持每周五周度增量训练、每月最后交易日完整重新训练（见 6.5）。

### 6.0 GET /api/lstm/recommended-range

用于调试页自动加载合适训练周期。Query：`years=1` 或 `2`（默认 1，表示最近 N 年）；`use_config=1` 时返回 config 中的 `start_date`/`end_date`（与数据管理配置一致）。响应：`{ "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "hint": "最近 1 年" | "与配置一致" }`。

### 6.1 POST /api/lstm/train

**两种训练模式**：  
- **完整训练**：用于月度/季度，时间序列交叉验证 + 训练后**样本外验证**（保留最近约 3 个月作测试集，新模型仅当显著优于旧模型才部署）。  
- **增量训练**：用于周度，由触发逻辑调用（加载当前模型 + 近期数据微调），不做交叉验证。

训练并保存到数据库 `lstm_model_version` 表（只保留最新 1 个版本）。支持 SHAP、预测 vs 实际曲线图（图存于表 `lstm_plot`，不写本地）。响应中含 `metadata.version_id`、可选 `validation`: { "deployed", "reason", "new_holdout_metrics", "old_holdout_metrics" }。训练失败会记录到 `training_failure_log`，供告警（失败 ≥3 次）使用。

**本机训练参数说明**：默认会做 5 折交叉验证并搜索 lr=[1e-3, 5e-4]、hidden_size=[32, 64]、epochs=[30, 50]，在 **仅 CPU** 上完整跑完可能需 20～60 分钟（视样本量）。若本机无 GPU 或希望缩短时间，可传 **`fast_training: true`**，将改为单组超参（lr=5e-4、hidden=32、epochs=25）、3 折 CV，单次约 2～8 分钟，仍适合本机训练。

**请求**：Body JSON

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `symbol` | string | 是 | 股票代码，如 `600519` |
| `start` | string | 否 | 开始日期 YYYY-MM-DD；缺省用 config 范围 |
| `end` | string | 否 | 结束日期 YYYY-MM-DD；缺省用 config 范围 |
| `do_cv_tune` | bool | 否 | 是否做交叉验证与超参搜索，默认 true |
| `do_shap` | bool | 否 | 是否计算 SHAP，默认 true |
| `do_plot` | bool | 否 | 是否生成预测 vs 实际图，默认 true |
| `fast_training` | bool | 否 | **本机/CPU 友好**：为 true 时用更少超参与 3 折 CV，单次约 2～8 分钟，适合无 GPU 或快速试跑 |

**响应 200**：`{ "symbol", "n_samples", "metrics": { "accuracy", "recall", "f1", "mse" }, "cross_validation", "interpretability": { "feature_importance", "shap_values" 等 }, "plot_path" }`。  
数据不足 65 个交易日时返回 400：`{ "error": "样本不足，需要至少 65 个交易日数据" }`。

### 6.2 GET /api/lstm/predict（每日预测流程）

步骤：获取最新数据 → **模型健康度检查** → 预测 → 记录 → 可选检查训练触发并**异步训练**。

**请求**：Query  
- `symbol=600519`（必填）  
- `use_fallback=1`：LSTM 不可用或异常时依次回退到 **ARIMA**、**技术指标**预测  
- `trigger_train_async=1`：预测完成后在后台检查触发条件并执行训练（不阻塞响应）

**响应 200**：`{ "symbol", "direction", "direction_label", "magnitude", "prob_up", "prob_down", "source": "lstm"|"arima"|"technical", "model_health": { "healthy", "message", "details" } }`。  
未找到模型且未使用回退时返回 404。每次预测会写入预测日志（见 6.6）。

### 6.2.1 POST /api/lstm/predict-all（预测全部）

对当前股票列表中的全部股票依次执行预测（逻辑同 6.2），模型在非回退模式下只加载一次以提速。

**请求**：Body  
- `use_fallback`：bool，默认 false，为 true 时 LSTM 不可用则回退 ARIMA/技术指标  
- `trigger_train_async`：bool，默认 false，为 true 时每只预测后触发异步训练（批量时建议关闭）

**响应 200**：`{ "results": [ { "symbol", "ok", "direction?", "direction_label?", "magnitude?", "prob_up?", "prob_down?", "source?", "error?" }, ... ], "success_count", "fail_count" }`。每只成功预测会写入预测日志。

### 6.3 GET /api/lstm/versions

列出 LSTM 模型版本（最近 5 个）。每项含 `version_id`、`training_time`、`data_start`、`data_end`、`validation_score`、`metrics`。

**响应 200**：`{ "current_version_id": "20250205_143022", "versions": [ ... ] }`。

### 6.4 POST /api/lstm/rollback

回滚到指定版本。

**请求**：Body `{ "version_id": "20250205_143022" }`。

**响应 200**：`{ "ok": true, "current_version_id": "20250205_143022" }`。版本不存在返回 404。

### 6.5 POST /api/lstm/check-triggers

检查训练触发条件；若 Body 传 `"run": true` 且满足条件则执行对应训练。

**触发规则**（优先级从高到低）：  
- **monthly**：当月最后交易日 → **完整训练**（交叉验证 + 样本外验证，仅更优则部署）  
- **quarterly**：当季最后交易日（3/6/9/12 月最后交易日）→ 完整训练  
- **performance_decay**：最近 20 日平均预测误差 > 历史平均 × 1.5 → 完整训练  
- **weekly**：周五 → **增量训练**（加载当前模型 + 近期数据微调）

**请求**：Body `{ "symbol": "600519", "run": true }`。`symbol` 可省略则用配置中第一只股票。

**响应 200**：`{ "triggers": { "weekly", "monthly", "quarterly", "performance_decay", ... }, "training": { ... } }`。建议由 cron 每日调用，`run=true` 时自动执行满足条件的训练。

### 6.6 POST /api/lstm/update-accuracy

根据实际行情回填预测准确性（预测日 + 5 个交易日后可计算）。用于积累误差记录，供性能衰减触发使用。

**请求**：Body `{ "symbol": "600519", "as_of_date": "2025-02-05" }`。`as_of_date` 可选，默认今天。

**响应 200**：`{ "ok": true, "symbol": "600519", "updated_count": 3 }`。

### 6.7 GET /api/lstm/monitoring

返回监控状态汇总，包括：  
- 当前版本、最后训练时间、数据范围、验证分数  
- 近 7 日预测次数  
- **MAE / RMSE**、方向准确率、**训练失败次数**  
- 最近/历史平均预测误差、性能衰减检测结果  

**响应 200**：`{ "current_version_id", "last_training_time", "data_start", "data_end", "validation_score", "metrics", "prediction_count_7d", "accuracy_recent_avg_error", "accuracy_historical_avg_error", "mae", "rmse", "direction_accuracy", "training_failure_count", "performance_decay": { ... } }`。

### 6.8 GET /api/lstm/performance-decay

执行一次**性能衰减检测**并返回报告，可选写入检测历史（供趋势查看）。

**请求**：Query `threshold=1.5`（倍数）、`n_recent=20`（最近 N 条）、`log=1`（是否写入检测日志）。

**响应 200**：`{ "triggered", "recent_avg_error", "historical_avg_error", "threshold", "n_recent_samples", "n_historical_samples", "detected_at" }`。

### 6.9 POST /api/lstm/alerts

检查告警条件并返回告警列表；若 Body 传 `"fire": true` 且配置了 webhook，则发送 HTTP POST 通知。

**告警类型**：`performance_decay`（性能衰减）、`no_recent_training`（超过 N 天未训练）、`low_accuracy`（近期方向正确率低于阈值）、`consecutive_high_error`（连续 5 天预测误差超过阈值）、`training_failure_count`（训练失败累计 ≥3 次）。

**请求**：Body `{ "fire": true }`。不传 `fire` 或 `fire=false` 时仅返回告警列表，不发送 webhook。

**响应 200**：`{ "alerts": [ { "type", "message", "severity", "at", "detail" }, ... ], "count": n, "fired": { "logged", "webhook_sent", "webhook_errors" } }`（当 `fire=true` 且有告警时含 `fired`）。

**告警配置**（可选）：在 `config.yaml` 中增加 `lstm` 段，或使用环境变量。

```yaml
# config.yaml 示例
lstm:
  webhook_url: "https://your-server.com/webhook/lstm-alert"   # 告警时 POST JSON
  performance_decay_multiplier: 1.5
  max_days_without_training: 30
  min_direction_accuracy: 0.5   # 可选，近期方向正确率低于 50% 告警
```

环境变量 `LSTM_ALERT_WEBHOOK` 可覆盖 `webhook_url`。

### 6.10 LSTM 训练数据与 MySQL

本地训练产生的**必要参数与流水**会写入当前项目使用的 **MySQL**（与 `stock_meta` / `stock_daily` 同库）。启动时 `create_all_tables()` 会创建以下 LSTM 表（若不存在）：

| 表名 | 说明 |
|------|------|
| `lstm_training_run` | 训练流水：每次训练（完整/增量）记录 version_id、symbol、training_type、trigger_type、data_start/end、params_json、metrics_json、validation_deployed、validation_reason、holdout_metrics_json、duration_seconds |
| `lstm_current_version` | 当前使用的模型版本（单行 id=1，version_id） |
| `lstm_model_version` | 模型版本本体：version_id、training_time、data_start/end、metadata_json、model_blob(LONGBLOB)，替代原 analysis_temp/lstm/versions 目录 |
| `lstm_prediction_log` | 预测记录：symbol、predict_date、direction、magnitude、prob_up、model_version_id、source(lstm/arima/technical) |
| `lstm_accuracy_record` | 准确性回填：symbol、predict_date、actual_date、pred/actual direction&magnitude、error_magnitude、direction_correct |
| `lstm_training_failure` | 训练失败记录：symbol、error_message，用于告警（失败≥3 次） |

- **当前版本**：`get_current_version_id()` 从 `lstm_current_version` 读取；`set_current_version()` 写入该表。模型权重与元数据存于 `lstm_model_version`（不再使用本地 analysis_temp/lstm/versions 目录）。
- **预测与准确性**：`record_prediction` 会写入 `lstm_prediction_log`；回填准确性时写入 `lstm_accuracy_record`。监控与性能衰减检测优先从 MySQL 读取上述记录。
- **训练流水**：每次成功训练（含手动训练与触发训练）会插入 `lstm_training_run`。查询接口：**GET /api/lstm/training-runs**（Query: symbol, limit）。

**GET /api/lstm/training-runs**：从 MySQL 查询训练流水。Query `symbol=600519`（可选）、`limit=50`。响应 `{ "runs": [ { "id", "version_id", "symbol", "training_type", "trigger_type", "data_start", "data_end", "params", "metrics", "validation_deployed", "validation_reason", "holdout_metrics", "duration_seconds", "created_at" }, ... ], "count": n }`。

---

## 七、server 包工具函数（拓展参考）

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

## 八、后期拓展建议

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
