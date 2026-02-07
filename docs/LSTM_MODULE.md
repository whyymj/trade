# LSTM 训练模块 — 架构与功能设计

本文档描述股票预测项目中 **LSTM 训练与预测** 子系统的架构设计、模块职责与功能详细设计，便于开发与维护。

---

## 一、概述

### 1.1 目标

- **输入**：股票日线数据（过去 60 个交易日：收盘、成交量、技术指标）。
- **输出**：未来 5 日价格**方向**（涨/跌，分类）与**涨跌幅**（回归）。
- **能力**：版本化模型、样本外验证与部署决策、定期/性能衰减触发训练、预测回退、监控与告警。

### 1.2 技术栈

- **模型**：PyTorch LSTM 双头（分类 + 回归）。
- **特征**：`analysis.technical`（RSI、MACD、布林、OBV、MFI、波动率、Aroon 等） + 归一化。
- **流程**：scikit-learn 时间序列交叉验证、SHAP 可解释性、本地/MySQL 双写（版本、流水、预测、准确性、失败）。

### 1.3 依赖

- `torch`、`scikit-learn`、`shap`（可选）、`analysis.technical`、`analysis.arima_model`（回退用）。
- MySQL：与 `stock_meta` / `stock_daily` 同库，用于训练流水、当前版本、预测日志、准确性回填、训练失败。

### 1.4 训练与架构规格（供 AI 分析）

- **文档**：`docs/LSTM_TRAINING_SPEC.md`（Markdown）、`docs/LSTM_TRAINING_SPEC.json`（机器可读）。
- **接口**：`GET /api/lstm/training-spec` 返回与 JSON 文件一致的完整规格。
- **生成**：`from analysis.lstm_spec import generate_training_spec_document; generate_training_spec_document()` 可重新生成上述文档。

---

## 二、架构设计

### 2.1 模块划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HTTP API (server/routes/api.py)                    │
│  /api/lstm/train | predict | training-runs | versions | rollback | ...       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    ▼                   ▼               ▼               ▼                   ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ lstm_model  │  │ lstm_       │  │ lstm_        │  │ lstm_       │  │ lstm_       │
│ 训练/预测   │  │ versioning  │  │ triggers     │  │ monitoring  │  │ predict_    │
│ 核心        │  │ 版本/准确性  │  │ 触发与执行   │  │ 状态/告警   │  │ flow/       │
│             │  │             │  │             │  │             │  │ fallback    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │                │
       │                │                │                │                │
       └────────────────┴────────────────┴────────────────┴────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                         ▼
            ┌─────────────┐                           ┌─────────────┐
            │ lstm_       │                           │ data.       │
            │ validation  │                           │ lstm_repo   │
            │ 样本外部署   │                           │ MySQL 仓储  │
            └─────────────┘                           └─────────────┘
```

| 模块 | 路径 | 职责简述 |
|------|------|----------|
| **lstm_model** | `analysis/lstm_model.py` | 特征构建、交叉验证与超参搜索、训练/增量训练、模型保存、SHAP、预测 vs 实际图；入口 `run_lstm_pipeline`、`load_model`、`incremental_train_and_save`。 |
| **lstm_versioning** | `analysis/lstm_versioning.py` | 版本目录管理、当前版本（本地 JSON + MySQL）、预测日志、准确性记录、回填逻辑、`get_recent_accuracy_for_trigger`。 |
| **lstm_validation** | `analysis/lstm_validation.py` | 样本外评估 `evaluate_model_on_holdout`、部署决策 `should_deploy_new_model`（F1/MSE 阈值）。 |
| **lstm_triggers** | `analysis/lstm_triggers.py` | 周/月/季/性能衰减触发判断；`run_triggered_training`（周度增量、月度/季度/衰减完整训练）。 |
| **lstm_monitoring** | `analysis/lstm_monitoring.py` | 性能衰减检测、监控状态汇总、告警检查与发送（含 webhook）。 |
| **lstm_predict_flow** | `analysis/lstm_predict_flow.py` | 模型健康检查、预测后可选异步触发训练。 |
| **lstm_fallback** | `analysis/lstm_fallback.py` | LSTM 不可用时依次回退 ARIMA、技术指标预测。 |
| **lstm_repo** | `data/lstm_repo.py` | 训练流水、当前版本、预测日志、准确性、训练失败的 MySQL 读写。 |

### 2.2 数据流概览

- **训练**：日线 → `build_features_from_df` → 交叉验证/超参 → `train_and_save` → 版本目录 +（可选）样本外验证 → 部署/丢弃 → 写 `lstm_training_run`。
- **预测**：日线 → 特征 → `load_model`（当前版本）→ 前向 → 写 `lstm_prediction_log`；可选回退、可选异步触发训练。
- **准确性**：`update_accuracy_for_symbol`（lstm_versioning）根据「预测日 +5 交易日」实际行情回填 `lstm_accuracy_record`，供性能衰减与监控。
- **版本**：当前版本从 MySQL `lstm_current_version` 读；回滚/部署时仅写 MySQL。

### 2.3 存储（数据库）

- **模型与版本**：存于 MySQL 表 `lstm_model_version`，每行含：
  - `version_id`、`training_time`、`data_start`、`data_end`
  - `metadata_json`（feature_names、seq_len、metrics、n_features、hidden_size 等）
  - `model_blob`（LONGBLOB：PyTorch state_dict 序列化）
- **当前版本**：MySQL `lstm_current_version.id=1` 的 `version_id`。
- **预测/准确性/告警等**：仍为 MySQL 表（lstm_prediction_log、lstm_accuracy_record、lstm_training_run、lstm_training_failure 等）。
- **拟合曲线**：由前端通过 `GET /api/lstm/plot-data?symbol=&years=1|2|3` 拉取数据，ECharts 实时绘制，不再生成或存储 PNG。

---

## 三、功能详细设计

### 3.1 特征与模型

- **序列长度**：`SEQ_LEN = 60`，预测未来 `FORECAST_DAYS = 5`。
- **特征**：`close_norm`、`volume_norm`、`rsi`、`macd_hist`、`bb_position`、`volatility`、`obv_norm`、`mfi`、`aroon_up` 等（见 `DEFAULT_FEATURE_NAMES`）。
- **模型结构**：LSTM → Dropout → 双头（分类 2 类 + 回归 1 维）；输入形状 `(batch, 60, n_features)`。

### 3.2 训练模式

| 模式 | 场景 | 交叉验证 | 样本外验证 | 部署规则 |
|------|------|----------|------------|----------|
| **完整训练** | 手动、月度/季度/性能衰减触发 | 是（5 折或 3 折） | 是（约最近 3 个月留出） | 仅新模型显著优于旧模型时部署 |
| **快速训练** | 手动且 `fast_training=True` | 3 折、单组超参 | 同完整 | 同完整 |
| **增量训练** | 周五周度触发 | 否 | 否 | 直接覆盖当前版本 |

- **完整训练**：`run_lstm_pipeline(..., do_cv_tune=True, do_post_training_validation=True)`；先保存新版本且不设为当前（`promote_to_current=False`），再做留出集评估，通过则 `set_current_version` 并裁剪版本数。
- **增量训练**：`incremental_train_and_save`：加载当前模型，仅用近期数据微调若干 epoch，保存为新版本并设为当前。

### 3.3 超参与交叉验证

- **默认网格**：`lr` ∈ {1e-3, 5e-4}，`hidden_size` ∈ {32, 64}，`epochs` ∈ {30, 50}，5 折 `TimeSeriesSplit`；打分：`score = avg_f1 - 0.1 * log1p(avg_mse)`。
- **快速训练**：`CPU_FRIENDLY_PARAM_GRID`（单组 lr=5e-4、hidden=32、epochs=25），3 折，约 2～8 分钟/次。

### 3.4 训练后验证与部署

- **留出集**：最近约 3 个月（或至少 10 条），`evaluate_model_on_holdout` 得到 `accuracy`、`f1`、`mse`、`mae`、`rmse`、`direction_accuracy`。
- **部署规则**（`should_deploy_new_model`）：  
  - 新模型 F1 提升 ≥ `min_f1_improvement`（默认 0.05），**或**  
  - 新模型 MSE ≤ 旧模型 MSE × `min_mse_ratio`（默认 0.9）。  
  满足任一即部署；否则删除新版本目录，保留旧当前版本。

### 3.5 版本管理

- **保留版本数**：`MAX_VERSIONS = 1`，只保留最新一个版本，新训练完成后删除更旧版本。
- **当前版本**：优先 MySQL `lstm_current_version`，其次本地 `current_version.json`；回滚/部署时双写。
- **回滚**：`set_current_version(version_id)`，要求该版本目录及 `lstm_model.pt` 存在。

### 3.6 训练触发

- **weekly**：每周五；执行**增量训练**（近期约 6 个月数据）。
- **monthly**：当月最后交易日；**完整训练**（配置全量日期，do_cv_tune + 样本外验证）。
- **quarterly**：当季最后交易日（3/6/9/12 月最后交易日）；同 monthly。
- **performance_decay**：最近 20 个交易日平均预测误差 > 历史平均误差 × 1.5；**完整训练**。

优先级：monthly > quarterly > performance_decay > weekly；一次只执行一种。可由 cron 每日调用 `check_triggers` + `run_triggered_training`，或预测接口传 `trigger_train_async=1` 在后台执行。

### 3.7 预测流程

1. 获取日线（config 日期范围）。
2. **模型健康检查**（可选）：无当前版本/文件缺失/超过 N 天未训练则标记不健康。
3. **预测**：若 `use_fallback=1` 则走 `predict_with_fallback`（LSTM → ARIMA → 技术指标）；否则仅 LSTM，失败可 404。
4. **记录**：写入 `lstm_prediction_log`（symbol、predict_date、direction、magnitude、prob_up、model_version_id、source）。
5. **可选**：`trigger_train_async=1` 时后台检查触发并执行训练（不阻塞响应）。

### 3.8 预测回退（lstm_fallback）

- **顺序**：LSTM → ARIMA → 技术指标。
- **ARIMA**：`analysis.arima_model.build_arima_model`，未来 5 日价格 → 方向与涨跌幅。
- **技术指标**：RSI/MACD 简单规则（超卖偏多、超买偏空、MACD 柱正负）+ 近期波动估计幅度。

### 3.9 准确性回填

- **时机**：预测日 + 5 个交易日后可计算实际涨跌幅。
- **逻辑**：`update_accuracy_for_symbol(symbol, as_of_date)`：取待回填预测 → 取日线 → 找预测日及 +5 日收盘 → 算 actual_direction/actual_magnitude、error_magnitude、direction_correct → 写 `lstm_accuracy_record`。
- **用途**：监控 MAE/RMSE/方向准确率、性能衰减检测、告警（低准确率、连续高误差）。

### 3.10 监控状态

- **内容**：当前版本、最后训练时间、数据范围、验证分数、近 7 日预测次数、近期/历史平均误差、MAE/RMSE、方向准确率、训练失败次数、最近一次性能衰减检测结果。
- **数据来源**：版本元数据、`lstm_prediction_log` 统计、`lstm_accuracy_record` 聚合、`lstm_training_failure` 计数、性能检测日志（或实时算一次不落库）。

### 3.11 性能衰减检测

- **条件**：最近 `n_recent` 条准确性记录的平均 `error_magnitude` > 历史平均 × `threshold_multiplier`（默认 1.5）。
- **输出**：triggered、recent_avg_error、historical_avg_error、threshold、n_recent_samples、n_historical_samples、detected_at；可选写入 `performance_detection_log.json`。

### 3.12 告警

- **类型**：performance_decay、no_recent_training、low_accuracy、consecutive_high_error、training_failure_count（≥3 次）。
- **配置**：`config.yaml` 的 `lstm.webhook_url`、`lstm.performance_decay_multiplier`、`lstm.max_days_without_training`、`lstm.min_direction_accuracy`；或环境变量 `LSTM_ALERT_WEBHOOK`。
- **动作**：检查告警列表；若 `fire=true` 则写告警日志并 POST 到 webhook。

### 3.13 训练失败记录

- 训练异常时调用 `record_training_failure(symbol, error_message)`，写 MySQL `lstm_training_failure` 与本地日志。
- 失败次数 ≥ 3 触发告警 `training_failure_count`。

---

## 四、数据层（MySQL）

由 `data.schema.create_lstm_tables()` 创建，与股票库一致。

| 表名 | 说明 |
|------|------|
| **lstm_training_run** | 训练流水：version_id、symbol、training_type、trigger_type、data_start/end、params_json、metrics_json、validation_deployed、validation_reason、holdout_metrics_json、duration_seconds、created_at。 |
| **lstm_current_version** | 当前版本（单行 id=1，version_id）。 |
| **lstm_prediction_log** | 预测记录：symbol、predict_date、direction、magnitude、prob_up、model_version_id、source。 |
| **lstm_accuracy_record** | 准确性回填：symbol、predict_date、actual_date、pred/actual direction&magnitude、error_magnitude、direction_correct。 |
| **lstm_training_failure** | 训练失败：symbol、error_message、created_at。 |

- 当前版本：`get_current_version_id()` 优先读 DB；`set_current_version()` 写本地 + DB。
- 预测/准确性/失败：能写 DB 时均写 DB；监控与性能衰减优先从 DB 读。

---

## 五、配置与 API 索引

### 5.1 配置（config.yaml 可选）

```yaml
lstm:
  webhook_url: "https://..."
  performance_decay_multiplier: 1.5
  max_days_without_training: 30
  min_direction_accuracy: 0.5
```

环境变量：`LSTM_ALERT_WEBHOOK` 覆盖 webhook_url。

### 5.2 HTTP 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/lstm/train` | 训练；Body: symbol, start?, end?, do_cv_tune?, do_shap?, do_plot?, fast_training? |
| GET  | `/api/lstm/predict` | 预测；Query: symbol, use_fallback?, trigger_train_async? |
| GET  | `/api/lstm/plot` | 按股票从数据库返回拟合曲线图（PNG）。Query: symbol（必填）, generate=1 按需生成 |
| GET  | `/api/lstm/training-runs` | 训练流水；Query: symbol?, limit? |
| GET  | `/api/lstm/versions` | 版本列表与当前版本 |
| POST | `/api/lstm/rollback` | 回滚；Body: version_id |
| POST | `/api/lstm/check-triggers` | 检查/执行触发；Body: symbol?, run? |
| POST | `/api/lstm/update-accuracy` | 回填准确性；Body: symbol, as_of_date? |
| GET  | `/api/lstm/monitoring` | 监控状态 |
| GET  | `/api/lstm/performance-decay` | 执行性能衰减检测；Query: threshold?, n_recent?, log? |
| POST | `/api/lstm/alerts` | 检查告警；Body: fire? 发送 webhook |

详细请求/响应见 [API.md](./API.md) 第六节。

---

## 六、前端

- **页面**：`/lstm`（LSTMView.vue），三个 Tab：**训练与预测**、**训练流水与版本**、**监控与告警**。
- **能力**：训练参数（含快速训练）、训练结果与样本外验证、预测（含回退与异步触发）、训练流水查询、版本列表与回滚、更新准确性、检查/执行触发、监控状态、性能衰减检测、告警查看与发送。

### 6.1 如何查看训练拟合曲线与预测走势

- **训练拟合曲线（预测 vs 实际）**  
  在「训练与预测」Tab 底部有卡片 **「训练拟合曲线（按股票，预测 vs 实际）」**。每只股票按 1/2/3 年分别展示，由前端请求 `GET /api/lstm/plot-data?symbol=xxx&years=1|2|3` 获取数据后 ECharts 实时绘制，不生成或存储图片。

- **预测走势（当前预测结果）**  
  同一 Tab 中的 **「预测结果（按股票）」** 表格即预测走势的汇总：每行一只股票，展示 **方向**（涨/跌）、**预测涨跌幅**、**上涨/下跌概率**、**来源**（lstm/arima/technical）。执行「预测全部」或单只「预测」后，该表格会更新；刷新页面后会从接口恢复各股票最近一次预测结果。

### 6.2 价格曲线红蓝线对齐说明（为何可能看起来「错位」）

- **数据含义**  
  - **蓝线（实际价格）**：每个点 = 某个「5 日窗口」**结束日**的真实收盘价。  
  - **红线（预测价格）**：同一结束日由「窗口前一日收盘价 × (1 + 该窗口的预测 5 日涨跌幅)」得到，与蓝线**按同一结束日、同一索引一一对应**。

- **红蓝线在同一图上**  
  后端对每个样本 k 使用同一 `dates_price[k]`（结束日）、同一 `actual_price[k]` 与 `predicted_price[k]`，前端用同一 x 轴绘制，因此**不存在故意的一格错位**。若出现整段横向错位，多为前端传参或 ECharts 系列顺序问题。

- **与上两块图的「错位感」**  
  - 上方「方向」「5日涨跌幅」的横轴用的是**窗口起始日**（`dates` = y_info.index，即预测发生日）。  
  - 下方「价格曲线」的横轴用的是**窗口结束日**（`dates_price`），比同一索引的起始日晚约 5 个交易日。  
  因此**同一索引**在上图是「某日预测」，在下图是「约 5 日后实现」；对比时会有约 5 日的视觉错位，这是设计如此，不是 bug。

---

*文档随代码更新，如有出入以代码为准。*
