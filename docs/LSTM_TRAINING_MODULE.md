# LSTM 训练功能模块 — 详细文档

本文档描述 LSTM **训练功能**的完整流程、入口函数、训练模式、触发机制与 HTTP 接口，便于开发、排查与 AI 分析。

---

## 一、概述

### 1.1 训练功能目标

- **输入**：股票日线 DataFrame（需含至少 65 个交易日：60 日序列 + 5 日用于构造目标）。
- **输出**：训练好的 LSTM 模型以**版本**形式持久化（当前为 MySQL `lstm_model_version`），并可选设为**当前版本**供预测使用。
- **维度**：按 `(symbol, years)` 区分模型，`years` 取 1、2、3 表示使用最近 1/2/3 年数据训练的模型。

### 1.2 代码入口与依赖

| 入口 | 所在模块 | 说明 |
|------|----------|------|
| `run_lstm_pipeline` | `analysis/lstm_model.py` | 端到端：特征 → 可选交叉验证 → 训练保存 → 可选样本外验证与部署 → 可选 SHAP |
| `train_and_save` | `analysis/lstm_model.py` | 在给定 (X, y_dir, y_mag) 上训练并保存版本 |
| `incremental_train_and_save` | `analysis/lstm_model.py` | 加载当前模型，仅用新数据微调若干 epoch，保存为新版本 |
| `cross_validate_and_tune` | `analysis/lstm_model.py` | 时间序列交叉验证 + 超参搜索，返回最佳超参 |
| `改进的训练策略` | `analysis/lstm_training.py` | AdamW + 余弦退火 + 早停（波动匹配度）+ 数据增强 |
| `check_triggers` / `run_triggered_training` | `analysis/lstm_triggers.py` | 检查并执行周/月/季/性能衰减触发训练 |

训练依赖：`torch`、`scikit-learn`；可选 `shap`。版本与流水写入 MySQL（`data/lstm_repo.py`）。

---

## 二、端到端训练流程：run_lstm_pipeline

### 2.1 函数签名与默认值

```python
def run_lstm_pipeline(
    df: pd.DataFrame,
    symbol: str = "",
    save_dir: Optional[os.PathLike | str] = None,
    do_cv_tune: bool = True,
    do_shap: bool = True,
    do_plot: bool = True,
    param_grid: Optional[dict] = None,
    do_post_training_validation: bool = True,
    fast_training: bool = False,
    years: int = 1,
    use_enhanced_model: bool = True,
    reg_loss_type: str = "full",
    use_improved_training: bool = True,
) -> dict
```

### 2.2 流程步骤

1. **特征构建**  
   `build_features_from_df(df)` → 得到 `(X, feature_names, y_info, y_dir, y_mag)`。  
   - 若样本数不足（`len(X) == 0`），直接返回 `{"error": "样本不足，需要至少 65 个交易日数据"}`。

2. **交叉验证与超参（可选）**  
   - 当 `do_cv_tune=True` 且 **未**使用改进训练（`use_improved_training=False`）时：  
     - 使用 `TimeSeriesSplit(n_splits=5)`（快速训练时为 3 折）和 `param_grid`（默认或 `CPU_FRIENDLY_PARAM_GRID`）做网格搜索。  
     - 打分：`score = avg_f1 - 0.1 * log1p(avg_mse)`，选出最佳 `lr`、`hidden_size`、`epochs`。  
   - 当 `use_improved_training=True` 时：**跳过交叉验证**，固定使用 `lr=5e-4, hidden_size=128（增强）/64（基础）, epochs=50`，以缩短单次训练时间。

3. **训练并保存**  
   `train_and_save(...)`：  
   - 使用增强模型（默认）或基础模型、改进训练（默认）或普通 Adam+train_epoch。  
   - 版本写入 MySQL（`save_versioned_model`）；若 `promote_to_current=False`（即开启了训练后验证），新版本**先不**设为当前。

4. **训练后验证与部署（可选）**  
   - 当 `do_post_training_validation=True` 时：  
     - 留出集：最近约 3 个月（或至少 10 条、最多 66 条）。  
     - 在新、旧模型上分别计算留出集指标（accuracy、f1、mse、mae、rmse、direction_accuracy）。  
     - `should_deploy_new_model(new_holdout_metrics, old_holdout_metrics)`：  
       - 若新模型 **F1 提升 ≥ 0.05** 或 **MSE ≤ 旧模型 × 0.9**，则部署（`set_current_version` + 裁剪版本）；  
       - 否则删除新版本（`remove_version`），保留旧当前版本。  
   - 若未做训练后验证，则新训练完的版本直接设为当前（`promote_to_current=True`）。

5. **可解释性（可选）**  
   `do_shap=True` 时调用 `compute_feature_importance_and_shap`，结果写入返回的 `interpretability`。

### 2.3 返回值结构

成功时返回字典包含（部分）：  
`symbol`、`n_samples`、`n_features`、`feature_names`、`data_start`、`data_end`、`cross_validation`（若做了 CV）、`metrics`、`metadata`（含 `version_id`、`diagnostics` 等）、`diagnostics`、`validation`（若做了训练后验证）、`interpretability`（若做了 SHAP）。

---

## 三、核心函数详解

### 3.1 build_features_from_df

- **文件**：`analysis/lstm_model.py`  
- **输入**：日线 DataFrame，列支持中文（收盘、成交量、最高、最低、开盘）或英文（close, volume, high, low, open）。  
- **输出**：  
  - `X`: shape `(n_samples, 60, n_features)`，float32；  
  - `feature_names`: 基础特征 + 波动增强特征名；  
  - `y_info`: 每样本的 direction、magnitude、end_date 等；  
  - `y_direction`: 0/1，未来 5 日累计涨跌方向；  
  - `y_magnitude`: shape `(n_samples, 5)`，未来 5 日逐日收益率。  
- **最少样本**：需要 `SEQ_LEN + FORECAST_DAYS = 65` 个交易日，否则返回空数组。

### 3.2 cross_validate_and_tune

- **文件**：`analysis/lstm_model.py`  
- **作用**：在给定 `(X, y_direction, y_magnitude)` 上做时间序列交叉验证，对 `lr`、`hidden_size`、`epochs` 网格搜索。  
- **默认网格**：`lr in [1e-3, 5e-4]`，`hidden_size in [32, 64]`，`epochs in [30, 50]`，`batch_size=32`。  
- **CV**：`TimeSeriesSplit(n_splits)`，每折训练基础 `LSTMDualHead`（非增强），用 `train_epoch` + 可选 `reg_loss_type`。  
- **打分**：`score = avg_f1 - 0.1 * log1p(avg_mse)`，取分数最高的一组超参。  
- **返回**：`cv_results`（每组合每折指标）、`best_params`、`best_score`、`n_splits`。

### 3.3 train_and_save

- **文件**：`analysis/lstm_model.py`  
- **作用**：在**全部**提供的 (X, y_dir, y_mag) 上训练一个模型并保存为版本。  
- **关键参数**：  
  - `use_enhanced_model=True`：使用 `LSTMDualHeadEnhanced`（双向 LSTM + 注意力 + 跳跃连接）；否则使用 `LSTMDualHead`。  
  - `use_improved_training=True`：使用 `lstm_training.改进的训练策略`（见下）；否则使用 Adam + 多 epoch `train_epoch`。  
  - `reg_loss_type`：`"mse"` / `"huber"` / `"volatility"` / `"full"`，回归头损失。  
  - `promote_to_current`：是否将本版本设为当前版本（通常由 pipeline 根据是否做训练后验证决定）。  
  - `symbol`、`years`：写入版本元数据，用于按股票与年份区分当前版本。  
- **改进训练时**：  
  - 按时间顺序划分 80% 训练、20% 验证（至少 20 条验证）；  
  - 使用 `DEFAULT_IMPROVED_CONFIG`（可被传入的 `配置` 覆盖），含 AdamW、CosineAnnealingWarmRestarts、早停（验证集波动匹配度）、可选数据增强；  
  - 回归损失由 `reg_loss_type` 决定（如 `"full"` 对应改进的损失函数）。  
- **保存**：通过 `save_versioned_model` 写入 MySQL（state_dict + metadata）；本地仅保留最新 1 个版本（`MAX_VERSIONS=1`）。  
- **返回**：`(model, metadata, final_metrics)`。

### 3.4 incremental_train_and_save

- **文件**：`analysis/lstm_model.py`  
- **作用**：加载当前模型（`load_model(save_dir, device)`），仅用传入的 `df` 构建特征后的数据微调若干 epoch，保存为**新版本**并设为当前。  
- **典型用途**：周度触发（周五）时快速更新，不跑交叉验证、不做训练后验证。  
- **参数**：`epochs=15`，`lr=1e-4`，`batch_size=32`，`reg_loss_type="full"`。  
- **注意**：必须先存在当前版本，否则返回 `{"error": "无现有模型可做增量训练: ..."}`。

---

## 四、训练模式对比

| 模式 | 交叉验证 | 样本外验证 | 部署规则 | 典型场景 |
|------|----------|------------|----------|----------|
| 完整训练（do_cv_tune + 验证） | 是（5 折或 3 折） | 是（约最近 3 个月） | 仅新模型显著优于旧模型时部署 | 手动、月度/季度/性能衰减触发 |
| 快速训练（fast_training=True） | 3 折 + 单组超参 | 同完整 | 同完整 | 批量或 CPU 本机快速跑通 |
| 改进训练（use_improved_training=True） | 否（固定超参） | 可开可关 | 同完整（若 do_post_training_validation=True） | 默认推荐，单次训练更快 |
| 增量训练 | 否 | 否 | 直接覆盖当前版本 | 周五周度触发 |

---

## 五、改进训练策略（lstm_training）

### 5.1 默认配置 DEFAULT_IMPROVED_CONFIG

```python
{
    "初始学习率": 0.001,
    "权重衰减": 0.01,
    "梯度裁剪": 1.0,
    "早停耐心": 20,
    "最小学习率": 1e-5,
    "T_0": 50,
    "T_mult": 2,
    "使用数据增强": True,
    "noise_std": 0.01,
    "scale_low": 0.98,
    "scale_high": 1.02,
    "reg_loss_type": "full",
    "weight_cls": 1.0,
    "weight_reg": 1.0,
    "max_epochs": 60,
}
```

### 5.2 组件说明

- **优化器**：AdamW，`lr=初始学习率`，`weight_decay=权重衰减`。  
- **学习率调度**：CosineAnnealingWarmRestarts，`T_0=50`，`T_mult=2`，`eta_min=最小学习率`。  
- **早停**：以验证集上的**波动匹配度**为指标（预测幅度与真实幅度的 std 越接近越好），连续 `早停耐心` 个 epoch 未提升则停止，并恢复最佳 state_dict。  
- **数据增强**：每个 batch 对输入做高斯噪声（`noise_std`）与逐样本缩放（`scale_low`～`scale_high`），目标不变。  
- **损失**：分类 CrossEntropyLoss；回归由 `reg_loss_type` 决定（如 `"full"` 为改进的损失函数：MSE + 波动/每步波动/方向/序列内方向/分位数）。

---

## 六、训练后验证与部署（lstm_validation）

- **留出集**：`n_holdout = min(66, max(10, len(X)//3))`，取最近一段作为测试。  
- **评估**：`evaluate_model_on_holdout(model, X_holdout, y_dir, y_mag, device)` → accuracy、f1、mse、mae、rmse、direction_accuracy。  
- **部署规则**（`should_deploy_new_model`）：  
  - 新模型 **F1 提升 ≥ 0.05**，**或**  
  - 新模型 **MSE ≤ 旧模型 MSE × 0.9**。  
  满足其一则部署；否则删除新版本，保留旧当前版本。

---

## 七、触发训练（lstm_triggers）

### 7.1 触发类型与优先级

| 触发 | 条件 | 执行内容 | 优先级 |
|------|------|----------|--------|
| weekly | 每周五 | 增量训练（近期约 6 个月数据） | 最低 |
| monthly | 当月最后交易日 | 完整训练（do_cv_tune + 训练后验证） | 高 |
| quarterly | 当季最后交易日（3/6/9/12 月最后工作日） | 同 monthly | 高 |
| performance_decay | 最近 20 日平均预测误差 > 历史平均 × 1.5 | 完整训练 | 中 |

一次只执行一种；优先级一般为 monthly > quarterly > performance_decay > weekly。

### 7.2 check_triggers

- **作用**：仅检查是否满足上述条件，不执行训练。  
- **返回**：`{ "weekly", "monthly", "quarterly", "performance_decay", "performance_recent_avg", "performance_historical_avg" }`。

### 7.3 run_triggered_training

- **参数**：`symbol`、`trigger_type`（上述之一）、`fetch_hist_fn`、`get_date_range_fn`、`run_lstm_pipeline_fn`、`incremental_train_fn`（可选）、`full_range_months_weekly=6`、`save_dir`。  
- **weekly**：拉取近期约 6 个月数据；若有 `incremental_train_fn` 则调用增量训练，否则调用 `run_lstm_pipeline(..., do_cv_tune=False, do_shap=False, do_plot=True)`。  
- **monthly / quarterly / performance_decay**：用配置的完整日期范围拉数据，调用 `run_lstm_pipeline(..., do_cv_tune=True, do_shap=True, do_plot=True, do_post_training_validation=True)`。

---

## 八、HTTP 接口

### 8.1 POST /api/lstm/train

- **用途**：对单只股票进行训练。  
- **请求体**（JSON）：  
  - `symbol`（必填）：股票代码。  
  - `all_years`（可选）：为 `true` 时依次训练 1、2、3 年三个模型（推荐）；否则按单年训练。  
  - 单年时可传 `start`、`end`（YYYY-MM-DD）或 `years`（1/2/3）；不传则使用配置的日期范围。  
  - `do_cv_tune`、`do_shap`、`do_plot`、`fast_training`、`use_improved_training`、`reg_loss_type`、`param_grid` 等与 `run_lstm_pipeline` 对齐。  
- **行为**：  
  - 同一 `symbol` 并发训练会排队等待（有超时）。  
  - 成功时写入训练流水（`lstm_training_run`），失败时可选写入 `lstm_training_failure`。  
- **响应**：成功返回 pipeline 的完整结果（含 `metadata.version_id`、`validation` 等）；`all_years` 时返回**最后一轮（3 年）**的结果供前端展示。

### 8.2 POST /api/lstm/train-all

- **用途**：一键训练当前列表中的全部股票。  
- **请求体**：`start`、`end` 或 `years`（默认 1）；`do_cv_tune`、`do_shap`、`do_plot`、`fast_training`（默认 true）、`use_improved_training`、`reg_loss_type`。  
- **行为**：对每只股票依次获取数据并调用 `run_lstm_pipeline`；每只股票同样受训练锁与超时约束。  
- **响应**：`{ "results": [ { "symbol", "ok", "version_id"? | "error"? }, ... ], "total", "success_count", "fail_count" }`。

---

## 九、版本与存储

- **版本存储**：MySQL 表 `lstm_model_version`，含 `version_id`、`symbol`、`years`、`training_time`、`data_start`、`data_end`、`metadata_json`、`model_blob`（state_dict 序列化）。  
- **当前版本**：按 `(symbol, years)` 维度的当前版本在 `lstm_current_version`（或等价逻辑）中维护；预测时按 `symbol` + `years` 加载对应当前版本。  
- **保留策略**：当前为只保留最新 1 个版本（`MAX_VERSIONS=1`），新训练完成后会裁剪旧版本。

---

## 十、错误与边界

- **样本不足**：`build_features_from_df` 发现不足 65 个交易日时返回空数组，pipeline 返回 `{"error": "样本不足，需要至少 65 个交易日数据"}`。  
- **训练失败**：API 层在失败时可调用 `record_training_failure(symbol, error_message, save_dir)` 写入 `lstm_training_failure`，便于监控与排查。  
- **并发**：同一 `symbol` 同时只能有一个训练任务，通过内存锁 + 超时（如 2 小时）避免长时间占用；超时后锁会自动释放。

---

## 十一、相关文档与接口

- **训练与架构规格（参数/结构）**：`docs/LSTM_TRAINING_SPEC.md`、`docs/LSTM_TRAINING_SPEC.json`；接口 `GET /api/lstm/training-spec`。  
- **模块架构总览**：`docs/LSTM_MODULE.md`。  
- **API 列表**：`docs/API.md`。

以上为 LSTM 训练功能模块的详细说明，可直接用于开发、运维与 AI 分析。
