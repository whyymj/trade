# 集成学习多因子预测 — 实现细节

## 1. 概述

在 `analysis` 包中实现了基于集成学习的多因子预测流水线，包含：

- **因子库**：200+ 量化因子（动量、波动率、价值、质量、技术、流动性、情绪代理）
- **三个基模型**：XGBoost、LightGBM、随机森林
- **自动特征选择**：递归特征消除（RFECV）
- **过拟合预防**：早停（early stopping）、L1/L2 正则化
- **集成策略**：加权平均 + 最优权重分配；可选 Stacking
- **因子绩效报告**：IC、Rank IC、换手率、分组收益、多空收益

---

## 2. 因子库（`analysis/factor_library.py`）

### 2.1 入口函数

- **`build_factor_library(df, ...)`**  
  从日线 OHLCV DataFrame 构建因子面板。  
  要求列：`收盘`/close、`成交量`/volume、`最高`/high、`最低`/low、`开盘`/open；可选：`成交额`、`换手率`、`涨跌幅`。  
  返回与 `df` 同索引的 DataFrame，每列为一个因子。

### 2.2 因子类别与数量（约 200+）

| 类别 | 内容 | 示例 |
|------|------|------|
| **动量** | ROC、动量收益、均线偏离、RSI 多周期、MACD 多组、Aroon | `momentum_roc_5/10/20/40/60`, `rsi_6/12/14/24`, `macd_hist`, `aroon_up/down` |
| **波动率** | 滚动波动率、ATR、布林带宽度/位置 | `volatility_5/10/20/40/60`, `atr_7/14/28`, `bb_position_10/20/30` |
| **成交量/流动性** | OBV、量比、MFI | `obv`, `obv_ratio_5/10/20`, `volume_ratio_*`, `mfi_10/14/20` |
| **价值/质量代理** | 价量比、收益波动率（无基本面时代理） | `pv_ratio_20/40/60`, `return_vol_20/40/60` |
| **情绪代理** | 近期收益、成交量偏离（无舆情时的代理） | `sentiment_ret_5/10/20`, `sentiment_vol_5/10/20` |
| **价格形态** | 与均线关系、高低点位置、振幅 | `close_to_ma_5/10/20/60`, `close_high_20`, `range_hl`, `range_oc` |
| **其他** | 涨跌幅、累计涨跌幅 | `change_pct`, `change_pct_cum_5/10/20` |

可配置参数：`include_rsi_periods`, `include_roc_periods`, `include_vol_windows`, `include_bb_periods`, `include_ma_periods`, `include_atr_periods`, `include_volume_windows`，用于控制周期与窗口，从而扩展或收缩因子数量。

### 2.3 辅助接口

- **`get_factor_names_by_category()`**：按类别返回因子名列表（与默认 `build_factor_library` 输出一致，用于报告分类）。
- **`get_all_factor_names(factor_df)`**：从已构建的因子 DataFrame 得到全部列名。

---

## 3. 集成模型（`analysis/ensemble_models.py`）

### 3.1 数据构建

- **`build_xy_from_factors(factor_df, forward_return, task=..., threshold=0)`**  
  - `task="classification"`：y = (forward_return > threshold)；`task="regression"`：y = forward_return。  
  - 返回 `X, y, feature_names, index`。

### 3.2 三个基模型

| 模型 | 函数 | 用途 | 过拟合预防 |
|------|------|------|------------|
| **XGBoost** | `train_xgb(..., early_stopping_rounds=30)` | 技术+基本面+情绪因子 | 早停、`reg_alpha`/`reg_lambda` |
| **LightGBM** | `train_lgb(..., early_stopping_rounds=30)` | 高效处理大量特征 | 早停、`reg_alpha`/`reg_lambda` |
| **随机森林** | `train_rf(...)` | 特征重要性分析 | `max_depth`、`min_samples_leaf`、`max_features='sqrt'` |

- 默认超参见 `DEFAULT_XGB_PARAMS`、`DEFAULT_LGB_PARAMS`、`DEFAULT_RF_PARAMS`（均含正则或树约束）。
- XGB/LGB 支持传入 `X_val, y_val` 做早停；未传则仅用训练集。

### 3.3 自动特征选择（递归特征消除）

- **`run_rfecv(X, y, feature_names, task=..., min_features_to_select=10, cv_splits=5, step=0.1, scoring=...)`**  
  - 使用时序交叉验证（`TimeSeriesSplit`），内部用随机森林做 RFECV。  
  - 返回：`(selected_feature_names, fitted_rfecv, estimator)`。

### 3.4 集成策略

- **加权平均**  
  - **`ensemble_weighted_predict(predictions_list, weights)`**：多模型预测概率加权平均，`weights` 建议和为 1。

- **最优权重分配**  
  - **`optimize_ensemble_weights(predictions_list, y_true, metric='auc', method='grid'|'scipy', n_grid=21)`**  
  - 在验证集上优化组合权重：`metric='auc'` 或 `'accuracy'`；`method='grid'` 网格搜索或 `'scipy'` 用 SLSQP 约束优化（sum(w)=1, w≥0）。  
  - 返回 `(best_weights, best_score)`。

- **Stacking**  
  - **`train_stacking(X_train, y_train, base_models, meta_model='logistic', cv_splits=5)`**：两层 Stacking，base 在每折上产生 OOF 预测作为 meta 特征，再训练 meta（逻辑回归或 Ridge）。  
  - **`predict_stacking(meta_model, base_models, X)`**：用各 base 预测拼成 meta 特征后由 meta 预测。

### 3.5 一站式流水线

- **`run_ensemble_pipeline(factor_df, forward_return, task='classification', train_ratio=0.7, use_rfe=True, rfe_min_features=15, rfe_cv=3, xgb_params=..., lgb_params=..., rf_params=..., early_stopping_rounds=30, ensemble_method='weighted', optimize_weights=True)`**  
  - 步骤：`build_xy_from_factors` →（可选）RFECV 特征选择 → 按时间划分训练/验证 → 训练 XGB/LGB/RF → 验证集预测 → 权重优化（默认 grid）→ 加权集成。  
  - 返回字典：`models`、`feature_names`、`ensemble_weights`、`val_auc`、`val_accuracy`、`val_rmse`、`rf_feature_importance`、`xgb_evals`/`lgb_evals` 等。

- **`save_ensemble_artifacts(result, out_dir)`**：保存 `ensemble_weights.json`、`rf_feature_importance.csv` 等。

---

## 4. 因子绩效分析报告（`analysis/ensemble_report.py`）

### 4.1 指标

- **IC**：因子与未来收益的 Pearson 相关系数（`calc_ic`）。
- **Rank IC**：Spearman 秩相关（`calc_rank_ic`）。
- **滚动 IC 统计**：`calc_ic_series(factor_df, forward_return, rolling_window=20)` 得到各因子滚动 IC/Rank IC 的均值、标准差、IR（IC/std）。
- **换手率**：`calc_turnover(factor_df, top_pct=0.2, rolling_window=1)`，前 `top_pct` 分位组合的日度换手。
- **分组收益**：`calc_group_returns(factor, forward_return, n_groups=5)`，按因子五分位后的组均收益。
- **多空收益**：最高组减最低组（在报告内按因子汇总）。

### 4.2 报告生成与导出

- **`generate_factor_performance_report(factor_df, forward_return, rolling_window=20, top_pct=0.2, n_groups=5)`**  
  返回字典：`summary`、`ic_full`、`rank_ic_full`、`ic_rolling_stats`、`group_returns`、`long_short_return`、`turnover_sample_mean`、`top_ic_factors`、`top_rank_ic_factors`。

- **`report_to_markdown(report)`**：将上述报告转为 Markdown 文本，便于保存或展示。

---

## 5. 依赖

- **requirements.txt** 新增：`xgboost>=2.0.0`、`lightgbm>=4.0.0`。  
- 已有：`scikit-learn`、`scipy`、`pandas`、`numpy`。

---

## 6. 使用示例

```python
import pandas as pd
from analysis.factor_library import build_factor_library, get_all_factor_names
from analysis.ensemble_models import run_ensemble_pipeline, save_ensemble_artifacts
from analysis.ensemble_report import generate_factor_performance_report, report_to_markdown

# 1) 准备日线 df 与未来收益
# df 列：收盘、成交量、最高、最低、开盘（及可选 成交额、换手率、涨跌幅）
forward_return = df["收盘"].pct_change(5).shift(-5)  # 未来 5 日收益

# 2) 构建因子库
factor_df = build_factor_library(df)
print("因子数:", len(get_all_factor_names(factor_df)))

# 3) 因子绩效报告
report = generate_factor_performance_report(factor_df, forward_return.dropna(), rolling_window=20)
print(report_to_markdown(report))

# 4) 集成流水线（含 RFE、早停、权重优化）
result = run_ensemble_pipeline(
    factor_df, forward_return,
    task="classification",
    use_rfe=True, rfe_min_features=15, rfe_cv=3,
    early_stopping_rounds=30,
    optimize_weights=True,
)
print("Val AUC:", result["val_auc"], "Weights:", result["ensemble_weights"])

# 5) 保存权重与 RF 重要性
save_ensemble_artifacts(result, "analysis_temp/ensemble")
```

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `analysis/factor_library.py` | 因子库构建、因子名按类别查询 |
| `analysis/ensemble_models.py` | XGB/LGB/RF、RFE、加权/Stacking、权重优化、流水线 |
| `analysis/ensemble_report.py` | IC、Rank IC、换手率、分组收益、报告生成与 Markdown 导出 |
| `docs/ENSEMBLE_MULTIFACTOR_IMPLEMENTATION.md` | 本文档 |

导出在 `analysis/__init__.py` 中：`build_factor_library`、`get_factor_names_by_category`、`get_all_factor_names`、`run_ensemble_pipeline`、`save_ensemble_artifacts`、`generate_factor_performance_report`、`report_to_markdown` 等按需加入 `__all__`。
