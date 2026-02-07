# LSTM 界面更新：波动优化与预测质量诊断

## 1. 训练配置面板

在「训练与预测」页的**训练**卡片内新增**训练配置**区域，采用选项卡：

- **常规**：交叉验证调参、SHAP 可解释性、生成曲线、快速训练（与原有逻辑一致，保持 API 兼容）。
- **波动优化**：
  - **改进训练策略**：勾选后使用 AdamW、余弦退火、早停（波动匹配度）、时间序列数据增强。
  - **回归损失**：可选 `mse`、`huber`、`volatility`、`full`。
  - **一键优化**：对当前选中股票，临时启用「改进训练策略 + 当前选择的回归损失」执行 1/2/3 年训练，结果写入训练流水；不永久修改表单勾选。

请求体兼容性：不传 `use_improved_training` / `reg_loss_type` 时行为与旧版一致；传参后启用对应逻辑。

## 2. 预测质量诊断面板

位于训练表格下方：

- **股票 + 年份**：选择股票与 1/2/3 年模型。
- **加载诊断**：调用 `GET /api/lstm/plot-data?symbol=xxx&years=n`，展示返回的 `diagnostics`（诊断摘要、波动率问题、模型容量等）；风险项高亮显示。
- **新旧配置对比**：折叠块内并排展示「默认配置」与「当前配置」（当前训练配置表单），便于对比与回滚。

## 3. 一键优化

- 入口：波动优化选项卡内的「一键优化（用波动优化配置训练选中）」。
- 行为：对已勾选股票依次执行训练，请求体中固定 `use_improved_training: true`、`reg_loss_type` 为当前选择（默认 `volatility`）；执行结束后恢复表单原勾选状态。
- 记录：每次训练写入训练流水，`params` 中包含 `reg_loss_type`、`use_improved_training`，便于在「训练流水与版本」中查看历史优化尝试与结果。

## 4. 验证与兼容

- **API**：`POST /api/lstm/train`、`POST /api/lstm/train-all` 均支持可选字段 `use_improved_training`、`reg_loss_type`；不传时默认 `false` / `volatility`，与旧客户端兼容。
- **训练流水**：`insert_training_run` 的 `params` 中持久化 `reg_loss_type`、`use_improved_training`，便于对比不同配置下的运行结果。
- **前端**：训练、一键优化均使用同一套 `trainForm`，仅一键优化在调用时临时覆盖上述两项并随后还原。

## 5. 新旧配置对比说明

- **默认配置**：`do_cv_tune: true`，`do_shap: true`，`do_plot: true`，`fast_training: false`，`use_improved_training: false`，`reg_loss_type: 'volatility'`。
- **当前配置**：即当前页面「训练配置」中勾选与选择的值，实时随表单变化；便于在开启波动优化前后对比差异。
