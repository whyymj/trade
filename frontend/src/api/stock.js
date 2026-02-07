const API_BASE = ''

/** 带超时的 fetch，用于长时间请求（如 LSTM 训练）。timeoutMs 不传则不限时。 */
async function request(url, options = {}) {
  const { timeoutMs, ...fetchOptions } = options
  let controller = null
  if (timeoutMs != null && timeoutMs > 0) {
    controller = new AbortController()
    fetchOptions.signal = controller.signal
    setTimeout(() => controller.abort(), timeoutMs)
  }
  const res = await fetch(API_BASE + url, {
    ...fetchOptions,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = new Error(data.error || data.message || '请求失败')
    err.data = data
    throw err
  }
  return data
}

export function apiList() {
  return request('/api/list')
}

/** 获取股票日线数据，可选 { start, end } 为 YYYY-MM-DD 指定时间范围。 */
export function apiData(filename, range) {
  const params = new URLSearchParams()
  params.set('file', filename)
  if (range?.start) params.set('start', range.start)
  if (range?.end) params.set('end', range.end)
  return request('/api/data?' + params.toString())
}

export function apiAddStock(code) {
  return request('/api/add_stock', {
    method: 'POST',
    body: JSON.stringify({ code }),
  })
}

/** 一键更新全部股票日线。options: { fromLastUpdate: true } | { months: 1 } | { years: 3|5|10 }。 */
export function apiUpdateAll(options = { fromLastUpdate: true }) {
  let body
  if (options.fromLastUpdate) {
    body = { fromLastUpdate: true }
  } else if (options.months != null) {
    body = { months: options.months }
  } else {
    body = { years: options.years ?? 5 }
  }
  return request('/api/update_all', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

// 数据管理（config、全量同步、移除股票）
export function apiGetConfig() {
  return request('/api/config')
}

export function apiUpdateConfig(body) {
  return request('/api/config', {
    method: 'PUT',
    body: JSON.stringify(body || {}),
  })
}

export function apiSyncAll() {
  return request('/api/sync_all', { method: 'POST' })
}

export function apiRemoveStock(code) {
  return request('/api/remove_stock', {
    method: 'POST',
    body: JSON.stringify({ code }),
  })
}

/** 更新股票名称与说明。name、remark 可选。 */
export function apiUpdateStockRemark(symbol, remark, name) {
  const body = { symbol, remark: remark ?? '' }
  if (name !== undefined) body.name = name
  return request('/api/stock_remark', {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

/** 按日期范围查询多只股票日线，分页。symbols 逗号分隔，start/end YYYY-MM-DD */
export function apiDataRange({ symbols, start, end, page = 1, size = 20 }) {
  const params = new URLSearchParams()
  params.set('symbols', symbols)
  if (start) params.set('start', start)
  if (end) params.set('end', end)
  params.set('page', String(page))
  params.set('size', String(size))
  return request('/api/data_range?' + params.toString())
}

/** 对指定股票在时间范围内运行综合分析。symbol, start, end 为 YYYY-MM-DD。返回 { summary, report_md } */
export function apiAnalyze(symbol, start, end) {
  const params = new URLSearchParams()
  params.set('symbol', symbol)
  params.set('start', start)
  params.set('end', end)
  return request('/api/analyze?' + params.toString())
}

/** 导出分析报告为 Markdown 文件（含结构化摘要，便于 AI 解析）。返回 Blob，需自行触发下载。 */
export async function apiExportReport(symbol, start, end) {
  const params = new URLSearchParams()
  params.set('symbol', symbol)
  params.set('start', start)
  params.set('end', end)
  const res = await fetch(API_BASE + '/api/analyze/export?' + params.toString())
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || err.message || '导出失败')
  }
  return res.blob()
}

// ---------- LSTM 训练与预测 ----------

/** 获取 LSTM 训练推荐日期范围。query: { years?: 1|2, use_config?: 1 }。返回 { start, end, hint }。 */
export function apiLstmRecommendedRange(query = {}) {
  const params = new URLSearchParams()
  if (query.years != null) params.set('years', String(query.years))
  if (query.use_config) params.set('use_config', '1')
  return request('/api/lstm/recommended-range?' + (params.toString() || 'years=1'))
}

/** 训练 LSTM 模型（单股 1/2/3 年可能需数分钟）。body: { symbol, all_years?, start?, end?, ... } */
const LSTM_TRAIN_TIMEOUT_MS = 30 * 60 * 1000 // 30 分钟，避免训练未完成就因超时断开
export function apiLstmTrain(body) {
  return request('/api/lstm/train', {
    method: 'POST',
    body: JSON.stringify(body || {}),
    timeoutMs: LSTM_TRAIN_TIMEOUT_MS,
  })
}

/** 请求停止指定股票正在进行的训练。body: { symbol }。返回 { ok, message } */
export function apiLstmTrainStop(body) {
  return request('/api/lstm/train/stop', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}

/** 一键训练全部股票。body: { start?, end?, years?, do_cv_tune?, do_shap?, do_plot?, fast_training? }。返回 { results, total, success_count, fail_count } */
export function apiLstmTrainAll(body) {
  return request('/api/lstm/train-all', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}

/** 清理指定股票的训练数据（训练流水、模型版本、曲线图），使可重新训练。body: { symbols: string[] }。返回 { cleared, symbols, message } */
export function apiLstmClearTraining(body) {
  return request('/api/lstm/clear-training', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}

/** 获取指定股票最近一次预测记录，用于刷新页面后恢复展示。 */
export function apiLstmLastPrediction(symbol) {
  const params = new URLSearchParams()
  params.set('symbol', symbol)
  return request('/api/lstm/last-prediction?' + params.toString())
}

/** 获取每只股票最近一次预测记录，用于按股票分别展示。返回 { predictions: [...] } */
export function apiLstmLastPredictions() {
  return request('/api/lstm/last-predictions')
}

/** 使用已保存模型预测；options: { years?: 1|2|3（默认3）, use_fallback?, trigger_train_async? }。返回含 direction, magnitude, prob_up, source, model_health。 */
export function apiLstmPredict(symbol, options = {}) {
  const params = new URLSearchParams()
  params.set('symbol', symbol)
  if (options.years != null) params.set('years', String(options.years))
  if (options.use_fallback) params.set('use_fallback', '1')
  if (options.trigger_train_async) params.set('trigger_train_async', '1')
  return request('/api/lstm/predict?' + params.toString())
}

/** 对全部股票执行预测。body: { use_fallback?, trigger_train_async? }。返回 { results, success_count, fail_count } */
export function apiLstmPredictAll(body = {}) {
  return request('/api/lstm/predict-all', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

/** 拟合曲线图数据（预测 vs 实际），供 ECharts 绘制。返回 { dates, actual_dir, pred_dir, actual_mag, pred_mag } */
export function apiLstmPlotData(symbol, years) {
  const params = new URLSearchParams()
  params.set('symbol', symbol)
  params.set('years', String(years === 1 || years === 2 || years === 3 ? years : 1))
  return request('/api/lstm/plot-data?' + params.toString())
}

/** 全部股票及其最后一次训练时间。返回 { stocks: [ { symbol, displayName, last_train } ] } */
export function apiLstmStocksTrainingStatus() {
  return request('/api/lstm/stocks-training-status')
}

/** 训练流水。query: { symbol?, limit?, dedupe? }，dedupe=1 时每只股票只返回最新一条 */
export function apiLstmTrainingRuns(query = {}) {
  const params = new URLSearchParams()
  if (query.symbol) params.set('symbol', query.symbol)
  if (query.limit != null) params.set('limit', String(query.limit))
  if (query.dedupe) params.set('dedupe', '1')
  return request('/api/lstm/training-runs?' + (params.toString() || 'limit=50'))
}

/** 数据库去重：每只股票只保留最新一条训练流水，返回 { deleted } */
export function apiLstmTrainingRunsDedupe() {
  return request('/api/lstm/training-runs/dedupe', { method: 'POST' })
}

/** 模型版本列表与当前版本。 */
export function apiLstmVersions() {
  return request('/api/lstm/versions')
}

/** 回滚到指定版本。body: { version_id } */
export function apiLstmRollback(versionId) {
  return request('/api/lstm/rollback', {
    method: 'POST',
    body: JSON.stringify({ version_id: versionId }),
  })
}

/** 检查/执行训练触发。body: { symbol?, run? } */
export function apiLstmCheckTriggers(body) {
  return request('/api/lstm/check-triggers', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}

/** 回填预测准确性。body: { symbol, as_of_date? } */
export function apiLstmUpdateAccuracy(body) {
  return request('/api/lstm/update-accuracy', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}

/** 监控状态汇总。 */
export function apiLstmMonitoring() {
  return request('/api/lstm/monitoring')
}

/** 执行一次性能衰减检测。query: { threshold?, n_recent?, log? } */
export function apiLstmPerformanceDecay(query = {}) {
  const params = new URLSearchParams()
  if (query.threshold != null) params.set('threshold', String(query.threshold))
  if (query.n_recent != null) params.set('n_recent', String(query.n_recent))
  if (query.log != null) params.set('log', query.log ? '1' : '0')
  return request('/api/lstm/performance-decay?' + (params.toString() || 'log=1'))
}

/** 检查告警；body.fire=true 时发送 webhook。 */
export function apiLstmAlerts(body) {
  return request('/api/lstm/alerts', {
    method: 'POST',
    body: JSON.stringify(body || {}),
  })
}
