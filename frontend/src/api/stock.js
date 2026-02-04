const API_BASE = ''

async function request(url, options = {}) {
  const res = await fetch(API_BASE + url, {
    ...options,
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
