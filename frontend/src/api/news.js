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

// ============ 新闻相关 API ============

export function getNewsList(params = {}) {
  const query = new URLSearchParams()
  if (params.days) query.set('days', String(params.days))
  if (params.category) query.set('category', params.category)
  if (params.source) query.set('source', params.source)
  if (params.limit) query.set('limit', String(params.limit))
  if (params.offset) query.set('offset', String(params.offset))
  return request('/api/news/latest?' + query.toString())
}

export function getNewsDetail(newsId) {
  const encoded = encodeURIComponent(newsId)
  return request('/api/news/detail/' + encoded)
}

export function syncNews() {
  return request('/api/news/sync', { method: 'POST' })
}

export function analyzeNews(params = {}) {
  return request('/api/news/analyze', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function getLatestAnalysis() {
  return request('/api/news/analysis/latest')
}

// ============ 市场数据 API ============

/**
 * 获取市场情绪数据
 * @param {number} days - 获取天数，默认7天
 */
export function getMarketSentiment(days = 7) {
  return request(`/api/market/sentiment?days=${days}`)
}

/**
 * 获取市场特征数据（涨跌停、成交额、换手率等）
 * @param {number} days - 获取天数
 */
export function getMarketFeatures(days = 30) {
  return request(`/api/market/features?days=${days}`)
}

/**
 * 获取宏观经济数据
 * @param {string} indicator - 指标类型
 * @param {number} days - 获取天数
 */
export function getMarketMacro(indicator = null, days = 30) {
  let url = '/api/market/macro?days=' + days
  if (indicator) {
    url += '&indicator=' + indicator
  }
  return request(url)
}

/**
 * 获取最新宏观经济数据
 * @param {string} indicator - 指标类型
 */
export function getMarketMacroLatest(indicator = null) {
  let url = '/api/market/macro/latest'
  if (indicator) {
    url += '?indicator=' + indicator
  }
  return request(url)
}

/**
 * 获取资金流向数据
 * @param {number} days - 获取天数
 */
export function getMarketMoneyFlow(days = 7) {
  return request(`/api/market/money-flow?days=${days}`)
}

/**
 * 获取全球市场数据
 * @param {number} days - 获取天数
 */
export function getMarketGlobal(days = 7) {
  return request(`/api/market/global?days=${days}`)
}

/**
 * 获取市场汇总数据
 */
export function getMarketSummary() {
  return request('/api/market/summary')
}
