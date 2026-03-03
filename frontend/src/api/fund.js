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

export function getFundList(params = {}) {
  const query = new URLSearchParams()
  if (params.page) query.set('page', String(params.page))
  if (params.size) query.set('size', String(params.size))
  if (params.fund_type) query.set('fund_type', params.fund_type)
  return request('/api/fund/list?' + query.toString())
}

export function getFundWatchlist(params = {}) {
  const query = new URLSearchParams()
  if (params.page) query.set('page', String(params.page))
  if (params.size) query.set('size', String(params.size))
  return request('/api/fund/watchlist?' + query.toString())
}

export function getFundInfo(code) {
  return request('/api/fund/list?page=1&size=1000').then(res => {
    const fund = (res.data || []).find(f => f.fund_code === code)
    return fund || null
  })
}

export function addFund(fund) {
  return request('/api/fund/add', {
    method: 'POST',
    body: JSON.stringify(fund),
  })
}

export function deleteFund(code) {
  return request(`/api/fund/${code}`, {
    method: 'DELETE',
  })
}

export function watchFund(code, watch = true) {
  return request(`/api/fund/${code}/watch`, {
    method: 'PUT',
    body: JSON.stringify({ watch }),
  })
}

export function getFundNav(code, params = {}) {
  const query = new URLSearchParams()
  if (params.start) query.set('start', params.start)
  if (params.end) query.set('end', params.end)
  return request(`/api/fund/nav/${code}?` + query.toString())
}

export function getLatestNav(code) {
  return request(`/api/fund/nav/latest/${code}`)
}

export function getIndicators(code, days = 365) {
  return request(`/api/fund/indicators/${code}?days=${days}`)
}

export function getBenchmark(code, benchmark = '000300') {
  return request(`/api/fund/benchmark/${code}?benchmark=${benchmark}`)
}

export function predict(code) {
  return request('/api/fund/predict', {
    method: 'POST',
    body: JSON.stringify({ fund_code: code }),
  })
}

export function getPrediction(code) {
  return request(`/api/fund/prediction/${code}`)
}

export function getFitPlot(code) {
  return request(`/api/fund/fit-plot/${code}`)
}

export function getLlmAnalysis(code, type) {
  const endpoints = {
    profile: `/api/fund/analysis/profile/${code}`,
    performance: `/api/fund/analysis/performance/${code}`,
    risk: `/api/fund/analysis/risk/${code}`,
    advice: `/api/fund/advice/${code}`,
    report: `/api/fund/report/${code}`,
  }
  return request(endpoints[type] || endpoints.profile)
}

export function getLlmStatus() {
  return request('/api/fund/llm-status')
}

export function syncFunds(fundCodes = []) {
  return request('/api/sync/funds', {
    method: 'POST',
    body: JSON.stringify({ fund_codes: fundCodes }),
  })
}
