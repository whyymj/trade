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

export function apiData(filename) {
  return request('/api/data?file=' + encodeURIComponent(filename))
}

export function apiAddStock(code) {
  return request('/api/add_stock', {
    method: 'POST',
    body: JSON.stringify({ code }),
  })
}

export function apiUpdateAll() {
  return request('/api/update_all', { method: 'POST' })
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
