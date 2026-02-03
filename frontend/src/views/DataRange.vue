<template>
  <div class="container">
    <h1>按日期范围查多只股票</h1>
    <p class="subtitle">选择多只股票与日期范围，分页查看日线数据</p>

    <div class="nav-row">
      <router-link to="/" class="link">← 曲线</router-link>
      <router-link to="/data-manage" class="link">数据管理</router-link>
    </div>

    <div class="card">
      <div class="card-title">查询条件</div>
      <div class="form-row">
        <label>股票代码</label>
        <input
          v-model="symbolsInput"
          type="text"
          placeholder="逗号分隔，如 600519,000001,09678.HK"
          class="symbols-input"
        />
        <span class="pick-hint">或从下方勾选</span>
      </div>
      <div class="checkbox-list">
        <label
          v-for="item in stockStore.fileList"
          :key="item.filename"
          class="checkbox-item"
        >
          <input
            type="checkbox"
            :value="item.filename"
            :checked="selectedSymbols.includes(item.filename)"
            @change="toggleSymbol(item.filename)"
          />
          <span>{{ item.displayName || item.filename }}</span>
        </label>
      </div>
      <div class="form-row">
        <label>开始日期</label>
        <input v-model="start" type="date" />
      </div>
      <div class="form-row">
        <label>结束日期</label>
        <input v-model="end" type="date" />
      </div>
      <div class="form-row">
        <label>每页条数</label>
        <select v-model.number="pageSize">
          <option :value="10">10</option>
          <option :value="20">20</option>
          <option :value="50">50</option>
          <option :value="100">100</option>
        </select>
        <button
          type="button"
          class="btn btn-primary"
          :disabled="loading || !symbolsList.length"
          @click="query(1)"
        >
          查询
        </button>
      </div>
    </div>

    <div class="card" v-if="result">
      <div class="card-title">
        共 {{ result.total }} 条，第 {{ result.page }} / {{ totalPages }} 页
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>日期</th>
              <th>股票</th>
              <th>开盘</th>
              <th>收盘</th>
              <th>最高</th>
              <th>最低</th>
              <th>成交量</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in result.data" :key="row.symbol + row.trade_date">
              <td>{{ row.trade_date }}</td>
              <td>{{ row.symbol }}</td>
              <td>{{ formatNum(row.open) }}</td>
              <td>{{ formatNum(row.close) }}</td>
              <td>{{ formatNum(row.high) }}</td>
              <td>{{ formatNum(row.low) }}</td>
              <td>{{ formatNum(row.volume) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <button
          type="button"
          class="btn btn-secondary"
          :disabled="result.page <= 1 || loading"
          @click="query(result.page - 1)"
        >
          上一页
        </button>
        <span class="page-info">
          第 {{ result.page }} / {{ totalPages }} 页
        </span>
        <button
          type="button"
          class="btn btn-secondary"
          :disabled="result.page >= totalPages || loading"
          @click="query(result.page + 1)"
        >
          下一页
        </button>
      </div>
    </div>

    <div v-else-if="searched && !loading" class="empty">暂无数据或请先选择股票与日期范围</div>

    <div v-show="loading" class="loading">加载中…</div>
    <div v-show="appStore.errorMessage" class="error">{{ appStore.errorMessage }}</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { useStockStore } from '@/stores/stock'
import { apiDataRange } from '@/api/stock'

const appStore = useAppStore()
const stockStore = useStockStore()

const symbolsInput = ref('')
const selectedSymbols = ref([])
const start = ref('')
const end = ref('')
const pageSize = ref(20)
const loading = ref(false)
const result = ref(null)
const searched = ref(false)

const symbolsList = computed(() => {
  const fromInput = symbolsInput.value
    .split(/[,，\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
  const combined = [...new Set([...selectedSymbols.value, ...fromInput])]
  return combined
})

const totalPages = computed(() => {
  if (!result.value || result.value.page_size <= 0) return 0
  return Math.max(1, Math.ceil(result.value.total / result.value.page_size))
})

function toggleSymbol(symbol) {
  const i = selectedSymbols.value.indexOf(symbol)
  if (i === -1) {
    selectedSymbols.value = [...selectedSymbols.value, symbol]
  } else {
    selectedSymbols.value = selectedSymbols.value.filter((s) => s !== symbol)
  }
}

function formatNum(v) {
  if (v == null || v === '') return '—'
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  if (Number.isInteger(n)) return n.toLocaleString()
  return n.toFixed(4)
}

async function query(page = 1) {
  if (!symbolsList.value.length) {
    appStore.setError('请至少选择一个股票代码')
    return
  }
  loading.value = true
  searched.value = true
  appStore.setError('')
  try {
    const data = await apiDataRange({
      symbols: symbolsList.value.join(','),
      start: start.value || undefined,
      end: end.value || undefined,
      page,
      size: pageSize.value,
    })
    result.value = data
  } catch (e) {
    appStore.setError(e.message || '查询失败')
    result.value = null
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await stockStore.fetchList()
})
</script>

<style scoped>
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  font-weight: 600;
  margin-bottom: 8px;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.subtitle {
  color: #8892a0;
  margin-bottom: 16px;
}
.nav-row {
  margin-bottom: 20px;
  display: flex;
  gap: 16px;
}
.link {
  color: #00d9ff;
  text-decoration: none;
  font-size: 14px;
}
.link:hover {
  text-decoration: underline;
}
.card {
  background: rgba(30, 41, 59, 0.6);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.card-title {
  font-size: 15px;
  color: #94a3b8;
  margin-bottom: 16px;
}
.form-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.form-row label {
  width: 90px;
  color: #a0aec0;
  font-size: 14px;
}
.form-row input,
.form-row select {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2d3748;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 14px;
  min-width: 140px;
}
.symbols-input {
  flex: 1;
  max-width: 360px;
}
.pick-hint {
  color: #64748b;
  font-size: 13px;
}
.checkbox-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 20px;
  margin-bottom: 16px;
  padding: 12px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.checkbox-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #e2e8f0;
  font-size: 13px;
  cursor: pointer;
}
.checkbox-item input {
  width: auto;
  min-width: auto;
}
.table-wrap {
  overflow-x: auto;
  margin-bottom: 16px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.data-table th,
.data-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.data-table th {
  color: #94a3b8;
  font-weight: 500;
}
.data-table td {
  color: #e2e8f0;
}
.data-table tbody tr:hover {
  background: rgba(255, 255, 255, 0.03);
}
.pagination {
  display: flex;
  align-items: center;
  gap: 16px;
}
.page-info {
  color: #94a3b8;
  font-size: 14px;
}
.btn {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn:hover:not(:disabled) {
  opacity: 0.9;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary {
  background: linear-gradient(135deg, #00d9ff, #00a8cc);
  color: #0f172a;
}
.btn-secondary {
  background: #334155;
  color: #e2e8f0;
}
.empty,
.loading,
.error {
  text-align: center;
  padding: 24px;
  color: #94a3b8;
}
.error {
  color: #f87171;
}
</style>
