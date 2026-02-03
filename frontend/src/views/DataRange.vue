<template>
  <div class="container">
    <h1>按日期范围查多只股票</h1>
    <p class="subtitle">选择多只股票与日期范围，分页查看日线数据</p>

    <div class="nav-row">
      <el-link type="primary" :underline="false" router to="/">← 曲线</el-link>
      <el-link type="primary" :underline="false" router to="/data-manage">数据管理</el-link>
    </div>

    <el-card shadow="never" class="card">
      <template #header><span>查询条件</span></template>
      <el-form label-width="90px" label-position="left">
        <el-form-item label="股票代码">
          <el-input
            v-model="symbolsInput"
            placeholder="逗号分隔，如 600519,000001,09678.HK"
            clearable
            style="width: 360px; max-width: 100%"
          />
          <span class="pick-hint">或从下方勾选</span>
        </el-form-item>
        <el-form-item label="">
          <el-checkbox-group v-model="selectedSymbols" class="checkbox-list">
            <el-checkbox v-for="item in stockStore.fileList" :key="item.filename" :label="item.filename">
              {{ item.displayName || item.filename }}
            </el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item label="开始日期">
          <el-date-picker
            v-model="start"
            type="date"
            placeholder="选择日期"
            value-format="YYYY-MM-DD"
            format="YYYY-MM-DD"
            style="width: 160px"
          />
        </el-form-item>
        <el-form-item label="结束日期">
          <el-date-picker
            v-model="end"
            type="date"
            placeholder="选择日期"
            value-format="YYYY-MM-DD"
            format="YYYY-MM-DD"
            style="width: 160px"
          />
        </el-form-item>
        <el-form-item label="每页条数">
          <el-select v-model.number="pageSize" style="width: 100px">
            <el-option :value="10" label="10" />
            <el-option :value="20" label="20" />
            <el-option :value="50" label="50" />
            <el-option :value="100" label="100" />
          </el-select>
          <el-button type="primary" :loading="loading" :disabled="!symbolsList.length" @click="query(1)">
            查询
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="result" shadow="never" class="card">
      <template #header>
        <span>共 {{ result.total }} 条，第 {{ result.page }} / {{ totalPages }} 页</span>
      </template>
      <el-table :data="result.data" stripe style="width: 100%">
        <el-table-column prop="trade_date" label="日期" width="110" />
        <el-table-column prop="symbol" label="股票" width="100" />
        <el-table-column prop="open" label="开盘" :formatter="(row, col, cell) => formatNum(cell)" />
        <el-table-column prop="close" label="收盘" :formatter="(row, col, cell) => formatNum(cell)" />
        <el-table-column prop="high" label="最高" :formatter="(row, col, cell) => formatNum(cell)" />
        <el-table-column prop="low" label="最低" :formatter="(row, col, cell) => formatNum(cell)" />
        <el-table-column prop="volume" label="成交量" :formatter="(row, col, cell) => formatNum(cell)" min-width="100" />
      </el-table>
      <div class="pagination-wrap">
        <el-pagination
          :current-page="result.page"
          :page-size="result.page_size"
          :total="result.total"
          layout="prev, pager, next, total"
          @current-change="onPageChange"
        />
      </div>
    </el-card>

    <el-empty v-else-if="searched && !loading" description="暂无数据或请先选择股票与日期范围" :image-size="80" />

    <div v-loading="loading" class="loading-wrap" />
    <el-alert v-if="appStore.errorMessage" type="error" :title="appStore.errorMessage" show-icon closable class="error-alert" @close="appStore.setError('')" />
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

function onPageChange(page) {
  query(page)
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
  color: var(--el-text-color-secondary);
  margin-bottom: 16px;
}
.nav-row {
  margin-bottom: 20px;
  display: flex;
  gap: 16px;
}
.nav-row .el-link {
  font-size: 14px;
}
.card {
  margin-bottom: 20px;
}
.pick-hint {
  margin-left: 12px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.checkbox-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 20px;
  padding: 12px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.pagination-wrap {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}
.loading-wrap {
  min-height: 2px;
  margin-top: 16px;
}
.error-alert {
  margin-top: 16px;
}
</style>
