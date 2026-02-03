<template>
  <div class="container">
    <h1>股票数据曲线</h1>
    <p class="subtitle">选择股票查看 K 线相关数据</p>

    <div class="nav-row">
      <router-link to="/data-manage" class="link">数据管理</router-link>
    </div>

    <div class="toolbar">
      <div class="toolbar-row">
        <label for="stock">选择股票：</label>
        <select id="stock" v-model="selectedValue" @change="onStockChange">
          <option value="">-- 请选择 --</option>
          <option
            v-for="item in stockStore.fileList"
            :key="item.filename"
            :value="encodeURIComponent(item.filename)"
          >
            {{ item.displayName || item.filename }}
          </option>
        </select>
      </div>
      <div class="toolbar-row">
        <label for="stockCode">股票代码：</label>
        <input
          id="stockCode"
          v-model="stockCode"
          type="text"
          placeholder="如 600519 或 09678.HK"
          maxlength="10"
          @input="onCodeInput"
        />
        <button
          type="button"
          class="btn btn-primary"
          :disabled="addingStock"
          @click="handleAddStock"
        >
          抓取并加入配置
        </button>
        <button
          type="button"
          class="btn btn-secondary"
          :disabled="updatingAll"
          @click="handleUpdateAll"
        >
          一键更新全部数据
        </button>
      </div>
    </div>

    <div class="chart-wrap">
      <div class="chart-title">价格走势（开盘 / 收盘 / 最高 / 最低）</div>
      <div ref="priceChartRef" class="chart"></div>
    </div>

    <div class="chart-wrap">
      <div class="chart-title">成交量</div>
      <div ref="volumeChartRef" class="chart volume"></div>
    </div>

    <div v-show="appStore.loading" class="loading">加载中…</div>
    <div v-show="appStore.errorMessage" class="error">
      {{ appStore.errorMessage }}
    </div>
    <div
      v-show="appStore.toastVisible"
      class="toast show"
    >
      {{ appStore.toastMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useStockStore } from '@/stores/stock'
import { useAppStore } from '@/stores/app'

const stockStore = useStockStore()
const appStore = useAppStore()

const priceChartRef = ref(null)
const volumeChartRef = ref(null)
const stockCode = ref('')
const addingStock = ref(false)
const updatingAll = ref(false)

const selectedValue = computed({
  get: () => stockStore.selectedFilename,
  set: (v) => stockStore.setSelected(v),
})

let priceChart = null
let volumeChart = null

function onCodeInput(e) {
  const v = e.target.value.trim().toUpperCase()
  if (/\.HK$/.test(v)) {
    stockCode.value = v
      .replace(/[^\d.]/g, '')
      .replace(/(\d{5})\.HK.*/, '$1.HK')
      .slice(0, 10)
  } else {
    stockCode.value = e.target.value.replace(/\D/g, '').slice(0, 6)
  }
}

function isStockCodeValid(code) {
  if (!code || !code.trim()) return false
  const c = code.trim().toUpperCase()
  return (
    (c.length === 6 && /^\d{6}$/.test(c)) ||
    (c.length === 5 && /^\d{5}$/.test(c)) ||
    /^\d{5}\.HK$/.test(c)
  )
}

function renderPriceChart(data) {
  if (!priceChartRef.value || !data) return
  if (!priceChart) {
    priceChart = echarts.init(priceChartRef.value)
  }
  const dates = data.dates || []
  const open = data['开盘'] || []
  const close = data['收盘'] || []
  const high = data['最高'] || []
  const low = data['最低'] || []

  priceChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(30,41,59,0.95)',
      borderColor: '#334155',
    },
    legend: {
      data: ['开盘', '收盘', '最高', '最低'],
      textStyle: { color: '#94a3b8' },
      top: 0,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      axisLabel: { color: '#94a3b8' },
    },
    series: [
      {
        name: '开盘',
        type: 'line',
        smooth: true,
        data: open,
        symbol: 'none',
        lineStyle: { width: 2 },
      },
      {
        name: '收盘',
        type: 'line',
        smooth: true,
        data: close,
        symbol: 'none',
        lineStyle: { width: 2 },
      },
      {
        name: '最高',
        type: 'line',
        smooth: true,
        data: high,
        symbol: 'none',
        lineStyle: { width: 2 },
      },
      {
        name: '最低',
        type: 'line',
        smooth: true,
        data: low,
        symbol: 'none',
        lineStyle: { width: 2 },
      },
    ],
    color: ['#00d9ff', '#00ff88', '#fbbf24', '#f87171'],
  })
}

function renderVolumeChart(data) {
  if (!volumeChartRef.value || !data) return
  if (!volumeChart) {
    volumeChart = echarts.init(volumeChartRef.value)
  }
  const dates = data.dates || []
  const volume = data['成交量'] || []

  volumeChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(30,41,59,0.95)',
      borderColor: '#334155',
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '8%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' },
    },
    yAxis: {
      type: 'value',
      name: '成交量',
      nameTextStyle: { color: '#94a3b8' },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      axisLabel: { color: '#94a3b8' },
    },
    series: [
      {
        name: '成交量',
        type: 'bar',
        data: volume,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#00d9ff' },
            { offset: 1, color: 'rgba(0,217,255,0.3)' },
          ]),
        },
      },
    ],
  })
}

async function onStockChange() {
  const value = selectedValue.value
  appStore.setError('')
  if (!value) {
    stockStore.clearChartData()
    priceChart?.clear()
    volumeChart?.clear()
    return
  }
  appStore.setLoading(true)
  try {
    const data = await stockStore.fetchData(value)
    if (data?.dates) {
      renderPriceChart(data)
      renderVolumeChart(data)
    } else {
      appStore.setError('数据格式异常')
    }
  } catch (e) {
    appStore.setError(e.message)
  } finally {
    appStore.setLoading(false)
  }
}

async function handleAddStock() {
  const code = stockCode.value.trim()
  if (!isStockCodeValid(code)) {
    appStore.showToast('请输入 A股6位 或 港股5位/xxxxx.HK')
    return
  }
  addingStock.value = true
  appStore.setError('')
  try {
    const data = await stockStore.addStock(code)
    if (data?.ok) {
      appStore.showToast(data.message || '已加入配置')
      stockCode.value = ''
      await stockStore.fetchList()
    } else {
      appStore.showToast(data?.message || '失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
  } finally {
    addingStock.value = false
  }
}

async function handleUpdateAll() {
  updatingAll.value = true
  appStore.setError('')
  appStore.setLoading(true)
  try {
    const data = await stockStore.updateAll()
    if (data?.ok && data?.results) {
      const ok = data.results.filter((r) => r.ok).length
      const fail = data.results.filter((r) => !r.ok).length
      appStore.showToast('更新完成：成功 ' + ok + '，失败 ' + fail)
      await stockStore.fetchList()
    } else {
      appStore.showToast('更新失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
  } finally {
    appStore.setLoading(false)
    updatingAll.value = false
  }
}

function onResize() {
  priceChart?.resize()
  volumeChart?.resize()
}

onMounted(async () => {
  await stockStore.fetchList()
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  priceChart?.dispose()
  volumeChart?.dispose()
})
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  text-align: center;
  font-weight: 600;
  margin-bottom: 8px;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.subtitle {
  text-align: center;
  color: #8892a0;
  margin-bottom: 8px;
}
.nav-row {
  text-align: right;
  margin-bottom: 16px;
}
.nav-row .link {
  color: #00d9ff;
  text-decoration: none;
  font-size: 14px;
}
.nav-row .link:hover {
  text-decoration: underline;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
label {
  color: #a0aec0;
  font-size: 14px;
}
select {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2d3748;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 14px;
  min-width: 200px;
  cursor: pointer;
}
select:hover,
select:focus {
  border-color: #00d9ff;
  outline: none;
}
.chart-wrap {
  background: rgba(30, 41, 59, 0.6);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.chart-title {
  font-size: 15px;
  color: #94a3b8;
  margin-bottom: 12px;
}
.chart {
  width: 100%;
  height: 380px;
}
.chart.volume {
  height: 220px;
}
.loading,
.error {
  text-align: center;
  padding: 40px;
  color: #94a3b8;
}
.error {
  color: #f87171;
}
.toolbar-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
input[type='text'] {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2d3748;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 14px;
  width: 120px;
}
input[type='text']:focus {
  border-color: #00d9ff;
  outline: none;
}
.btn {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn:hover {
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
.toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 24px;
  border-radius: 8px;
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  font-size: 14px;
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.3s;
}
.toast.show {
  opacity: 1;
}
</style>
