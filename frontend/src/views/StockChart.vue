<template>
  <div class="container">
    <h1>股票数据曲线</h1>
    <p class="subtitle">选择股票查看 K 线相关数据</p>

    <div class="nav-row">
      <el-link type="primary" :underline="false" router to="/data-manage">数据管理</el-link>
      <el-link type="primary" :underline="false" router to="/data-range">按日期范围查询</el-link>
    </div>

    <div class="toolbar">
      <div class="toolbar-row">
        <span class="label">选择股票：</span>
        <el-select
          v-model="selectedValue"
          placeholder="请选择"
          clearable
          filterable
          style="width: 220px"
          @change="onStockChange"
        >
          <el-option
            v-for="item in stockStore.fileList"
            :key="item.filename"
            :label="item.displayName || item.filename"
            :value="encodeURIComponent(item.filename)"
          />
        </el-select>
      </div>
      <div class="toolbar-row">
        <span class="label">股票代码：</span>
        <el-input
          v-model="stockCode"
          placeholder="如 600519 或 09678.HK"
          maxlength="10"
          style="width: 160px"
          @input="onCodeInput"
        />
        <el-button type="primary" :loading="addingStock" @click="handleAddStock">
          抓取近5年并加入配置
        </el-button>
        <el-button :loading="updatingAll" @click="handleUpdateAll">
          增量更新（补全至今日）
        </el-button>
      </div>
    </div>

    <div v-loading="appStore.loading" class="chart-wrap">
      <div class="chart-title">价格走势（开盘 / 收盘 / 最高 / 最低）</div>
      <div ref="priceChartRef" class="chart"></div>
    </div>

    <div class="chart-wrap">
      <div class="chart-title">成交量</div>
      <div ref="volumeChartRef" class="chart volume"></div>
    </div>

    <el-alert v-if="appStore.errorMessage" type="error" :title="appStore.errorMessage" show-icon closable class="error-alert" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
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

function onCodeInput(val) {
  const v = (val || '').trim().toUpperCase()
  if (/\.HK$/.test(v)) {
    stockCode.value = v
      .replace(/[^\d.]/g, '')
      .replace(/(\d{5})\.HK.*/, '$1.HK')
      .slice(0, 10)
  } else {
    stockCode.value = (val || '').replace(/\D/g, '').slice(0, 6)
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
    ElMessage.warning('请输入 A股6位 或 港股5位/xxxxx.HK')
    return
  }
  addingStock.value = true
  appStore.setError('')
  try {
    const data = await stockStore.addStock(code)
    if (data?.ok) {
      ElMessage.success(data.message || '已加入配置')
      stockCode.value = ''
      await stockStore.fetchList()
    } else {
      ElMessage.error(data?.message || '失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
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
      ElMessage.success('更新完成：成功 ' + ok + '，失败 ' + fail)
      await stockStore.fetchList()
    } else {
      ElMessage.error('更新失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
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
.label {
  color: var(--el-text-color-regular);
  font-size: 14px;
}
.error-alert {
  margin-top: 16px;
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
.nav-row .el-link {
  margin-left: 12px;
}
.nav-row .el-link:first-child {
  margin-left: 0;
}
</style>
