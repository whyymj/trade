<template>
  <div class="lstm-fit-chart-wrap">
    <div v-if="loading" class="chart-loading">加载中…</div>
    <div v-else-if="error" class="chart-error">{{ error }}</div>
    <template v-else>
      <div class="chart-legend-bar">
        <span class="legend-model">预测曲线：{{ props.years }} 年模型</span>
        <span class="legend-label">图例：</span>
        <span class="legend-item"><i class="dot actual"></i> 实际</span>
        <span class="legend-item"><i class="dot pred"></i> 预测</span>
        <span class="legend-divider">|</span>
        <span class="legend-desc">每点 = 该日对「未来 5 日」的预测；下方曲线为按 5 日窗口结束日对齐的实际价与预测价。图中未来 5 日虚线也来自该 {{ props.years }} 年模型。</span>
      </div>
      <div v-if="diagnosticsList.length" class="diagnostics-bar">
        <span class="diagnostics-label">诊断：</span>
        <ul class="diagnostics-list">
          <li v-for="(msg, key) in diagnosticsList" :key="key">{{ msg }}</li>
        </ul>
      </div>
      <div ref="chartRef" class="lstm-fit-chart"></div>
      <div class="chart-tips">
        <strong>说明：</strong> 上为方向(涨/跌)、中为 5 日涨跌幅(%)、下为价格曲线；红线与蓝线越接近表示拟合越好。
        <strong class="tip-suggest">建议：</strong> 仅供研究参考，不构成投资建议；若红线过于平滑可查看诊断。
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { apiLstmPlotData } from '@/api/stock'

const props = defineProps({
  symbol: { type: String, required: true },
  years: { type: Number, required: true },
  refreshKey: { type: [Number, String], default: 0 },
})

const chartRef = ref(null)
const loading = ref(false)
const error = ref('')
const diagnosticsList = ref([])
let chart = null

function buildOption(data) {
  const {
    dates = [],
    actual_dir = [],
    pred_dir = [],
    actual_mag = [],
    pred_mag = [],
    dates_price = [],
    actual_price = [],
    predicted_price,
    forecast,
  } = data
  const actualPrice = Array.isArray(actual_price) ? actual_price : []
  const predictedPrice = Array.isArray(predicted_price) ? predicted_price : []
  const priceDates = Array.isArray(dates_price) ? dates_price : []
  const nPrice = priceDates.length
  // 与 dates_price 长度对齐，避免长度不一致导致预测线不画
  const actualPriceAligned = actualPrice.length >= nPrice ? actualPrice.slice(0, nPrice) : [...actualPrice, ...Array(nPrice - actualPrice.length).fill(null)]
  const predictedPriceAligned = predictedPrice.length >= nPrice ? predictedPrice.slice(0, nPrice) : [...predictedPrice, ...Array(nPrice - predictedPrice.length).fill(null)]
  const hasForecast = forecast?.forecast_dates?.length > 0
  const xCategories = hasForecast ? [...dates, ...forecast.forecast_dates] : dates
  const nHist = dates.length
  const nFc = hasForecast ? forecast.forecast_dates.length : 0
  const hasPrice = nPrice > 0 && (actualPrice.length > 0 || predictedPrice.length > 0)
  // 价格曲线 x 轴与方向/涨跌幅对齐：有 forecast 时也延伸到 forecast 末
  const priceXCategories = hasPrice ? (hasForecast ? [...priceDates, ...forecast.forecast_dates] : priceDates) : []
  // 支持 magnitude/direction 为 5 日数组（逐日）或单值（重复 5 次）
  const dirFc = Array.isArray(forecast?.direction) ? forecast.direction.slice(0, nFc) : Array(nFc).fill(forecast?.direction ?? 0)
  const magFc = Array.isArray(forecast?.magnitude) ? forecast.magnitude.slice(0, nFc) : Array(nFc).fill(forecast?.magnitude ?? 0)
  const forecastDirData = hasForecast ? [...Array(nHist).fill(null), ...dirFc] : []
  const forecastMagData = hasForecast ? [...Array(nHist).fill(null), ...magFc] : []
  // 实际/预测系列在扩展 x 轴时末尾补 null，避免与预测走势连在一起
  const pad = (arr, len) => (len > 0 && arr.length === nHist ? [...arr, ...Array(len).fill(null)] : arr)
  const actualDirPadded = pad(actual_dir, nFc)
  const predDirPadded = pad(pred_dir, nFc)
  const actualMagPadded = pad(actual_mag, nFc)
  const predMagPadded = pad(pred_mag, nFc)

  const series = [
    { name: '实际方向', type: 'line', data: actualDirPadded, xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { width: 1 }, color: '#5470c6' },
    { name: '预测方向', type: 'line', data: predDirPadded, xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { width: 1 }, color: '#ee6666' },
    { name: '实际涨跌幅', type: 'line', data: actualMagPadded, xAxisIndex: 1, yAxisIndex: 1, symbol: 'none', lineStyle: { width: 1 }, color: '#5470c6' },
    { name: '预测涨跌幅', type: 'line', data: predMagPadded, xAxisIndex: 1, yAxisIndex: 1, symbol: 'none', lineStyle: { width: 1 }, color: '#ee6666' },
  ]
  if (hasForecast) {
    series.push(
      { name: '预测走势(方向)', type: 'line', data: forecastDirData, xAxisIndex: 0, yAxisIndex: 0, symbol: 'circle', symbolSize: 6, lineStyle: { type: 'dashed', width: 2 }, color: '#91cc75' },
      { name: '预测走势(涨跌幅)', type: 'line', data: forecastMagData, xAxisIndex: 1, yAxisIndex: 1, symbol: 'circle', symbolSize: 6, lineStyle: { type: 'dashed', width: 2 }, color: '#91cc75' }
    )
  }
  let hasForecastPriceSeries = false
  if (hasPrice) {
    const pricePad = (arr, len) => (len > 0 && arr.length === nPrice ? [...arr, ...Array(len).fill(null)] : arr)
    const actualPricePadded = hasForecast ? pricePad(actualPriceAligned, nFc) : actualPriceAligned
    const predictedPricePadded = hasForecast ? pricePad(predictedPriceAligned, nFc) : predictedPriceAligned
    series.push(
      { name: '实际价格', type: 'line', data: actualPricePadded, xAxisIndex: 2, yAxisIndex: 2, symbol: 'none', connectNulls: false, lineStyle: { width: 1.5 }, color: '#5470c6' },
      { name: '预测价格', type: 'line', data: predictedPricePadded, xAxisIndex: 2, yAxisIndex: 2, symbol: 'none', connectNulls: false, lineStyle: { width: 1.5 }, color: '#ee6666' }
    )
    // 价格曲线也延伸未来预测：用最近价格 + forecast.magnitude 逐日推算未来 5 日价格
    if (hasForecast && nFc > 0 && Array.isArray(magFc) && magFc.length > 0) {
      const lastPrice = actualPriceAligned[nPrice - 1] ?? predictedPriceAligned[nPrice - 1]
      if (lastPrice != null && Number(lastPrice) > 0) {
        let p = Number(lastPrice)
        const forecastPrices = []
        for (let i = 0; i < nFc; i++) {
          const m = Number(magFc[i]) || 0
          p = p * (1 + m)
          forecastPrices.push(p)
        }
        const forecastPriceData = [...Array(nPrice).fill(null), ...forecastPrices]
        series.push({
          name: '预测走势(价格)',
          type: 'line',
          data: forecastPriceData,
          xAxisIndex: 2,
          yAxisIndex: 2,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: { type: 'dashed', width: 2 },
          color: '#91cc75',
          connectNulls: false,
        })
        hasForecastPriceSeries = true
      }
    }
  }

  // 每块预留：标题行 24px + 绘图区 + 横轴预留 36px，块间不重叠，底部留足避免日期被截断
  const titleRow = 24
  const axisMargin = 36
  const grid0Top = 34
  const grid0Height = hasPrice ? 128 : 220
  const grid1Top = grid0Top + grid0Height + axisMargin + titleRow
  const grid1Height = hasPrice ? 128 : 220
  const grid2Top = grid1Top + grid1Height + axisMargin + titleRow
  const grid2Height = 200
  const grid = [
    { left: 56, right: 40, top: grid0Top, height: grid0Height },
    { left: 56, right: 40, top: grid1Top, height: grid1Height },
  ]
  const xAxis = [
    { type: 'category', data: xCategories, gridIndex: 0, axisLabel: { fontSize: 10, margin: 12 }, axisTick: { show: true } },
    { type: 'category', data: xCategories, gridIndex: 1, axisLabel: { fontSize: 10, margin: 12 }, axisTick: { show: true } },
  ]
  const yAxis = [
    { type: 'value', min: 0, max: 1, gridIndex: 0, axisLabel: { formatter: (v) => (v === 0 ? '跌' : '涨'), fontSize: 10 }, splitLine: { show: false } },
    { type: 'value', gridIndex: 1, axisLabel: { formatter: (v) => (v * 100).toFixed(1) + '%', fontSize: 10 }, splitLine: { show: false } },
  ]
  if (hasPrice) {
    grid.push({ left: 56, right: 40, top: grid2Top, height: grid2Height })
    xAxis.push({ type: 'category', data: priceXCategories, gridIndex: 2, axisLabel: { fontSize: 10, margin: 16 }, axisTick: { show: true } })
    yAxis.push({
      type: 'value',
      gridIndex: 2,
      axisLabel: { formatter: (v) => (v != null && Number(v) >= 10000 ? (v / 10000).toFixed(1) + 'w' : v), fontSize: 10 },
      scale: true,
      splitLine: { show: false },
    })
  }

  // 各块标题（放在预留的标题行内，避免与 Y 轴/图例重叠）
  const titles = [
    { text: '方向（涨/跌）', left: 56, top: 8, textStyle: { fontSize: 12, fontWeight: 600 } },
    { text: '5日涨跌幅(%)', left: 56, top: grid1Top - titleRow + 6, textStyle: { fontSize: 12, fontWeight: 600 } },
  ]
  if (hasPrice) {
    titles.push({ text: '价格曲线', left: 56, top: grid2Top - titleRow + 6, textStyle: { fontSize: 12, fontWeight: 600 } })
  }

  // 各块图例（与标题同一行，靠右，避免压住下一块横轴）
  const legends = [
    { data: ['实际方向', '预测方向'], top: 8, right: 40, itemWidth: 12, itemHeight: 8, itemGap: 8, textStyle: { fontSize: 10 }, orient: 'horizontal', show: true },
    { data: ['实际涨跌幅', '预测涨跌幅'], top: grid1Top - titleRow + 6, right: 40, itemWidth: 12, itemHeight: 8, itemGap: 8, textStyle: { fontSize: 10 }, orient: 'horizontal', show: true },
  ]
  if (hasPrice) {
    const priceLegendItems = hasForecastPriceSeries ? ['实际价格', '预测价格', '预测走势(价格)'] : ['实际价格', '预测价格']
    legends.push({ data: priceLegendItems, top: grid2Top - titleRow + 6, right: 40, itemWidth: 12, itemHeight: 8, itemGap: 8, textStyle: { fontSize: 10 }, orient: 'horizontal', show: true })
  }

  return {
    animation: false,
    title: titles,
    grid,
    xAxis,
    yAxis,
    series,
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        if (!Array.isArray(params)) return ''
        const lines = params.map((p) => `${p.marker} ${p.seriesName}: ${p.value != null ? (p.axisDimension === 'y' && p.axisIndex === 1 ? (Number(p.value) * 100).toFixed(2) + '%' : p.value) : '—'}`)
        const axisVal = params[0]?.axisValue ?? ''
        return `<div style="font-size:11px">${axisVal}</div>${lines.join('<br/>')}`
      },
    },
    legend: legends,
  }
}

async function loadAndRender() {
  if (!props.symbol || !props.years) return
  loading.value = true
  error.value = ''
  if (chart) {
    chart.dispose()
    chart = null
  }
  try {
    const data = await apiLstmPlotData(props.symbol, props.years)
    const hasData = data?.dates?.length > 0
    if (!hasData) {
      error.value = '无曲线数据'
      loading.value = false
      return
    }
    diagnosticsList.value = data.diagnostics ? Object.values(data.diagnostics) : []
    loading.value = false
    await nextTick()
    if (!chartRef.value) return
    chart = echarts.init(chartRef.value)
    chart.setOption(buildOption(data), true)
    chart.resize()
  } catch (e) {
    error.value = e?.data?.error || e?.message || '加载失败'
    loading.value = false
  }
}

function dispose() {
  if (chart) {
    chart.dispose()
    chart = null
  }
}

onMounted(() => loadAndRender())
onUnmounted(dispose)
watch(() => [props.symbol, props.years, props.refreshKey], () => loadAndRender())
</script>

<style scoped>
.lstm-fit-chart-wrap {
  min-height: 720px;
  position: relative;
}
.lstm-fit-chart {
  width: 100%;
  min-width: 120px;
  height: 720px;
}
.chart-loading,
.chart-error {
  height: 140px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: #909399;
  background: #1e1e1e;
  border-radius: 8px;
  border: 1px dashed #3a3a3a;
}
.chart-error {
  color: #f56c6c;
}
.diagnostics-bar {
  font-size: 11px;
  color: #e6a23c;
  background: rgba(230, 162, 60, 0.08);
  border: 1px solid rgba(230, 162, 60, 0.3);
  border-radius: 6px;
  padding: 6px 8px;
  margin-bottom: 6px;
}
.diagnostics-label {
  font-weight: 600;
  margin-right: 4px;
}
.diagnostics-list {
  margin: 0;
  padding-left: 16px;
  line-height: 1.4;
}
.chart-legend-bar {
  font-size: 11px;
  color: #909399;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.legend-model {
  font-weight: 600;
  color: #409eff;
  margin-right: 8px;
}
.legend-label {
  font-weight: 600;
  color: #606266;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.legend-item .dot {
  display: inline-block;
  width: 8px;
  height: 4px;
  border-radius: 1px;
}
.legend-item .dot.actual {
  background: #5470c6;
}
.legend-item .dot.pred {
  background: #ee6666;
}
.legend-divider {
  color: #dcdfe6;
  margin: 0 2px;
}
.legend-desc {
  color: #909399;
}
.chart-tips {
  font-size: 11px;
  color: #909399;
  margin-top: 8px;
  padding: 6px 8px;
  background: #1e1e1e;
  border-radius: 6px;
  border: 1px solid #2a2a2a;
  line-height: 1.5;
}
.chart-tips strong {
  color: #c0c4cc;
  margin-right: 4px;
}
.chart-tips .tip-suggest {
  margin-left: 8px;
}
</style>
