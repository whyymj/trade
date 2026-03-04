<template>
  <div class="sentiment-chart card">
    <div class="chart-header">
      <h3 class="chart-title">
        <span class="title-icon">📈</span>
        市场情绪
      </h3>
      <div class="chart-legend">
        <span class="legend-item up">上涨</span>
        <span class="legend-item down">下跌</span>
      </div>
    </div>
    
    <!-- 核心指标 -->
    <div class="metrics-grid" v-if="sentimentData">
      <div class="metric-box">
        <span class="metric-label">成交额</span>
        <span class="metric-value">{{ formatVolume(sentimentData.volume) }}</span>
      </div>
      <div class="metric-box up">
        <span class="metric-label">涨停</span>
        <span class="metric-value">{{ sentimentData.up_count || 0 }}</span>
      </div>
      <div class="metric-box down">
        <span class="metric-label">跌停</span>
        <span class="metric-value">{{ sentimentData.down_count || 0 }}</span>
      </div>
      <div class="metric-box">
        <span class="metric-label">换手率</span>
        <span class="metric-value">{{ (sentimentData.turnover_rate || 0).toFixed(2) }}%</span>
      </div>
    </div>

    <!-- 图表 -->
    <div ref="chartRef" class="echarts-container"></div>
    
    <!-- 加载状态 -->
    <div v-if="loading" class="chart-loading">
      <div class="loading-spinner"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { getMarketFeatures, getMarketSentiment } from '@/api/news'

const chartRef = ref(null)
const loading = ref(false)
const sentimentData = ref(null)
const chartData = ref([])
let chartInstance = null

// 加载数据
async function loadData() {
  loading.value = true
  try {
    // 加载最新的情绪数据
    const sentimentRes = await getMarketSentiment(1)
    if (sentimentRes.code === 0 && sentimentRes.data) {
      sentimentData.value = sentimentRes.data
    }
    
    // 加载历史特征数据用于图表
    const featuresRes = await getMarketFeatures(30)
    if (featuresRes.code === 0 && featuresRes.data) {
      const data = featuresRes.data
      chartData.value = data.map(item => ({
        date: item.trade_date,
        upCount: item.up_count,
        downCount: item.down_count,
        volume: item.volume,
        turnover: item.turnover_rate,
      })).reverse()
      updateChart()
    }
  } catch (e) {
    console.error('加载市场情绪数据失败:', e)
  } finally {
    loading.value = false
  }
}

// 格式化成交额
function formatVolume(vol) {
  if (!vol) return '-'
  if (vol >= 100000000) {
    return (vol / 100000000).toFixed(2) + '万亿'
  }
  return (vol / 10000).toFixed(1) + '亿'
}

// 更新图表
function updateChart() {
  if (!chartInstance || chartData.value.length === 0) return
  
  const dates = chartData.value.map(d => d.date?.substring(5) || '')
  const upData = chartData.value.map(d => d.upCount || 0)
  const downData = chartData.value.map(d => d.downCount || 0)
  
  const option = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e0e0e0',
      borderWidth: 1,
      textStyle: {
        color: '#333',
      },
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#999',
        },
      },
    },
    legend: {
      data: ['涨停数', '跌停数'],
      top: 10,
      textStyle: {
        color: '#666',
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '20%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: {
        lineStyle: {
          color: '#e0e0e0',
        },
      },
      axisLabel: {
        color: '#999',
        fontSize: 11,
      },
    },
    yAxis: {
      type: 'value',
      axisLine: {
        show: false,
      },
      axisLabel: {
        color: '#999',
      },
      splitLine: {
        lineStyle: {
          color: '#f0f0f0',
        },
      },
    },
    series: [
      {
        name: '涨停数',
        type: 'bar',
        data: upData,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#7ed957' },
            { offset: 1, color: '#5ab845' },
          ]),
          borderRadius: [4, 4, 0, 0],
        },
        barWidth: '30%',
      },
      {
        name: '跌停数',
        type: 'bar',
        data: downData,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#ff6b6b' },
            { offset: 1, color: '#e55555' },
          ]),
          borderRadius: [4, 4, 0, 0],
        },
        barWidth: '30%',
      },
    ],
  }
  
  chartInstance.setOption(option)
}

// 初始化图表
function initChart() {
  if (!chartRef.value) return
  
  chartInstance = echarts.init(chartRef.value)
  updateChart()
  
  // 响应式调整
  window.addEventListener('resize', () => {
    chartInstance?.resize()
  })
}

onMounted(() => {
  loadData()
  // 延迟初始化图表，确保 DOM 已渲染
  setTimeout(initChart, 100)
})

onUnmounted(() => {
  chartInstance?.dispose()
})

// 监听数据变化
watch(chartData, () => {
  updateChart()
})
</script>

<style scoped>
.sentiment-chart {
  height: 100%;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.title-icon {
  font-size: 18px;
}

.chart-legend {
  display: flex;
  gap: 16px;
}

.legend-item {
  font-size: 12px;
  color: var(--text-secondary);
}

.legend-item::before {
  content: '';
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 6px;
  vertical-align: middle;
}

.legend-item.up::before {
  background: #7ed957;
}

.legend-item.down::before {
  background: #ff6b6b;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.metric-box {
  background: var(--bg-primary);
  border-radius: 12px;
  padding: 14px;
  text-align: center;
}

.metric-label {
  display: block;
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.metric-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.metric-box.up .metric-value {
  color: var(--success);
}

.metric-box.down .metric-value {
  color: var(--danger);
}

.echarts-container {
  width: 100%;
  height: 280px;
}

.chart-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.loading-spinner {
  width: 30px;
  height: 30px;
  border: 3px solid var(--border-color);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 768px) {
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
