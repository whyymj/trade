<template>
  <div class="page-container">
    <a class="back-btn" @click="router.push('/')">← 返回基金列表</a>
    
    <div v-if="loading" class="loading">加载中...</div>
    <template v-else-if="fund">
      <div class="card" style="margin-bottom: 24px;">
        <div class="flex-between">
          <div>
            <h1 class="page-title" style="margin-bottom: 8px;">{{ fund.fund_name }}</h1>
            <span class="tag tag-primary">{{ fund.fund_type || '混合型' }}</span>
            <span style="margin-left: 12px; color: var(--text-muted);">{{ fund.fund_code }}</span>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 32px; font-weight: 700; color: var(--primary);">
              {{ latestNav?.unit_nav?.toFixed(4) || '--' }}
            </div>
            <div :class="(latestNav?.daily_return || 0) >= 0 ? 'positive' : 'negative'" style="font-size: 16px;">
              {{ (latestNav?.daily_return || 0) >= 0 ? '+' : '' }}{{ ((latestNav?.daily_return || 0) * 100).toFixed(2) }}%
            </div>
          </div>
        </div>
        
        <div v-if="fund.manager || fund.fund_scale" style="margin-top: 16px; color: var(--text-secondary);">
          <span v-if="fund.manager">经理: {{ fund.manager }}</span>
          <span v-if="fund.fund_scale" style="margin-left: 20px;">规模: {{ fund.fund_scale }}亿</span>
        </div>
      </div>
      
      <div class="nav-tabs" style="margin-bottom: 20px;">
        <button 
          class="nav-tab" 
          :class="{ active: activeTab === 'chart' }"
          @click="activeTab = 'chart'"
        >
          净值走势
        </button>
        <button 
          class="nav-tab" 
          :class="{ active: activeTab === 'indicators' }"
          @click="activeTab = 'indicators'"
        >
          业绩指标
        </button>
        <button 
          class="nav-tab" 
          :class="{ active: activeTab === 'benchmark' }"
          @click="activeTab = 'benchmark'"
        >
          基准对比
        </button>
        <button 
          class="nav-tab" 
          :class="{ active: activeTab === 'analysis' }"
          @click="activeTab = 'analysis'"
        >
          LLM分析
        </button>
      </div>
      
      <div v-if="activeTab === 'chart'" class="card">
        <div class="flex-between" style="margin-bottom: 16px;">
          <div class="filter-bar" style="margin-bottom: 0;">
            <button 
              v-for="range in chartRanges" 
              :key="range.value"
              class="filter-btn"
              :class="{ active: chartRange === range.value }"
              @click="changeChartRange(range.value)"
            >
              {{ range.label }}
            </button>
          </div>
        </div>
        <div ref="chartRef" class="chart-container"></div>
      </div>
      
      <div v-if="activeTab === 'indicators'" class="card">
        <div v-if="indicatorsLoading" class="loading">计算中...</div>
        <div v-else-if="indicators" class="metric-grid">
          <div class="metric-item">
            <div class="metric-label">近1月</div>
            <div class="metric-value" :class="(indicators.return_1m || 0) >= 0 ? 'positive' : 'negative'">
              {{ ((indicators.return_1m || 0) * 100).toFixed(2) }}%
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">近3月</div>
            <div class="metric-value" :class="(indicators.return_3m || 0) >= 0 ? 'positive' : 'negative'">
              {{ ((indicators.return_3m || 0) * 100).toFixed(2) }}%
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">近6月</div>
            <div class="metric-value" :class="(indicators.return_6m || 0) >= 0 ? 'positive' : 'negative'">
              {{ ((indicators.return_6m || 0) * 100).toFixed(2) }}%
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">近1年</div>
            <div class="metric-value" :class="(indicators.return_1y || 0) >= 0 ? 'positive' : 'negative'">
              {{ ((indicators.return_1y || 0) * 100).toFixed(2) }}%
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">年化收益</div>
            <div class="metric-value" :class="(indicators.annual_return || 0) >= 0 ? 'positive' : 'negative'">
              {{ ((indicators.annual_return || 0) * 100).toFixed(2) }}%
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">波动率</div>
            <div class="metric-value">{{ ((indicators.volatility || 0) * 100).toFixed(2) }}%</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">夏普比率</div>
            <div class="metric-value">{{ (indicators.sharpe_ratio || 0).toFixed(2) }}</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">最大回撤</div>
            <div class="metric-value negative">{{ ((indicators.max_drawdown || 0) * 100).toFixed(2) }}%</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">卡玛比率</div>
            <div class="metric-value">{{ (indicators.calmar_ratio || 0).toFixed(2) }}</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">胜率</div>
            <div class="metric-value">{{ (indicators.win_rate || 0).toFixed(1) }}%</div>
          </div>
        </div>
        <div v-else class="empty">暂无指标数据</div>
      </div>
      
      <div v-if="activeTab === 'benchmark'" class="card">
        <div v-if="benchmarkLoading" class="loading">加载中...</div>
        <template v-else-if="benchmark">
          <div class="metric-grid" style="margin-bottom: 24px;">
            <div class="metric-item">
              <div class="metric-label">Alpha</div>
              <div class="metric-value" :class="(benchmark.alpha || 0) >= 0 ? 'positive' : 'negative'">
                {{ (benchmark.alpha || 0).toFixed(4) }}
              </div>
            </div>
            <div class="metric-item">
              <div class="metric-label">Beta</div>
              <div class="metric-value">{{ (benchmark.beta || 0).toFixed(4) }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">超额收益</div>
              <div class="metric-value" :class="(benchmark.excess_return || 0) >= 0 ? 'positive' : 'negative'">
                {{ ((benchmark.excess_return || 0) * 100).toFixed(2) }}%
              </div>
            </div>
            <div class="metric-item">
              <div class="metric-label">跟踪误差</div>
              <div class="metric-value">{{ ((benchmark.tracking_error || 0) * 100).toFixed(2) }}%</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">信息比率</div>
              <div class="metric-value">{{ (benchmark.information_ratio || 0).toFixed(2) }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">R²</div>
              <div class="metric-value">{{ (benchmark.r_squared || 0).toFixed(4) }}</div>
            </div>
          </div>
          <div ref="benchmarkChartRef" class="chart-container"></div>
        </template>
        <div v-else class="empty">暂无基准对比数据</div>
      </div>
      
      <div v-if="activeTab === 'analysis'" class="card">
        <div class="nav-tabs" style="margin-bottom: 20px;">
          <button 
            class="nav-tab" 
            :class="{ active: analysisType === 'profile' }"
            @click="loadAnalysis('profile')"
          >
            概况分析
          </button>
          <button 
            class="nav-tab" 
            :class="{ active: analysisType === 'performance' }"
            @click="loadAnalysis('performance')"
          >
            业绩归因
          </button>
          <button 
            class="nav-tab" 
            :class="{ active: analysisType === 'risk' }"
            @click="loadAnalysis('risk')"
          >
            风险评估
          </button>
          <button 
            class="nav-tab" 
            :class="{ active: analysisType === 'advice' }"
            @click="loadAnalysis('advice')"
          >
            投资建议
          </button>
          <button 
            class="nav-tab" 
            :class="{ active: analysisType === 'report' }"
            @click="loadAnalysis('report')"
          >
            完整报告
          </button>
        </div>
        
        <div v-if="analysisLoading" class="loading">AI 分析中...</div>
        <div v-else-if="analysisResult" class="analysis-content">
          <pre style="white-space: pre-wrap; font-family: inherit;">{{ analysisResult }}</pre>
        </div>
        <div v-else class="empty">点击上方选项获取分析</div>
      </div>
    </template>
    <div v-else class="empty">基金不存在</div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { getFundList, getLatestNav, getFundNav, getIndicators, getBenchmark, getLlmAnalysis } from '@/api/fund'

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const fund = ref(null)
const latestNav = ref(null)
const activeTab = ref('chart')
const chartRange = ref('1y')
const chartRanges = [
  { label: '近1月', value: '1m' },
  { label: '近3月', value: '3m' },
  { label: '近6月', value: '6m' },
  { label: '近1年', value: '1y' },
  { label: '近3年', value: '3y' },
]

const chartRef = ref(null)
const benchmarkChartRef = ref(null)
let chartInstance = null
let benchmarkChartInstance = null

const indicators = ref(null)
const indicatorsLoading = ref(false)

const benchmark = ref(null)
const benchmarkLoading = ref(false)

const analysisType = ref('profile')
const analysisLoading = ref(false)
const analysisResult = ref('')
const analysisCache = ref({}) // 缓存分析结果

const fundCode = ref('')

async function changeChartRange(value) {
  console.log('changeChartRange:', value, 'chartInstance:', chartInstance)
  chartRange.value = value
  if (!chartInstance) {
    console.log('initChart called in changeChartRange')
    initChart()
  }
  await loadNavData()
}

function getDateRange(value) {
  const end = new Date()
  const start = new Date()
  switch (value) {
    case '1m': start.setMonth(start.getMonth() - 1); break
    case '3m': start.setMonth(start.getMonth() - 3); break
    case '6m': start.setMonth(start.getMonth() - 6); break
    case '1y': start.setFullYear(start.getFullYear() - 1); break
    case '3y': start.setFullYear(start.getFullYear() - 3); break
    default: start.setFullYear(start.getFullYear() - 1)
  }
  return {
    start: start.toISOString().split('T')[0],
    end: end.toISOString().split('T')[0]
  }
}

async function loadNavData() {
  if (!fundCode.value) return
  const range = getDateRange(chartRange.value)
  console.log('loadNavData called, range:', range)
  try {
    const res = await getFundNav(fundCode.value, range)
    const data = res.data || []
    console.log('loadNavData: got', data.length, 'records')
    console.log('chartInstance:', chartInstance)
    
    if (chartInstance && data.length > 0) {
      const dates = data.map(d => d.nav_date)
      const navs = data.map(d => d.unit_nav)
      const returns = data.map(d => d.daily_return ? d.daily_return * 100 : 0)
      
      console.log('Setting chart option, dates:', dates.slice(0, 3), 'navs:', navs.slice(0, 3))
      
      chartInstance.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['单位净值', '日涨跌幅'] },
        xAxis: { type: 'category', data: dates },
        yAxis: [
          { type: 'value', name: '净值' },
          { type: 'value', name: '涨跌幅%' }
        ],
        series: [
          { name: '单位净值', type: 'line', data: navs, smooth: true },
          { name: '日涨跌幅', type: 'bar', yAxisIndex: 1, data: returns }
        ]
      })
      console.log('Chart option set')
    } else {
      console.warn('No data or chartInstance is null')
    }
  } catch (e) {
    console.error('加载净值数据失败:', e)
  }
}

async function loadIndicators() {
  indicatorsLoading.value = true
  try {
    indicators.value = await getIndicators(fundCode.value)
  } catch (e) {
    console.error('加载指标失败:', e)
    indicators.value = null
  } finally {
    indicatorsLoading.value = false
  }
}

async function loadBenchmark() {
  benchmarkLoading.value = true
  try {
    benchmark.value = await getBenchmark(fundCode.value)
    
    if (benchmark.value && benchmarkChartInstance) {
      const dates = benchmark.value.dates || []
      const fundReturn = benchmark.value.fund_cum_return || []
      const benchmarkReturn = benchmark.value.benchmark_cum_return || []
      
      benchmarkChartInstance.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['基金', '沪深300'] },
        xAxis: { type: 'category', data: dates },
        yAxis: { type: 'value', name: '累计收益率%' },
        series: [
          { name: '基金', type: 'line', data: fundReturn, smooth: true },
          { name: '沪深300', type: 'line', data: benchmarkReturn, smooth: true }
        ]
      })
    }
  } catch (e) {
    console.error('加载基准对比失败:', e)
    benchmark.value = null
  } finally {
    benchmarkLoading.value = false
  }
}

async function loadAnalysis(type) {
  // 检查缓存
  const cacheKey = `${fundCode.value}_${type}`
  if (analysisCache.value[cacheKey]) {
    analysisType.value = type
    analysisResult.value = analysisCache.value[cacheKey]
    return
  }
  
  // 防止重复请求
  if (analysisLoading.value) return
  
  analysisType.value = type
  analysisLoading.value = true
  try {
    const res = await getLlmAnalysis(fundCode.value, type)
    const result = res.analysis || res.advice || res.report || '暂无分析结果'
    analysisResult.value = result
    // 缓存结果
    analysisCache.value[cacheKey] = result
  } catch (e) {
    console.error('加载分析失败:', e)
    analysisResult.value = '分析失败: ' + e.message
  } finally {
    analysisLoading.value = false
  }
}

function initChart() {
  console.log('initChart called')
  console.log('  chartRef:', chartRef.value)
  console.log('  benchmarkChartRef:', benchmarkChartRef.value)
  if (chartRef.value) {
    chartInstance = echarts.init(chartRef.value)
    console.log('chartInstance initialized:', chartInstance)
  }
  if (benchmarkChartRef.value) {
    benchmarkChartInstance = echarts.init(benchmarkChartRef.value)
    console.log('benchmarkChartInstance initialized:', benchmarkChartInstance)
  }
}

onMounted(async () => {
  fundCode.value = route.params.code
  loading.value = true
  
  try {
    const res = await getFundList({ page: 1, size: 1000 })
    fund.value = (res.data || []).find(f => f.fund_code === fundCode.value)
    
    if (fund.value) {
      latestNav.value = await getLatestNav(fundCode.value).catch(() => null)
    } else {
      console.warn('Fund not found in list!')
    }
  } catch (e) {
    console.error('加载失败:', e)
  } finally {
    loading.value = false
  }
  
  // 页面加载完成后初始化图表
  await nextTick()
  initChart()
  await loadNavData()
})

watch(fundCode, async (newCode) => {
  if (newCode) {
    // 切换基金时清空分析缓存
    analysisCache.value = {}
  }
  if (newCode && activeTab.value === 'chart') {
    await nextTick()
    initChart()
    await loadNavData()
  }
}, { immediate: true })

watch(activeTab, async (tab) => {
  await nextTick()
  if (tab === 'chart') {
    initChart()
    await loadNavData()
  } else if (tab === 'indicators') {
    await loadIndicators()
  } else if (tab === 'benchmark') {
    await loadBenchmark()
  }
})
</script>
