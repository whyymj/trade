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
        
        <div v-if="cycleLoading" class="loading" style="margin-top: 20px;">周期分析计算中...</div>
        <div v-else-if="cycleAnalysis && cycleAnalysis.length > 0" class="cycle-analysis" style="margin-top: 20px; padding: 16px; background: #f5f7fa; border-radius: 8px;">
          <div style="font-size: 14px; font-weight: bold; margin-bottom: 12px;">📊 周期分析（近1年数据）</div>
          <div style="font-size: 12px; color: #666; margin-bottom: 12px;">
            通过频域分析发现的价格波动规律，强度越高越可靠
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <div 
              v-for="cycle in cycleAnalysis" 
              :key="cycle.rank" 
              :style="{
                padding: '10px', 
                background: selectedCycle === cycle.rank ? '#e6f7ff' : 'white', 
                borderRadius: '6px', 
                borderLeft: '3px solid ' + (selectedCycle === cycle.rank ? '#1890ff' : '#409eff'),
                cursor: 'pointer'
              }"
              @click="toggleCycle(cycle)"
            >
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold;" :style="{color: selectedCycle === cycle.rank ? '#1890ff' : '#409eff'}">
                  {{ selectedCycle === cycle.rank ? '✓ ' : '' }}🏈 主周期{{ cycle.rank }}: {{ cycle.period_days }}天
                </span>
                <span style="font-size: 12px; color: #909399;">强度: {{ (cycle.power * 100).toFixed(2) }}%</span>
              </div>
              <div style="font-size: 13px; color: #606266; margin-top: 4px;">{{ cycle.explanation }}</div>
            </div>
          </div>
          <div style="margin-top: 12px; font-size: 12px; color: #909399; background: #fff; padding: 8px; border-radius: 4px;">
            <div style="font-weight: bold; margin-bottom: 4px;">💡 什么是周期？</div>
            <div>• <strong>主周期1/2/3</strong> = 按强度排序的不同周期跨度</div>
            <div>• 主周期1 = 最强波动规律；主周期2/3 = 次要规律</div>
            <div>• 点击周期可在曲线上显示分界线</div>
            <div>• <strong>周期跨度</strong> = 波动重复的时间间隔（天数）</div>
          </div>
        </div>
      </div>
      
      <div v-if="activeTab === 'indicators'" class="card">
        <div v-if="indicatorsLoading" class="loading">计算中...</div>
        <template v-else-if="indicators">
          <div class="metric-grid">
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
          
          <div style="margin-top: 24px; padding: 16px; background: #f0f9ff; border-radius: 8px; font-size: 13px;">
            <div style="font-weight: bold; margin-bottom: 12px;">📖 指标说明</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
              <div><strong>年化收益</strong>: 折算成年度的收益率，方便跨年限比较</div>
              <div><strong>波动率</strong>: 收益变化幅度，越高风险越大</div>
              <div><strong>夏普比率</strong>: 每承担1分风险获得的超额收益，越高越好（>1优秀）</div>
              <div><strong>最大回撤</strong>: 从最高点到最低点的最大跌幅，越小越安全</div>
              <div><strong>卡玛比率</strong>: 年化收益/最大回撤，越高说明收益风险比越好</div>
              <div><strong>胜率</strong>: 获得正收益的天数占比，越高越稳定</div>
            </div>
          </div>
        </template>
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
          
          <div style="margin-top: 16px; padding: 16px; background: #f0f9ff; border-radius: 8px; font-size: 13px;">
            <div style="font-weight: bold; margin-bottom: 12px;">📖 基准对比说明（对比沪深300指数）</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
              <div><strong>Alpha (α)</strong>: 超越市场的额外收益，正值表示跑赢市场</div>
              <div><strong>Beta (β)</strong>: 跟随市场的程度，>1波动更大，<1波动更小</div>
              <div><strong>超额收益</strong>: 相比沪深300的额外收益</div>
              <div><strong>跟踪误差</strong>: 与基准收益的偏离程度，越小越稳定</div>
              <div><strong>信息比率</strong>: 超额收益/跟踪误差，越高说明主动管理能力越强</div>
              <div><strong>R²</strong>: 与基准的相关程度，接近1表示与基准走势相似</div>
            </div>
          </div>
          <div ref="benchmarkChartRef" class="chart-container"></div>
        </template>
        <div v-else class="empty">暂无基准对比数据</div>
      </div>
      
      <div v-if="activeTab === 'analysis'" class="card">
        <div v-if="analysisLoading" class="loading">AI 分析中，请稍候...</div>
        <div v-else-if="analysisResult" class="analysis-content" v-html="analysisRendered"></div>
        <div v-else class="empty">暂无分析结果</div>
      </div>
    </template>
    <div v-else class="empty">基金不存在</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { marked } from 'marked'
import { getFundList, getLatestNav, getFundNav, getIndicators, getBenchmark, getLlmAnalysis, getFundCycle } from '@/api/fund'

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

const cycleAnalysis = ref(null)
const cycleLoading = ref(false)
const selectedCycle = ref(null)
let navChartData = null // 存储净值数据用于绘制周期线

const analysisType = ref('profile')
const analysisLoading = ref(false)
const analysisResult = ref('')
const analysisRendered = computed(() => {
  if (!analysisResult.value) return ''
  return marked(analysisResult.value)
})
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
  const toLocalDate = (d) => {
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${y}-${m}-${day}`
  }
  return {
    start: toLocalDate(start),
    end: toLocalDate(end)
  }
}

function renderChart() {
  if (chartInstance && navChartData && navChartData.dates) {
    const dates = navChartData.dates
    const navs = navChartData.navs
    const returns = navChartData.returns
    
    console.log('renderChart: dates:', dates.slice(0, 3), 'navs:', navs.slice(0, 3))
    
    let markLine = null
    if (selectedCycle.value) {
      const cycle = cycleAnalysis.value?.find(c => c.rank === selectedCycle.value)
      if (cycle) {
        const periodDays = cycle.period_days
        const lineData = []
        for (let i = 0; i < dates.length; i += Math.round(periodDays)) {
          lineData.push({ xAxis: i })
        }
        console.log('renderChart: periodDays:', periodDays, 'lines:', lineData.length)
        markLine = {
          symbol: ['none', 'none'],
          lineStyle: { type: 'dashed', color: '#ff6b6b', width: 2 },
          data: lineData
        }
      }
    }
    
    chartInstance.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['单位净值', '日涨跌幅'] },
      xAxis: { type: 'category', data: dates },
      yAxis: [
        { type: 'value', name: '净值' },
        { type: 'value', name: '涨跌幅%' }
      ],
      series: [
        { name: '单位净值', type: 'line', data: navs, smooth: true, markLine },
        { name: '日涨跌幅', type: 'bar', yAxisIndex: 1, data: returns }
      ]
    })
    console.log('renderChart: chart option set')
  } else {
    console.warn('renderChart: No data or chartInstance is null')
  }
}

async function loadNavData() {
  if (!fundCode.value) return
  const range = getDateRange(chartRange.value)
  console.log('loadNavData called, range:', range)
  try {
    console.log('Calling getFundNav with:', fundCode.value, range)
    const res = await getFundNav(fundCode.value, range)
    console.log('getFundNav response:', res)
    const data = res.data || []
    console.log('loadNavData: got', data.length, 'records')
    console.log('chartInstance:', chartInstance)
    
    navChartData = {
      dates: data.map(d => d.nav_date),
      navs: data.map(d => d.unit_nav),
      returns: data.map(d => d.daily_return ? d.daily_return * 100 : 0)
    }
    
    renderChart()
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

async function loadCycleData() {
  if (!fundCode.value) return
  cycleLoading.value = true
  try {
    const res = await getFundCycle(fundCode.value, 365)
    cycleAnalysis.value = res.dominant_periods || []
  } catch (e) {
    console.error('加载周期分析失败:', e)
    cycleAnalysis.value = []
  } finally {
    cycleLoading.value = false
  }
}

function toggleCycle(cycle) {
  console.log('toggleCycle called:', cycle)
  if (selectedCycle.value === cycle.rank) {
    selectedCycle.value = null
    console.log('Deselected cycle')
  } else {
    selectedCycle.value = cycle.rank
    console.log('Selected cycle:', cycle.rank)
  }
  // 重新渲染完整图表
  if (chartInstance && navChartData && navChartData.dates) {
    renderChart()
  }
}

async function loadAnalysis(type = 'report') {
  // 固定使用完整报告
  type = 'report'
  
  // 构建缓存键：包含基金代码和最新净值日期
  const navDate = latestNav.value?.nav_date || ''
  const cacheKey = `llm_report_${fundCode.value}_${navDate}`
  
  // 检查 localStorage 缓存
  const cached = localStorage.getItem(cacheKey)
  if (cached) {
    analysisResult.value = cached
    return
  }
  
  // 检查内存缓存
  const memCacheKey = `${fundCode.value}_${type}`
  if (analysisCache.value[memCacheKey]) {
    analysisResult.value = analysisCache.value[memCacheKey]
    return
  }
  
  // 防止重复请求
  if (analysisLoading.value) return
  
  analysisLoading.value = true
  try {
    const res = await getLlmAnalysis(fundCode.value, 'report')
    const result = res.report || res.analysis || res.advice || '暂无分析结果'
    analysisResult.value = result
    
    // 缓存到 localStorage（包含数据日期，过期则自动失效）
    localStorage.setItem(cacheKey, result)
    // 同时缓存到内存
    analysisCache.value[memCacheKey] = result
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
  await loadCycleData() // 加载周期分析
  
  // 自动加载完整报告
  await loadAnalysis('report')
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
    await loadCycleData()
  } else if (tab === 'indicators') {
    await loadIndicators()
  } else if (tab === 'benchmark') {
    await loadBenchmark()
  }
})
</script>

<style>
.analysis-content {
  line-height: 1.8;
  font-size: 14px;
}
.analysis-content h1 { font-size: 20px; margin: 16px 0 12px; }
.analysis-content h2 { font-size: 18px; margin: 14px 0 10px; }
.analysis-content h3 { font-size: 16px; margin: 12px 0 8px; }
.analysis-content p { margin: 8px 0; }
.analysis-content ul, .analysis-content ol { margin: 8px 0; padding-left: 20px; }
.analysis-content li { margin: 4px 0; }
.analysis-content code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
.analysis-content table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.analysis-content th, .analysis-content td { border: 1px solid #eee; padding: 8px; text-align: left; }
.analysis-content th { background: #f9f9f9; }
.analysis-content blockquote { border-left: 3px solid #409eff; margin: 12px 0; padding: 8px 16px; background: #f9f9f9; }
</style>
