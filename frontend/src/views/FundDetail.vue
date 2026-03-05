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
            <span v-for="tag in (fund.industry_tags || []).slice(0, 3)" :key="tag" class="tag tag-secondary" style="margin-left: 6px;">
              {{ tag }}
            </span>
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
        <button class="nav-tab" :class="{ active: activeTab === 'chart' }" @click="activeTab = 'chart'">
          📈 净值走势
        </button>
        <button class="nav-tab" :class="{ active: activeTab === 'holdings' }" @click="activeTab = 'holdings'">
          🏆 持仓明细
        </button>
        <button class="nav-tab" :class="{ active: activeTab === 'performance' }" @click="activeTab = 'performance'">
          📊 业绩分析
        </button>
        <button class="nav-tab" :class="{ active: activeTab === 'analysis' }" @click="activeTab = 'analysis'">
          🤖 AI分析
        </button>
      </div>
      
      <div v-if="activeTab === 'holdings'" class="card">
        <h3 style="margin: 0 0 16px 0; color: var(--primary);">🏆 持仓明细</h3>
        
        <div v-if="holdingsLoading" class="loading">加载中...</div>
        <template v-else-if="holdingsData && holdingsData.length > 0">
          <div class="holdings-table">
            <table style="width: 100%; border-collapse: collapse;">
              <thead>
                <tr style="background: #f5f7fa;">
                  <th style="padding: 12px; text-align: left; border-bottom: 2px solid #eee;">序号</th>
                  <th style="padding: 12px; text-align: left; border-bottom: 2px solid #eee;">股票代码</th>
                  <th style="padding: 12px; text-align: left; border-bottom: 2px solid #eee;">股票名称</th>
                  <th style="padding: 12px; text-align: right; border-bottom: 2px solid #eee;">持仓占比</th>
                  <th style="padding: 12px; text-align: right; border-bottom: 2px solid #eee;">涨跌幅</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(stock, index) in holdingsData" :key="stock.stock_code" style="border-bottom: 1px solid #f0f0f0;">
                  <td style="padding: 12px;">{{ index + 1 }}</td>
                  <td style="padding: 12px; color: #666;">{{ stock.stock_code }}</td>
                  <td style="padding: 12px; font-weight: 600;">{{ stock.stock_name }}</td>
                  <td style="padding: 12px; text-align: right; color: var(--primary); font-weight: 600;">
                    {{ stock.hold_ratio }}%
                  </td>
                  <td style="padding: 12px; text-align: right;" :class="(stock.change_pct || 0) >= 0 ? 'positive' : 'negative'">
                    {{ (stock.change_pct || 0) >= 0 ? '+' : '' }}{{ stock.change_pct || 0 }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
        <div v-else class="empty">暂无持仓数据</div>
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
        </div>
      </div>
      
      <div v-if="activeTab === 'performance'" class="performance-tab">
        <div class="card">
          <h3 style="margin: 0 0 16px 0; color: var(--primary);">📈 业绩指标</h3>
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
          </template>
          <div v-else class="empty">暂无指标数据</div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
          <h3 style="margin: 0 0 16px 0; color: var(--primary);">📉 基准对比（沪深300）</h3>
          <div v-if="benchmarkLoading" class="loading">加载中...</div>
          <template v-else-if="benchmark">
            <div class="metric-grid" style="margin-bottom: 20px;">
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
      </div>
      
      <div v-if="activeTab === 'analysis'" class="card">
        <div class="analysis-header">
          <h3 style="margin: 0;">🤖 AI 分析报告</h3>
          <button class="btn btn-primary" @click="triggerAnalysis" :disabled="analysisLoading">
            {{ analysisLoading ? '分析中...' : '重新分析' }}
          </button>
        </div>
        
        <div v-if="analysisLoading" class="analysis-loading">
          <div class="loading-steps">
            <div class="step" :class="{ active: analysisStep >= 1 }">
              <span class="step-icon">🔍</span>
              <span>正在分析基金概况...</span>
            </div>
            <div class="step" :class="{ active: analysisStep >= 2 }">
              <span class="step-icon">📊</span>
              <span>正在分析业绩表现...</span>
            </div>
            <div class="step" :class="{ active: analysisStep >= 3 }">
              <span class="step-icon">🏭</span>
              <span>正在推断行业配置...</span>
            </div>
            <div class="step" :class="{ active: analysisStep >= 4 }">
              <span class="step-icon">💡</span>
              <span>正在生成投资建议...</span>
            </div>
          </div>
          <div class="thinking-process" v-if="thinkingProcess">
            <div class="thinking-label">💭 AI 思考过程：</div>
            <div class="thinking-content">{{ thinkingProcess }}</div>
          </div>
        </div>
        
        <div v-else-if="analysisResult" class="analysis-content" v-html="analysisRendered"></div>
        <div v-else class="empty">
          <p>暂无分析结果，点击"重新分析"获取AI分析报告</p>
        </div>
        
        <div v-if="fundNewsSummary" class="industry-news-section" style="margin-top: 24px;">
          <h4 style="margin: 0 0 12px 0;">📰 相关行业新闻</h4>
          <div class="summary-header" style="display: flex; justify-content: space-between; margin-bottom: 12px;">
            <div class="industry-tags">
              <span v-for="ind in fundNewsSummary.industries" :key="ind.industry" class="tag tag-primary">
                {{ ind.industry }} ({{ ind.confidence }}%)
              </span>
            </div>
            <div class="sentiment-badge" :class="fundNewsSummary.sentiment">
              {{ getSentimentText(fundNewsSummary.sentiment) }}
            </div>
          </div>
          
          <div v-if="investmentAdvice" class="advice-card" style="padding: 16px; background: #f0f9ff; border-radius: 8px; margin-bottom: 16px;">
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
              <div style="flex: 1; min-width: 150px;">
                <div style="font-size: 12px; color: #ff4d4f;">短期(1周)</div>
                <div style="font-size: 13px;">{{ investmentAdvice.short_term }}</div>
              </div>
              <div style="flex: 1; min-width: 150px;">
                <div style="font-size: 12px; color: #fa8c16;">中期(1-3月)</div>
                <div style="font-size: 13px;">{{ investmentAdvice.medium_term }}</div>
              </div>
              <div style="flex: 1; min-width: 150px;">
                <div style="font-size: 12px; color: #52c41a;">长期(6个月+)</div>
                <div style="font-size: 13px;">{{ investmentAdvice.long_term }}</div>
              </div>
            </div>
          </div>
          
          <div v-if="fundNewsSummary.latest_news && fundNewsSummary.latest_news.length" class="news-list">
            <div v-for="news in fundNewsSummary.latest_news.slice(0, 5)" :key="news.title" class="news-item" style="padding: 8px 0; border-bottom: 1px solid #eee;">
              <span class="tag tag-secondary" style="margin-right: 8px;">{{ news.industry }}</span>
              <a :href="news.url" target="_blank" style="color: var(--text-primary);">{{ news.title }}</a>
            </div>
          </div>
        </div>
      </div>
    </template>
    <div v-else class="empty">基金不存在</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { marked } from 'marked'
import { ElMessage } from 'element-plus'
import { getFundList, getLatestNav, getFundNav, getIndicators, getBenchmark, getLlmAnalysis, getFundCycle } from '@/api/fund'
import { getSentimentText, sanitizeHtml } from '@/utils/formatters'

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
let navChartData = null

const analysisLoading = ref(false)
const analysisStep = ref(0)
const thinkingProcess = ref('')
const analysisResult = ref('')
const analysisRendered = computed(() => {
  if (!analysisResult.value) return ''
  return sanitizeHtml(marked(analysisResult.value))
})

const fundCode = ref('')

const fundNewsSummary = ref(null)
const investmentAdvice = ref(null)

const holdingsData = ref([])
const holdingsLoading = ref(false)

async function loadHoldings() {
  if (!fundCode.value) return
  holdingsLoading.value = true
  try {
    const res = await fetch(`/api/fund/holdings/${fundCode.value}`)
    const data = await res.json()
    if (data.code === 0 && data.data) {
      holdingsData.value = data.data.holdings || []
    }
  } catch (e) {
    console.error('加载持仓数据失败:', e)
    holdingsData.value = []
  } finally {
    holdingsLoading.value = false
  }
}

async function loadIndustryNews() {
  if (!fundCode.value) return
  try {
    const res = await fetch(`/api/fund-news/summary/${fundCode.value}?days=7`)
    const data = await res.json()
    if (data.code === 0) {
      fundNewsSummary.value = data.data
    }
  } catch (e) {
    console.error('加载行业新闻失败:', e)
  }
}

async function loadInvestmentAdvice() {
  if (!fundCode.value) return
  try {
    const res = await fetch(`/api/investment-advice/${fundCode.value}?days=7`)
    const data = await res.json()
    if (data.code === 0) {
      investmentAdvice.value = data.data
    }
  } catch (e) {
    console.error('获取投资建议失败:', e)
  }
}

async function changeChartRange(value) {
  chartRange.value = value
  if (!chartInstance) initChart()
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
  const toLocalDate = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  return { start: toLocalDate(start), end: toLocalDate(end) }
}

function renderChart() {
  if (chartInstance && navChartData && navChartData.dates) {
    const dates = navChartData.dates
    const navs = navChartData.navs
    const returns = navChartData.returns
    
    let markLine = null
    if (selectedCycle.value) {
      const cycle = cycleAnalysis.value?.find(c => c.rank === selectedCycle.value)
      if (cycle) {
        const periodDays = cycle.period_days
        const lineData = []
        for (let i = 0; i < dates.length; i += Math.round(periodDays)) {
          lineData.push({ xAxis: i })
        }
        markLine = { symbol: ['none', 'none'], lineStyle: { type: 'dashed', color: '#ff6b6b', width: 2 }, data: lineData }
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
  }
}

async function loadNavData() {
  if (!fundCode.value) return
  const range = getDateRange(chartRange.value)
  try {
    const res = await getFundNav(fundCode.value, range)
    const data = res.data || []
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
  selectedCycle.value = selectedCycle.value === cycle.rank ? null : cycle.rank
  if (chartInstance && navChartData && navChartData.dates) {
    renderChart()
  }
}

async function triggerAnalysis() {
  analysisLoading.value = true
  analysisStep.value = 0
  thinkingProcess.value = ''
  
  analysisStep.value = 1
  thinkingProcess.value = '正在分析基金的基本信息和投资策略...'
  await new Promise(r => setTimeout(r, 800))
  
  analysisStep.value = 2
  thinkingProcess.value += '\n正在计算基金的历史业绩表现和风险指标...'
  await new Promise(r => setTimeout(r, 800))
  
  analysisStep.value = 3
  thinkingProcess.value += '\n正在根据基金名称和持仓特征推断行业配置...'
  await new Promise(r => setTimeout(r, 800))
  
  analysisStep.value = 4
  thinkingProcess.value += '\n正在生成综合投资建议...'
  
  try {
    const res = await getLlmAnalysis(fundCode.value, 'report')
    const result = res.report || res.analysis || res.advice || '暂无分析结果'
    analysisResult.value = result
    thinkingProcess.value = ''
  } catch (e) {
    console.error('加载分析失败:', e)
    analysisResult.value = '分析失败: ' + e.message
    ElMessage.error('AI分析失败，请稍后重试')
  } finally {
    analysisLoading.value = false
    analysisStep.value = 0
  }
}

async function loadAnalysis() {
  const navDate = latestNav.value?.nav_date || ''
  const cacheKey = `llm_report_${fundCode.value}_${navDate}`
  
  const cached = localStorage.getItem(cacheKey)
  if (cached) {
    analysisResult.value = cached
    return
  }
  
  await triggerAnalysis()
  if (analysisResult.value) {
    localStorage.setItem(cacheKey, analysisResult.value)
  }
}

function initChart() {
  if (chartRef.value) {
    chartInstance = echarts.init(chartRef.value)
  }
  if (benchmarkChartRef.value) {
    benchmarkChartInstance = echarts.init(benchmarkChartRef.value)
  }
}

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
  if (benchmarkChartInstance) {
    benchmarkChartInstance.dispose()
    benchmarkChartInstance = null
  }
})

onMounted(async () => {
  fundCode.value = route.params.code
  loading.value = true
  
  try {
    const res = await getFundList({ page: 1, size: 1000 })
    fund.value = (res.data || []).find(f => f.fund_code === fundCode.value)
    
    if (fund.value) {
      latestNav.value = await getLatestNav(fundCode.value).catch(() => null)
    }
  } catch (e) {
    console.error('加载失败:', e)
  } finally {
    loading.value = false
  }
  
  await nextTick()
  initChart()
  await loadNavData()
  await loadCycleData()
  await loadAnalysis()
  await loadIndustryNews()
  await loadInvestmentAdvice()
})

watch(activeTab, async (tab) => {
  await nextTick()
  if (tab === 'chart') {
    initChart()
    await loadNavData()
    await loadCycleData()
  } else if (tab === 'performance') {
    await loadIndicators()
    await loadBenchmark()
  } else if (tab === 'holdings') {
    await loadHoldings()
  }
})
</script>

<style>
.performance-tab .metric-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
}

.performance-tab .metric-item {
  text-align: center;
  padding: 12px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.performance-tab .metric-label {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.performance-tab .metric-value {
  font-size: 18px;
  font-weight: 600;
}

.analysis-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.analysis-loading {
  padding: 24px;
  text-align: center;
}

.loading-steps {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
}

.step {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: var(--bg-secondary);
  border-radius: 8px;
  opacity: 0.5;
  transition: all 0.3s;
}

.step.active {
  opacity: 1;
  background: #e6f7ff;
}

.step-icon {
  font-size: 20px;
}

.thinking-process {
  text-align: left;
  padding: 16px;
  background: #f9f9f9;
  border-radius: 8px;
  margin-top: 16px;
}

.thinking-label {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--primary);
}

.thinking-content {
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
}

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

.sentiment-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

.sentiment-badge.positive { background: #f6ffed; color: #52c41a; }
.sentiment-badge.negative { background: #fff2f0; color: #ff4d4f; }
.sentiment-badge.neutral { background: #fafafa; color: #666; }
</style>
