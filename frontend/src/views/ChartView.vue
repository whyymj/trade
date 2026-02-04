<template>
  <div class="container">
    <div class="nav-row">
      <el-link type="primary" underline="never" @click="goBack">← 返回列表</el-link>
    </div>

    <template v-if="symbol">
      <h1>{{ chartTitle }}</h1>
      <div class="range-row">
        <span class="range-label">时间范围：</span>
        <el-radio-group v-model="rangeKey" size="default" @change="onRangeChange">
          <el-radio-button label="5y">5年</el-radio-button>
          <el-radio-button label="3y">3年</el-radio-button>
          <el-radio-button label="1y">1年</el-radio-button>
          <el-radio-button label="6m">6个月</el-radio-button>
          <el-radio-button label="3m">3个月</el-radio-button>
          <el-radio-button label="1m">1个月</el-radio-button>
        </el-radio-group>
        <el-button
          type="primary"
          :loading="analyzeLoading"
          class="analyze-btn"
          @click="runAnalyze"
        >
          分析
        </el-button>
        <el-button
          type="success"
          :loading="exportLoading"
          :disabled="!symbol"
          class="export-btn"
          @click="downloadReport"
        >
          下载报告
        </el-button>
      </div>
      <div v-loading="loading" class="chart-wrap">
        <div class="chart-title">价格走势（开盘 / 收盘 / 最高 / 最低）</div>
        <div ref="priceChartRef" class="chart"></div>
      </div>
      <div class="chart-wrap">
        <div class="chart-title">成交量</div>
        <div ref="volumeChartRef" class="chart volume"></div>
      </div>

      <!-- 分析结果 -->
      <div v-if="analyzeResult" class="chart-wrap analysis-section">
        <div class="chart-title">分析结果</div>
        <div class="analysis-summary">
          <el-descriptions :column="2" border size="small" class="summary-desc">
            <el-descriptions-item label="股票">{{ analyzeResult.summary?.stock_info?.name ?? '-' }}</el-descriptions-item>
            <el-descriptions-item label="数据区间">{{ analyzeResult.summary?.stock_info?.start_date ?? '-' }} ~ {{ analyzeResult.summary?.stock_info?.end_date ?? '-' }}</el-descriptions-item>
            <el-descriptions-item label="区间涨跌幅">{{ analyzeResult.summary?.stock_info?.total_return != null ? analyzeResult.summary.stock_info.total_return.toFixed(2) + '%' : '-' }}</el-descriptions-item>
            <el-descriptions-item label="最大回撤">{{ analyzeResult.summary?.time_domain?.max_drawdown != null ? (analyzeResult.summary.time_domain.max_drawdown * 100).toFixed(2) + '%' : '-' }}</el-descriptions-item>
            <el-descriptions-item label="Hurst 指数">{{ analyzeResult.summary?.complexity?.hurst_exponent != null ? analyzeResult.summary.complexity.hurst_exponent.toFixed(3) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="样本熵">{{ analyzeResult.summary?.complexity?.sample_entropy != null ? analyzeResult.summary.complexity.sample_entropy.toFixed(3) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="ARIMA R²">{{ analyzeResult.summary?.arima?.r_squared != null ? analyzeResult.summary.arima.r_squared.toFixed(4) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="主周期(天)">{{ analyzeResult.summary?.frequency_domain?.dominant_periods?.[0]?.period_days != null ? analyzeResult.summary.frequency_domain.dominant_periods[0].period_days.toFixed(1) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="RSI(14)">{{ analyzeResult.summary?.technical?.rsi != null ? analyzeResult.summary.technical.rsi.toFixed(1) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="VaR(95%)">{{ analyzeResult.summary?.technical?.var_95 != null ? (analyzeResult.summary.technical.var_95 * 100).toFixed(2) + '%' : '-' }}</el-descriptions-item>
            <el-descriptions-item label="MFI(14)">{{ analyzeResult.summary?.technical?.mfi != null ? analyzeResult.summary.technical.mfi.toFixed(1) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="Aroon上/下">{{ analyzeResult.summary?.technical?.aroon_up != null && analyzeResult.summary?.technical?.aroon_down != null ? analyzeResult.summary.technical.aroon_up.toFixed(0) + ' / ' + analyzeResult.summary.technical.aroon_down.toFixed(0) : '-' }}</el-descriptions-item>
            <el-descriptions-item label="信号">{{ analyzeResult.summary?.technical?.signals?.combined || '-' }}</el-descriptions-item>
          </el-descriptions>
        </div>
        <!-- 分析曲线：频域 / 时域 STL·ACF / 复杂度 Hurst -->
        <div v-if="analyzeResult?.charts" class="analysis-charts">
          <div class="analysis-chart-item">
            <div class="analysis-chart-title">功率谱密度（频域）</div>
            <div class="analysis-chart-desc">频率 vs 能量，峰值对应主要周期（如月周期、周周期）</div>
            <div ref="freqChartRef" class="analysis-chart"></div>
          </div>
          <div class="analysis-chart-item">
            <div class="analysis-chart-title">STL 分解（趋势 / 季节 / 残差）</div>
            <div class="analysis-chart-desc">价格 = 趋势 + 周期性波动 + 残差，用于观察长期走向与周期</div>
            <div ref="stlChartRef" class="analysis-chart"></div>
          </div>
          <div class="analysis-chart-item">
            <div class="analysis-chart-title">自相关 ACF</div>
            <div class="analysis-chart-desc">序列与自身滞后 k 期的相关性，衰减快慢反映可预测性</div>
            <div ref="acfChartRef" class="analysis-chart"></div>
          </div>
          <div class="analysis-chart-item">
            <div class="analysis-chart-title">赫斯特指数 R/S</div>
            <div class="analysis-chart-desc">H &lt; 0.5 偏反转，H &gt; 0.5 偏趋势；散点为 R/S，红线为拟合</div>
            <div ref="hurstChartRef" class="analysis-chart"></div>
          </div>
          <template v-if="analyzeResult?.charts?.technical">
            <div class="analysis-chart-item">
              <div class="analysis-chart-title">RSI（相对强弱）</div>
              <div class="analysis-chart-desc">0～100，&gt;70 超买 &lt;30 超卖</div>
              <div ref="rsiChartRef" class="analysis-chart"></div>
            </div>
            <div class="analysis-chart-item">
              <div class="analysis-chart-title">MACD</div>
              <div class="analysis-chart-desc">快慢线差、信号线、柱状图；柱由负转正可视为金叉</div>
              <div ref="macdChartRef" class="analysis-chart"></div>
            </div>
            <div class="analysis-chart-item">
              <div class="analysis-chart-title">布林带</div>
              <div class="analysis-chart-desc">中轨=均线，上下轨=±2 倍标准差；价格触及上/下轨可参考超买超卖</div>
              <div ref="bbChartRef" class="analysis-chart"></div>
            </div>
            <div class="analysis-chart-item">
              <div class="analysis-chart-title">滚动波动率（年化）</div>
              <div class="analysis-chart-desc">20 日收益率标准差年化，衡量近期波动程度</div>
              <div ref="volChartRef" class="analysis-chart"></div>
            </div>
            <div v-if="analyzeResult?.charts?.technical?.obv?.length" class="analysis-chart-item">
              <div class="analysis-chart-title">OBV（能量潮）</div>
              <div class="analysis-chart-desc">价涨加成交量、价跌减成交量累计，反映资金堆积</div>
              <div ref="obvChartRef" class="analysis-chart"></div>
            </div>
            <div v-if="analyzeResult?.charts?.technical?.mfi?.length" class="analysis-chart-item">
              <div class="analysis-chart-title">MFI（资金流量指数）</div>
              <div class="analysis-chart-desc">14 日典型价×成交量，&gt;80 超买 &lt;20 超卖</div>
              <div ref="mfiChartRef" class="analysis-chart"></div>
            </div>
            <div v-if="analyzeResult?.charts?.technical?.aroon_up?.length" class="analysis-chart-item">
              <div class="analysis-chart-title">阿隆指标（Aroon）</div>
              <div class="analysis-chart-desc">20 日上行/下行线，识别趋势强度与方向</div>
              <div ref="aroonChartRef" class="analysis-chart"></div>
            </div>
            <div v-if="analyzeResult?.charts?.technical?.money_flow_cumulative?.length" class="analysis-chart-item">
              <div class="analysis-chart-title">资金流向（累计净流入）</div>
              <div class="analysis-chart-desc">典型价×成交量按涨跌方向累计；大单/中单/小单为按成交额分档近似</div>
              <div ref="moneyFlowChartRef" class="analysis-chart"></div>
            </div>
          </template>
        </div>
        <el-collapse class="report-collapse">
          <el-collapse-item title="术语与曲线说明" name="glossary">
            <div class="glossary-section">
              <h4>价格图曲线</h4>
              <dl>
                <dt>MA5 / MA20 / MA60</dt>
                <dd>移动平均线：分别对最近 5、20、60 个交易日收盘价取平均并连线。短期均线反应快，长期均线代表中期趋势；价格在均线上方多为偏多，下方多为偏空；短期上穿长期称“金叉”、下穿称“死叉”。</dd>
                <dt>ARIMA 预测</dt>
                <dd>基于历史收盘价拟合的时间序列模型对未来若干交易日的点估计；虚线为预测值，半透明带为置信区间。仅供参考，不构成投资建议。</dd>
              </dl>
              <h4>功率谱密度（频域）</h4>
              <dl>
                <dt>功率谱 / 频域</dt>
                <dd>将收益率序列从“时间 vs 幅度”转换到“频率 vs 能量”：横轴为频率（1/天），纵轴为功率（对数刻度）。用于发现序列中存在的周期成分（如约 20 天、约 5 天的周期）。</dd>
                <dt>如何看</dt>
                <dd>峰值对应的频率越高，周期越短；主周期在 15～25 天常与月度效应相关，在 5～7 天可能与周内模式相关。</dd>
              </dl>
              <h4>STL 分解（趋势 / 季节 / 残差）</h4>
              <dl>
                <dt>STL</dt>
                <dd>Seasonal and Trend decomposition using Loess：把价格序列拆成“趋势 + 季节 + 残差”。趋势描述长期走向，季节描述固定周期的波动，残差为剩余不规则部分。</dd>
                <dt>如何看</dt>
                <dd>趋势线向上/向下反映中长期方向；季节分量幅度大说明周期性明显；残差若杂乱无章则说明其余可解释结构较少。</dd>
              </dl>
              <h4>自相关 ACF</h4>
              <dl>
                <dt>ACF（自相关函数）</dt>
                <dd>衡量序列与自身“滞后 k 期”的线性相关程度。横轴为滞后阶数（0 表示与自身完全相关，恒为 1），纵轴为相关系数（-1～1）。</dd>
                <dt>如何看</dt>
                <dd>若滞后 1、2 阶明显为正，说明相邻日收益有延续性；若在若干阶后快速衰减到 0 附近，则接近白噪声；若长期不衰减，可能具有趋势或长记忆。</dd>
              </dl>
              <h4>赫斯特指数 R/S</h4>
              <dl>
                <dt>赫斯特指数 H（R/S 分析）</dt>
                <dd>用“重标极差”（R/S）随滞后长度的变化估计序列的持续性。横轴为 log(滞后)，纵轴为 log(R/S)；斜率为 H。H &lt; 0.5 均值回复，H = 0.5 接近随机游走，H &gt; 0.5 趋势增强。</dd>
                <dt>如何看</dt>
                <dd>散点为不同滞后下的 R/S，红色直线为拟合；斜率 H 越大越偏趋势性，越小越偏反转性。</dd>
              </dl>
              <h4>技术指标</h4>
              <dl>
                <dt>RSI（相对强弱指数）</dt>
                <dd>根据一段时间内涨跌幅比例计算，0～100。&gt;70 常视为超买，&lt;30 视为超卖；50 附近为多空均衡。仅作参考，强势/弱势行情中可长期超买/超卖。</dd>
                <dt>MACD（指数平滑异同移动平均）</dt>
                <dd>快线（如 12 日 EMA）减慢线（如 26 日 EMA）得 MACD 线；再对 MACD 做信号线（如 9 日 EMA）。柱状图 = MACD − 信号线。柱由负转正可视为金叉、由正转负为死叉，常作趋势与动量参考。</dd>
                <dt>布林带（Bollinger Bands）</dt>
                <dd>中轨为移动平均线，上轨 = 中轨 + 2 倍标准差，下轨 = 中轨 − 2 倍标准差。价格触及上轨偏超买、下轨偏超卖；收口后常伴随波动放大。</dd>
                <dt>滚动波动率（年化）</dt>
                <dd>过去 N 日（如 20 日）收益率的标准差，再按年化（×√252）。数值越大表示近期波动越剧烈。</dd>
                <dt>VaR / CVaR（在险价值 / 条件在险价值）</dt>
                <dd>VaR(95%)：日收益在 95% 置信下的下界（负值表示亏损幅度）。CVaR 为超过 VaR 时的平均亏损。用于量化下行风险。</dd>
                <dt>OBV（能量潮）</dt>
                <dd>收盘价上涨日加上当日成交量、下跌日减去成交量、平盘不变，做累计。用于观察价量配合与资金堆积方向。</dd>
                <dt>MFI（资金流量指数）</dt>
                <dd>典型价格 = (高+低+收)/3，原始资金流 = 典型价×成交量；14 日内正负资金流比换算为 0～100。&gt;80 超买，&lt;20 超卖。</dd>
                <dt>阿隆指标（Aroon）</dt>
                <dd>20 日内最高价/最低价距今的天数换算为 0～100。Aroon 上 &gt; Aroon 下为偏多，上 &gt;70 且下 &lt;30 为强势，反之为弱势。</dd>
                <dt>资金流向（大单/中单/小单）</dt>
                <dd>按典型价×成交量与涨跌方向得每日净流入；大单/中单/小单为按成交额分档的近似（真实分档需 level-2 逐笔数据）。</dd>
              </dl>
              <h4>摘要指标简述</h4>
              <dl>
                <dt>最大回撤</dt>
                <dd>区间内“从某高点跌到后续最低点”的最大跌幅比例，衡量下行风险。</dd>
                <dt>样本熵 / 近似熵</dt>
                <dd>衡量序列的复杂度和不规则程度：值越大越随机、可预测性越低。</dd>
                <dt>ARIMA R²</dt>
                <dd>模型对历史数据的拟合优度，越接近 1 拟合越好（不代表未来预测准确）。</dd>
              </dl>
            </div>
          </el-collapse-item>
          <el-collapse-item title="查看完整报告（Markdown）" name="report">
            <div class="report-md vd" v-html="reportHtml"></div>
          </el-collapse-item>
        </el-collapse>
      </div>

      <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon closable class="error-alert" @close="errorMessage = ''" />
      <el-alert v-if="analyzeError" type="error" :title="analyzeError" show-icon closable class="error-alert" @close="analyzeError = ''" />
    </template>

    <el-empty v-else description="请从列表页点击「查看」打开股票曲线" :image-size="100" />
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { marked } from 'marked'
import * as echarts from 'echarts'
import { apiData, apiAnalyze, apiExportReport } from '@/api/stock'

marked.setOptions({ gfm: true, breaks: true })

const route = useRoute()
const router = useRouter()

const priceChartRef = ref(null)
const volumeChartRef = ref(null)
const freqChartRef = ref(null)
const stlChartRef = ref(null)
const acfChartRef = ref(null)
const hurstChartRef = ref(null)
const rsiChartRef = ref(null)
const macdChartRef = ref(null)
const bbChartRef = ref(null)
const volChartRef = ref(null)
const obvChartRef = ref(null)
const mfiChartRef = ref(null)
const aroonChartRef = ref(null)
const moneyFlowChartRef = ref(null)
const loading = ref(false)
const errorMessage = ref('')
const rangeKey = ref('1y')
const analyzeLoading = ref(false)
const analyzeError = ref('')
const analyzeResult = ref(null)
const exportLoading = ref(false)
/** 当前图表用的 K 线/成交量数据，用于分析结果出来后叠加 MA、ARIMA 预测等 */
const chartData = ref(null)

const RANGE_DAYS = {
  '5y': 5 * 365,
  '3y': 3 * 365,
  '1y': 365,
  '6m': 180,
  '3m': 90,
  '1m': 30,
}

function getDateRange() {
  const days = RANGE_DAYS[rangeKey.value] ?? 365
  const end = new Date()
  const start = new Date()
  start.setDate(start.getDate() - days)
  return {
    start: start.toISOString().slice(0, 10),
    end: end.toISOString().slice(0, 10),
  }
}

function onRangeChange() {
  loadData()
}

async function runAnalyze() {
  if (!symbol.value) return
  analyzeLoading.value = true
  analyzeError.value = ''
  analyzeResult.value = null
  try {
    const range = getDateRange()
    const data = await apiAnalyze(symbol.value, range.start, range.end)
    analyzeResult.value = data
  } catch (e) {
    analyzeError.value = e.message || '分析失败'
  } finally {
    analyzeLoading.value = false
  }
}

async function downloadReport() {
  if (!symbol.value) return
  exportLoading.value = true
  analyzeError.value = ''
  try {
    const range = getDateRange()
    const blob = await apiExportReport(symbol.value, range.start, range.end)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${symbol.value}_分析报告_${range.start}_${range.end}.md`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    analyzeError.value = e.message || '导出失败'
  } finally {
    exportLoading.value = false
  }
}

const symbol = computed(() => {
  const s = route.query.symbol
  return typeof s === 'string' ? s.trim() : ''
})

const chartTitle = computed(() => {
  return symbol.value ? `${symbol.value} - 股票数据曲线` : '股票数据曲线'
})

const reportHtml = computed(() => {
  const md = analyzeResult.value?.report_md
  if (!md) return ''
  return marked.parse(md)
})

let priceChart = null
let volumeChart = null
let freqChart = null
let stlChart = null
let acfChart = null
let hurstChart = null
let rsiChart = null
let macdChart = null
let bbChart = null
let volChart = null
let obvChart = null
let mfiChart = null
let aroonChart = null
let moneyFlowChart = null

/** 计算移动平均，与 close 等长，前 period-1 个为 null，其后为 period 日均值 */
function calcMA(close, period) {
  if (!close?.length || period < 1) return []
  const out = []
  let sum = 0
  for (let i = 0; i < close.length; i++) {
    sum += close[i]
    if (i >= period) sum -= close[i - period]
    out.push(i >= period - 1 ? sum / period : null)
  }
  return out
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

  const ma5 = calcMA(close, 5)
  const ma20 = calcMA(close, 20)
  const ma60 = calcMA(close, 60)

  let xAxisData = [...dates]
  const forecast = analyzeResult.value?.summary?.arima?.forecast
  const forecastList = Array.isArray(forecast) ? forecast : []
  let forecastDates = []
  let forecastPredicted = []
  let forecastLower = []
  let forecastUpper = []
  if (forecastList.length > 0) {
    forecastDates = forecastList.map((f) => {
      const d = f.date
      if (typeof d === 'string' && d.length >= 10) return d.slice(0, 10)
      return String(d)
    })
    forecastPredicted = forecastList.map((f) => f.predicted)
    forecastLower = forecastList.map((f) => f.lower)
    forecastUpper = forecastList.map((f) => f.upper)
    xAxisData = [...dates, ...forecastDates]
  }

  const lastClose = close.length ? close[close.length - 1] : null
  const series = [
    { name: '开盘', type: 'line', smooth: true, data: open, symbol: 'none', lineStyle: { width: 2 } },
    { name: '收盘', type: 'line', smooth: true, data: close, symbol: 'none', lineStyle: { width: 2 } },
    { name: '最高', type: 'line', smooth: true, data: high, symbol: 'none', lineStyle: { width: 2 } },
    { name: '最低', type: 'line', smooth: true, data: low, symbol: 'none', lineStyle: { width: 2 } },
    { name: 'MA5', type: 'line', smooth: true, data: ma5, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
    { name: 'MA20', type: 'line', smooth: true, data: ma20, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
    { name: 'MA60', type: 'line', smooth: true, data: ma60, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
  ]

  if (forecastList.length > 0 && lastClose != null) {
    const arimaHistorical = dates.map(() => null)
    const arimaLine = [...arimaHistorical, lastClose, ...forecastPredicted]
    series.push({
      name: 'ARIMA预测',
      type: 'line',
      smooth: false,
      data: arimaLine,
      symbol: 'circle',
      symbolSize: 4,
      lineStyle: { width: 2, type: 'dashed' },
      connectNulls: true,
    })
    if (forecastLower.length && forecastUpper.length) {
      const bandLow = [...arimaHistorical, lastClose, ...forecastLower]
      const bandHigh = [...arimaHistorical, lastClose, ...forecastUpper]
      const bandDiff = bandHigh.map((h, i) => (bandLow[i] != null ? h - bandLow[i] : null))
      series.push({
        name: '预测区间',
        type: 'line',
        data: bandLow,
        symbol: 'none',
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0.2 },
        stack: 'band',
        connectNulls: true,
        showInLegend: false,
      })
      series.push({
        name: '',
        type: 'line',
        data: bandDiff,
        symbol: 'none',
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0.2 },
        stack: 'band',
        connectNulls: true,
        showInLegend: false,
      })
    }
  }

  const legendData = series.filter((s) => s.showInLegend !== false && s.name).map((s) => s.name)
  const color = ['#00d9ff', '#00ff88', '#fbbf24', '#f87171', '#a78bfa', '#34d399', '#fcd34d', '#f472b6']

  priceChart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
    legend: { data: legendData, textStyle: { color: '#94a3b8' }, top: 0, type: 'scroll' },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '18%', containLabel: true },
    xAxis: { type: 'category', data: xAxisData, axisLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
    yAxis: { type: 'value', axisLine: { show: false }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }, axisLabel: { color: '#94a3b8' } },
    series,
    color,
  }, { notMerge: true })
}

function renderVolumeChart(data) {
  if (!volumeChartRef.value || !data) return
  if (!volumeChart) volumeChart = echarts.init(volumeChartRef.value)
  const dates = data.dates || []
  const volume = data['成交量'] || []

  volumeChart.setOption({
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '8%', containLabel: true },
    xAxis: { type: 'category', data: dates, axisLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
    yAxis: { type: 'value', name: '成交量', nameTextStyle: { color: '#94a3b8' }, axisLine: { show: false }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } }, axisLabel: { color: '#94a3b8' } },
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

const gridCommon = { left: '10%', right: '5%', bottom: '15%', top: '12%', containLabel: true }
const axisStyle = { axisLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } }

function renderAnalysisCharts() {
  const charts = analyzeResult.value?.charts
  if (!charts) return

  if (charts.frequency_domain && freqChartRef.value) {
    if (!freqChart) freqChart = echarts.init(freqChartRef.value)
    const { frequencies, psd } = charts.frequency_domain
    const data = (frequencies || []).map((f, i) => [f, (psd || [])[i]])
    freqChart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
      grid: gridCommon,
      xAxis: { type: 'value', name: '频率 (1/天)', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      yAxis: { type: 'log', name: '功率', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      series: [{ type: 'line', data, symbol: 'none', lineStyle: { width: 1.5, color: '#00d9ff' }, smooth: true }],
    })
  }

  if (charts.time_domain?.stl && stlChartRef.value) {
    if (!stlChart) stlChart = echarts.init(stlChartRef.value)
    const { dates, trend, seasonal, resid } = charts.time_domain.stl
    const opts = dates || []
    stlChart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
      grid: gridCommon,
      legend: { data: ['趋势', '季节', '残差'], textStyle: { color: '#94a3b8' }, top: 0 },
      xAxis: { type: 'category', data: opts, ...axisStyle },
      yAxis: { type: 'value', ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      series: [
        { name: '趋势', type: 'line', data: trend || [], symbol: 'none', lineStyle: { width: 1.5 }, smooth: true },
        { name: '季节', type: 'line', data: seasonal || [], symbol: 'none', lineStyle: { width: 1.5 }, smooth: true },
        { name: '残差', type: 'line', data: resid || [], symbol: 'none', lineStyle: { width: 1.5 }, smooth: true },
      ],
      color: ['#00d9ff', '#fbbf24', '#a78bfa'],
    })
  }

  if (charts.time_domain?.acf && acfChartRef.value) {
    if (!acfChart) acfChart = echarts.init(acfChartRef.value)
    const { lags, values } = charts.time_domain.acf
    const v = values || []
    acfChart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
      grid: gridCommon,
      xAxis: { type: 'category', name: '滞后', nameTextStyle: { color: '#94a3b8' }, data: (lags || []).map(String), ...axisStyle },
      yAxis: { type: 'value', name: 'ACF', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      series: [{
        type: 'bar',
        data: v,
        itemStyle: {
          color: (params) => (params.dataIndex === 0 ? '#00ff88' : '#00d9ff'),
        },
      }],
    })
  }

  if (charts.complexity?.hurst && hurstChartRef.value) {
    if (!hurstChart) hurstChart = echarts.init(hurstChartRef.value)
    const { log_lags, log_rs, hurst, coefficients } = charts.complexity.hurst
    const scatterData = (log_lags || []).map((x, i) => [x, (log_rs || [])[i]])
    let lineData = []
    if (Array.isArray(coefficients) && coefficients.length >= 2 && log_lags?.length) {
      const minX = Math.min(...log_lags)
      const maxX = Math.max(...log_lags)
      lineData = [
        [minX, coefficients[0] * minX + coefficients[1]],
        [maxX, coefficients[0] * maxX + coefficients[1]],
      ]
    }
    hurstChart.setOption({
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
      grid: gridCommon,
      title: { text: hurst != null ? `H = ${Number(hurst).toFixed(4)}` : '', left: 'center', top: 4, textStyle: { color: '#94a3b8', fontSize: 12 } },
      xAxis: { type: 'value', name: 'log(滞后)', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      yAxis: { type: 'value', name: 'log(R/S)', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
      series: [
        { type: 'scatter', data: scatterData, symbolSize: 6, itemStyle: { color: '#00d9ff' } },
        { type: 'line', data: lineData, symbol: 'none', lineStyle: { width: 2, color: '#f472b6' } },
      ],
    })
  }

  const tech = charts.technical
  if (tech && tech.dates?.length) {
    const dates = tech.dates
    if (rsiChartRef.value && tech.rsi?.length) {
      if (!rsiChart) rsiChart = echarts.init(rsiChartRef.value)
      rsiChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', min: 0, max: 100, name: 'RSI', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [{ type: 'line', data: tech.rsi, symbol: 'none', lineStyle: { width: 1.5, color: '#a78bfa' }, connectNulls: true }],
        markLine: { data: [{ yAxis: 70, lineStyle: { color: '#f87171', type: 'dashed' } }, { yAxis: 30, lineStyle: { color: '#34d399', type: 'dashed' } }], silent: true },
      })
    }
    if (macdChartRef.value && tech.macd?.length) {
      if (!macdChart) macdChart = echarts.init(macdChartRef.value)
      macdChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        legend: { data: ['MACD', '信号线', '柱'], textStyle: { color: '#94a3b8' }, top: 0 },
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [
          { name: 'MACD', type: 'line', data: tech.macd, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
          { name: '信号线', type: 'line', data: tech.macd_signal, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
          { name: '柱', type: 'bar', data: tech.macd_hist, itemStyle: { color: (p) => (p.value >= 0 ? '#00ff88' : '#f87171') }, connectNulls: true },
        ],
        color: ['#00d9ff', '#fbbf24', '#94a3b8'],
      })
    }
    if (bbChartRef.value && tech.bb_mid?.length) {
      if (!bbChart) bbChart = echarts.init(bbChartRef.value)
      bbChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        legend: { data: ['上轨', '中轨', '下轨'], textStyle: { color: '#94a3b8' }, top: 0 },
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [
          { name: '上轨', type: 'line', data: tech.bb_upper, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
          { name: '中轨', type: 'line', data: tech.bb_mid, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
          { name: '下轨', type: 'line', data: tech.bb_lower, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
        ],
        color: ['#f87171', '#00d9ff', '#34d399'],
      })
    }
    if (volChartRef.value && tech.rolling_volatility?.length) {
      if (!volChart) volChart = echarts.init(volChartRef.value)
      volChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', name: '年化波动率', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [{ type: 'line', data: tech.rolling_volatility, symbol: 'none', lineStyle: { width: 1.5, color: '#fbbf24' }, areaStyle: { opacity: 0.2 }, connectNulls: true }],
      })
    }
    if (obvChartRef.value && tech.obv?.length) {
      if (!obvChart) obvChart = echarts.init(obvChartRef.value)
      obvChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', name: 'OBV', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [{ type: 'line', data: tech.obv, symbol: 'none', lineStyle: { width: 1.5, color: '#22d3ee' }, areaStyle: { opacity: 0.15 }, connectNulls: true }],
      })
    }
    if (mfiChartRef.value && tech.mfi?.length) {
      if (!mfiChart) mfiChart = echarts.init(mfiChartRef.value)
      mfiChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', min: 0, max: 100, name: 'MFI', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [{ type: 'line', data: tech.mfi, symbol: 'none', lineStyle: { width: 1.5, color: '#a78bfa' }, connectNulls: true }],
        markLine: { data: [{ yAxis: 80, lineStyle: { color: '#f87171', type: 'dashed' } }, { yAxis: 20, lineStyle: { color: '#34d399', type: 'dashed' } }], silent: true },
      })
    }
    if (aroonChartRef.value && tech.aroon_up?.length) {
      if (!aroonChart) aroonChart = echarts.init(aroonChartRef.value)
      aroonChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        legend: { data: ['Aroon上', 'Aroon下'], textStyle: { color: '#94a3b8' }, top: 0 },
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', min: 0, max: 100, name: 'Aroon', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series: [
          { name: 'Aroon上', type: 'line', data: tech.aroon_up, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
          { name: 'Aroon下', type: 'line', data: tech.aroon_down, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: true },
        ],
        color: ['#34d399', '#f87171'],
      })
    }
    if (moneyFlowChartRef.value && tech.money_flow_cumulative?.length) {
      if (!moneyFlowChart) moneyFlowChart = echarts.init(moneyFlowChartRef.value)
      const hasTier = tech.money_flow_large?.length && tech.money_flow_mid?.length && tech.money_flow_small?.length
      const series = [
        { name: '累计净流入', type: 'line', data: tech.money_flow_cumulative, symbol: 'none', lineStyle: { width: 2, color: '#00d9ff' }, connectNulls: true },
      ]
      if (hasTier) {
        series.push(
          { name: '大单', type: 'line', data: tech.money_flow_large, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
          { name: '中单', type: 'line', data: tech.money_flow_mid, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
          { name: '小单', type: 'line', data: tech.money_flow_small, symbol: 'none', lineStyle: { width: 1 }, connectNulls: true },
        )
      }
      moneyFlowChart.setOption({
        tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155' },
        grid: gridCommon,
        legend: { data: series.map(s => s.name), textStyle: { color: '#94a3b8' }, top: 0 },
        xAxis: { type: 'category', data: dates, ...axisStyle },
        yAxis: { type: 'value', name: '资金流', nameTextStyle: { color: '#94a3b8' }, ...axisStyle, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } } },
        series,
        color: ['#00d9ff', '#f87171', '#fbbf24', '#34d399'],
      })
    }
  }
}

async function loadData() {
  if (!symbol.value) {
    priceChart?.clear()
    volumeChart?.clear()
    return
  }
  loading.value = true
  errorMessage.value = ''
  try {
    const range = getDateRange()
    const data = await apiData(symbol.value, range)
    if (data?.dates) {
      chartData.value = data
      nextTick(() => {
        renderPriceChart(data)
        renderVolumeChart(data)
        priceChart?.resize()
        volumeChart?.resize()
      })
    } else {
      errorMessage.value = '数据格式异常'
    }
  } catch (e) {
    errorMessage.value = e.message || '加载失败'
    priceChart?.clear()
    volumeChart?.clear()
  } finally {
    loading.value = false
  }
}

function onResize() {
  priceChart?.resize()
  volumeChart?.resize()
  freqChart?.resize()
  stlChart?.resize()
  acfChart?.resize()
  hurstChart?.resize()
  rsiChart?.resize()
  macdChart?.resize()
  bbChart?.resize()
  volChart?.resize()
  obvChart?.resize()
  mfiChart?.resize()
  aroonChart?.resize()
  moneyFlowChart?.resize()
}

function goBack() {
  router.push('/')
}

function disposeAllCharts() {
  priceChart?.dispose()
  volumeChart?.dispose()
  freqChart?.dispose()
  stlChart?.dispose()
  acfChart?.dispose()
  hurstChart?.dispose()
  rsiChart?.dispose()
  macdChart?.dispose()
  bbChart?.dispose()
  volChart?.dispose()
  obvChart?.dispose()
  mfiChart?.dispose()
  aroonChart?.dispose()
  moneyFlowChart?.dispose()
  priceChart = null
  volumeChart = null
  freqChart = null
  stlChart = null
  acfChart = null
  hurstChart = null
  rsiChart = null
  macdChart = null
  bbChart = null
  volChart = null
  obvChart = null
  mfiChart = null
  aroonChart = null
  moneyFlowChart = null
}

watch(symbol, () => {
  analyzeResult.value = null
  analyzeError.value = ''
  chartData.value = null
  if (!symbol.value) disposeAllCharts()
  loadData()
}, { immediate: true })

watch(analyzeResult, (val) => {
  if (!val?.charts) {
    freqChart?.dispose()
    stlChart?.dispose()
    acfChart?.dispose()
    hurstChart?.dispose()
    rsiChart?.dispose()
    macdChart?.dispose()
    bbChart?.dispose()
    volChart?.dispose()
    obvChart?.dispose()
    mfiChart?.dispose()
    aroonChart?.dispose()
    moneyFlowChart?.dispose()
    freqChart = null
    stlChart = null
    acfChart = null
    hurstChart = null
    rsiChart = null
    macdChart = null
    bbChart = null
    volChart = null
    obvChart = null
    mfiChart = null
    aroonChart = null
    moneyFlowChart = null
  }
  if (chartData.value) renderPriceChart(chartData.value)
  nextTick(() => renderAnalysisCharts())
})

onMounted(() => {
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  priceChart?.dispose()
  volumeChart?.dispose()
  freqChart?.dispose()
  stlChart?.dispose()
  acfChart?.dispose()
  hurstChart?.dispose()
  rsiChart?.dispose()
  macdChart?.dispose()
  bbChart?.dispose()
  volChart?.dispose()
  obvChart?.dispose()
  mfiChart?.dispose()
  aroonChart?.dispose()
  moneyFlowChart?.dispose()
})
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  font-weight: 600;
  margin-bottom: 16px;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.nav-row {
  margin-bottom: 16px;
}
.range-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.range-label {
  color: #94a3b8;
  font-size: 14px;
}
.analyze-btn,
.export-btn {
  margin-left: 8px;
}
.analysis-section {
  margin-top: 8px;
}
.analysis-summary {
  margin-bottom: 16px;
}
.summary-desc {
  --el-descriptions-item-bg-color: rgba(15, 23, 42, 0.5);
}
.summary-desc :deep(.el-descriptions__label) {
  color: #94a3b8;
}
.summary-desc :deep(.el-descriptions__content) {
  color: #e2e8f0;
}
.analysis-charts {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin: 16px 0;
}
@media (max-width: 768px) {
  .analysis-charts { grid-template-columns: 1fr; }
}
.analysis-chart-item {
  background: rgba(15, 23, 42, 0.5);
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.analysis-chart-title {
  font-size: 13px;
  color: #e2e8f0;
  font-weight: 600;
  margin-bottom: 4px;
}
.analysis-chart-desc {
  font-size: 12px;
  color: #64748b;
  margin-bottom: 8px;
  line-height: 1.4;
}
.analysis-chart {
  width: 100%;
  height: 240px;
}
.report-collapse {
  margin-top: 12px;
}
.report-collapse :deep(.el-collapse-item__header) {
  color: #94a3b8;
}
.glossary-section {
  padding: 4px 0;
  font-size: 13px;
  line-height: 1.65;
  color: #cbd5e1;
}
.glossary-section h4 {
  color: #e2e8f0;
  font-size: 14px;
  margin: 16px 0 8px;
  font-weight: 600;
}
.glossary-section h4:first-child {
  margin-top: 0;
}
.glossary-section dl {
  margin: 0 0 12px;
}
.glossary-section dt {
  color: #00d9ff;
  font-weight: 600;
  margin-bottom: 4px;
}
.glossary-section dd {
  margin: 0 0 0 1em;
  color: #94a3b8;
}
.report-md {
  margin: 0;
  padding: 12px 16px;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.6;
  color: #cbd5e1;
  word-break: break-word;
  max-height: 500px;
  overflow: auto;
}
.report-md.vd :deep(h1),
.report-md.vd :deep(h2),
.report-md.vd :deep(h3) {
  color: #e2e8f0;
  margin: 1em 0 0.5em;
  font-weight: 600;
}
.report-md.vd :deep(h1) { font-size: 1.25em; }
.report-md.vd :deep(h2) { font-size: 1.1em; }
.report-md.vd :deep(h3) { font-size: 1em; }
.report-md.vd :deep(p) { margin: 0.5em 0; }
.report-md.vd :deep(ul),
.report-md.vd :deep(ol) { margin: 0.5em 0; padding-left: 1.5em; }
.report-md.vd :deep(li) { margin: 0.25em 0; }
.report-md.vd :deep(code) {
  background: rgba(0, 217, 255, 0.15);
  color: #00d9ff;
  padding: 0.15em 0.4em;
  border-radius: 4px;
  font-size: 0.9em;
}
.report-md.vd :deep(pre) {
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 10px 12px;
  overflow: auto;
  margin: 0.5em 0;
}
.report-md.vd :deep(pre code) {
  background: none;
  color: #cbd5e1;
  padding: 0;
}
.report-md.vd :deep(hr) {
  border: none;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin: 1em 0;
}
.report-md.vd :deep(strong) { color: #00ff88; }
.report-md.vd :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5em 0;
}
.report-md.vd :deep(th),
.report-md.vd :deep(td) {
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 6px 10px;
  text-align: left;
}
.report-md.vd :deep(th) { background: rgba(0, 217, 255, 0.1); color: #94a3b8; }
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
.error-alert {
  margin-top: 16px;
}
</style>
