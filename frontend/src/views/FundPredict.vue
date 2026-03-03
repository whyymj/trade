<template>
  <div class="page-container">
    <h1 class="page-title">预测中心</h1>
    
    <div class="grid grid-2" style="gap: 24px;">
      <div class="card">
        <h3 style="margin-bottom: 20px;">发起预测</h3>
        <div class="form-group">
          <label class="form-label">选择基金</label>
          <select v-model="selectedFund" class="input" @change="loadPrediction">
            <option value="">请选择基金</option>
            <option v-for="f in funds" :key="f.fund_code" :value="f.fund_code">
              {{ f.fund_name }} ({{ f.fund_code }})
            </option>
          </select>
        </div>
        
        <button 
          class="btn btn-primary" 
          style="width: 100%;"
          :disabled="!selectedFund || predicting"
          @click="handlePredict"
        >
          {{ predicting ? '预测中...' : '开始预测' }}
        </button>
        
        <div v-if="prediction" class="prediction-card" style="margin-top: 30px;">
          <div class="prediction-direction" :class="prediction.direction === 1 ? 'up' : 'down'">
            {{ prediction.direction === 1 ? '📈 上涨' : '📉 下跌' }}
          </div>
          <div class="prediction-prob">
            看涨概率: {{ ((prediction.prob_up || 0) * 100).toFixed(1) }}%
          </div>
          <div class="prediction-magnitude">
            预测涨幅: {{ ((prediction.magnitude || 0) * 100).toFixed(2) }}%
          </div>
          <div v-if="prediction.magnitude_5" style="margin-top: 20px; text-align: left;">
            <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">未来5日预测:</div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
              <span 
                v-for="(mag, idx) in prediction.magnitude_5" 
                :key="idx"
                class="tag"
                :class="mag >= 0 ? 'tag-success' : 'tag-danger'"
              >
                第{{ idx + 1 }}天: {{ (mag * 100).toFixed(2) }}%
              </span>
            </div>
          </div>
          <div style="margin-top: 20px; color: var(--text-muted); font-size: 12px;">
            预测时间: {{ prediction.predict_date }}
          </div>
        </div>
        
        <div ref="chartRef" class="chart-container" style="margin-top: 30px;"></div>
      </div>
      
      <div class="card">
        <h3 style="margin-bottom: 20px;">我的关注基金</h3>
        <div v-if="watchlistLoading" class="loading">加载中...</div>
        <div v-else-if="watchlist.length === 0" class="empty">暂无关注的基金</div>
        <div v-else class="history-list">
          <div 
            v-for="fund in watchlist" 
            :key="fund.fund_code"
            class="history-item"
          >
            <div>
              <div style="font-weight: 500;">{{ fund.fund_name }}</div>
              <div style="font-size: 12px; color: var(--text-muted);">{{ fund.fund_code }}</div>
            </div>
            <button 
              class="btn btn-outline" 
              style="padding: 6px 16px;"
              @click="selectFundAndPredict(fund.fund_code)"
            >
              预测
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <div class="card" style="margin-top: 24px;">
      <h3 style="margin-bottom: 20px;">所有基金预测</h3>
      <div v-if="fundsLoading" class="loading">加载中...</div>
      <div v-else-if="funds.length === 0" class="empty">暂无基金数据</div>
      <div v-else class="grid grid-4">
        <div 
          v-for="fund in funds" 
          :key="fund.fund_code" 
          class="card"
          style="padding: 16px;"
        >
          <div style="font-weight: 500; margin-bottom: 4px;">{{ fund.fund_name }}</div>
          <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">{{ fund.fund_code }}</div>
          <button 
            class="btn btn-primary" 
            style="width: 100%; padding: 8px;"
            :disabled="predicting"
            @click="selectFundAndPredict(fund.fund_code)"
          >
            预测
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getFundList, getFundWatchlist, predict, getPrediction, getFitPlot } from '@/api/fund'

const funds = ref([])
const watchlist = ref([])
const selectedFund = ref('')
const predicting = ref(false)
const prediction = ref(null)
const fundsLoading = ref(true)
const watchlistLoading = ref(true)

let chartInstance = null
const chartRef = ref(null)

async function loadFunds() {
  fundsLoading.value = true
  try {
    const res = await getFundList({ page: 1, size: 100 })
    funds.value = res.data || []
  } catch (e) {
    console.error('加载基金列表失败:', e)
  } finally {
    fundsLoading.value = false
  }
}

async function loadWatchlist() {
  watchlistLoading.value = true
  try {
    const res = await getFundWatchlist({ page: 1, size: 20 })
    watchlist.value = res.data || []
  } catch (e) {
    console.error('加载关注列表失败:', e)
  } finally {
    watchlistLoading.value = false
  }
}

async function loadPrediction() {
  if (!selectedFund.value) {
    prediction.value = null
    return
  }
  try {
    prediction.value = await getPrediction(selectedFund.value)
  } catch (e) {
    console.error('加载预测结果失败:', e)
    prediction.value = null
  }
}

async function handlePredict() {
  if (!selectedFund.value || predicting.value) return
  
  predicting.value = true
  prediction.value = null
  
  try {
    prediction.value = await predict(selectedFund.value)
  } catch (e) {
    alert('预测失败: ' + e.message)
  } finally {
    predicting.value = false
  }
}

function selectFundAndPredict(code) {
  selectedFund.value = code
  loadPrediction()
  loadFitPlot()
}

async function loadFitPlot() {
  if (!selectedFund.value) return
  
  try {
    const data = await getFitPlot(selectedFund.value)
    if (data && data.dates && chartInstance) {
      const dates = data.dates
      const actual = data.actual
      const fitted = data.fitted || []
      const fittedStart = data.fitted_start_index || 0
      
      const fittedPadded = [...Array(fittedStart).fill(null), ...fitted]
      
      chartInstance.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['实际净值', '模型拟合'] },
        xAxis: { type: 'category', data: dates },
        yAxis: { type: 'value', name: '净值' },
        series: [
          { name: '实际净值', type: 'line', data: actual, smooth: true },
          { name: '模型拟合', type: 'line', data: fittedPadded, smooth: true, lineStyle: { color: '#ff6b6b', width: 2 } }
        ]
      })
    }
  } catch (e) {
    console.error('加载拟合曲线失败:', e)
  }
}

function initChart() {
  if (chartRef.value && !chartInstance) {
    chartInstance = echarts.init(chartRef.value)
  }
}

onMounted(() => {
  loadFunds()
  loadWatchlist()
  nextTick(() => {
    initChart()
  })
})
</script>
