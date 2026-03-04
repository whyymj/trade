<template>
  <div class="page-container">
    <div class="page-header">
      <h1>📊 市场数据</h1>
      <button class="btn btn-primary" @click="syncData" :disabled="syncing">
        {{ syncing ? '同步中...' : '🔄 同步数据' }}
      </button>
    </div>

      <!-- 市场情绪 -->
    <div class="section sentiment-card" v-if="sentiment">
      <div class="section-header">
        <h2>📈 市场情绪</h2>
        <span class="date">📅 {{ sentiment.trade_date }}</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <span class="metric-label">💵 成交额</span>
          <span class="metric-value">{{ formatVolume(sentiment.volume) }}</span>
        </div>
        <div class="metric-item up">
          <span class="metric-label">🚀 涨停</span>
          <span class="metric-value">{{ sentiment.up_count || 0 }}</span>
        </div>
        <div class="metric-item down">
          <span class="metric-label">📉 跌停</span>
          <span class="metric-value">{{ sentiment.down_count || 0 }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">🔄 换手率</span>
          <span class="metric-value">{{ (sentiment.turnover_rate || 0).toFixed(2) }}%</span>
        </div>
        <div class="metric-item up">
          <span class="metric-label">⬆️ 上涨</span>
          <span class="metric-value">{{ sentiment.advance_count || 0 }}</span>
        </div>
        <div class="metric-item down">
          <span class="metric-label">⬇️ 下跌</span>
          <span class="metric-value">{{ sentiment.decline_count || 0 }}</span>
        </div>
      </div>
    </div>

    <!-- 无市场情绪数据 -->
    <div class="section empty-card" v-else>
      <div class="empty-state">
        <div class="icon">📊</div>
        <p>📈 暂无市场情绪数据</p>
        <button class="btn btn-primary" @click="syncData">🔄 同步数据</button>
      </div>
    </div>

    <!-- 资金流向 -->
    <div class="section money-flow-card" v-if="moneyFlow">
      <div class="section-header">
        <h2>💰 资金流向</h2>
        <span class="date">📅 {{ moneyFlow.trade_date }}</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-item" :class="getMoneyClass(moneyFlow.north_money)">
          <span class="metric-label">🌏 北向资金</span>
          <span class="metric-value">{{ formatMoney(moneyFlow.north_money) }}</span>
        </div>
        <div class="metric-item" :class="getMoneyClass(moneyFlow.north_buy)">
          <span class="metric-label">🟢 北向买入</span>
          <span class="metric-value">{{ formatMoney(moneyFlow.north_buy) }}</span>
        </div>
        <div class="metric-item" :class="getMoneyClass(moneyFlow.north_sell)">
          <span class="metric-label">🔴 北向卖出</span>
          <span class="metric-value">{{ formatMoney(moneyFlow.north_sell) }}</span>
        </div>
        <div class="metric-item" :class="getMoneyClass(moneyFlow.main_money)">
          <span class="metric-label">💎 主力资金</span>
          <span class="metric-value">{{ formatMoney(moneyFlow.main_money) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">🏦 融资余额</span>
          <span class="metric-value">{{ formatMoney(moneyFlow.margin_balance) }}</span>
        </div>
      </div>
    </div>

    <!-- 无资金流向数据 -->
    <div class="section empty-card" v-else>
      <div class="empty-state">
        <div class="icon">💰</div>
        <p>💸 暂无资金流向数据</p>
      </div>
    </div>

    <!-- 全球宏观 -->
    <div class="section global-macro-card" v-if="globalMacro && globalMacro.length > 0">
      <div class="section-header">
        <h2>🌏 全球宏观</h2>
      </div>
      <div class="metrics-grid">
        <div class="metric-item" v-for="item in globalMacro" :key="item.symbol" :class="getMoneyClass(item.change_pct)">
          <span class="metric-label">{{ getSymbolName(item.symbol) }}</span>
          <span class="metric-value">{{ formatPrice(item.close_price) }}</span>
          <span class="metric-change" :class="item.change_pct >= 0 ? 'up' : 'down'">
            {{ formatChange(item.change_pct) }}
          </span>
        </div>
      </div>
    </div>

    <!-- 宏观经济 -->
    <div class="section macro-card" v-if="macroData && macroData.length > 0">
      <div class="section-header">
        <h2>🏛️ 宏观经济</h2>
      </div>
      <div class="metrics-grid">
        <div class="metric-item" v-for="item in macroData" :key="item.indicator">
          <span class="metric-label">{{ getIndicatorName(item.indicator) }}</span>
          <span class="metric-value">{{ item.value || '-' }}</span>
          <span class="metric-unit">{{ item.unit || '' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { getMarketSentimentLatest, getMarketMoneyFlowLatest, getMarketGlobal, getMarketMacro } from '@/api/news'

export default {
  name: 'MarketHome',
  data() {
    return {
      sentiment: null,
      moneyFlow: null,
      globalMacro: [],
      macroData: [],
      syncing: false,
    }
  },
  mounted() {
    this.loadData()
  },
  methods: {
    async loadData() {
      await Promise.all([
        this.loadSentiment(),
        this.loadMoneyFlow(),
        this.loadGlobalMacro(),
        this.loadMacro(),
      ])
    },
    async loadSentiment() {
      try {
        const res = await getMarketSentimentLatest()
        if (res.code === 0 && res.data) {
          this.sentiment = res.data
        }
      } catch (e) {
        console.error(e)
      }
    },
    async loadMoneyFlow() {
      try {
        const res = await getMarketMoneyFlowLatest()
        if (res.code === 0 && res.data) {
          this.moneyFlow = res.data
        }
      } catch (e) {
        console.error(e)
      }
    },
    async loadGlobalMacro() {
      try {
        const res = await getMarketGlobal(7)
        if (res.code === 0 && res.data) {
          this.globalMacro = res.data.slice(0, 6)
        }
      } catch (e) {
        console.error(e)
      }
    },
    async loadMacro() {
      try {
        const res = await getMarketMacro(null, 12)
        if (res.code === 0 && res.data) {
          this.macroData = res.data.slice(0, 6)
        }
      } catch (e) {
        console.error(e)
      }
    },
    async syncData() {
      this.syncing = true
      try {
        const res = await fetch('/api/market/sync', { method: 'POST' })
        const data = await res.json()
        if (data.code === 0) {
          this.$message.success('同步成功')
          await this.loadData()
        }
      } catch (e) {
        console.error(e)
        this.$message.error('同步失败')
      } finally {
        this.syncing = false
      }
    },
    formatVolume(vol) {
      if (!vol) return '-'
      if (vol >= 100000000) return (vol / 100000000).toFixed(2) + '万亿'
      if (vol >= 10000) return (vol / 10000).toFixed(1) + '亿'
      return vol.toString()
    },
    formatMoney(val) {
      if (!val && val !== 0) return '-'
      const sign = val > 0 ? '+' : ''
      if (Math.abs(val) >= 100000000) return sign + (val / 100000000).toFixed(2) + '亿'
      if (Math.abs(val) >= 10000) return sign + (val / 10000).toFixed(1) + '万'
      return sign + val.toFixed(0)
    },
    formatPrice(price) {
      if (!price && price !== 0) return '-'
      return price.toFixed(2)
    },
    formatChange(val) {
      if (!val && val !== 0) return '-'
      const sign = val > 0 ? '+' : ''
      return sign + val.toFixed(2) + '%'
    },
    getMoneyClass(val) {
      if (!val) return ''
      return val > 0 ? 'up' : 'down'
    },
    getSymbolName(symbol) {
      const names = {
        'USDX': '美元指数',
        'USDCNY': '美元/人民币',
        'CL.BRENT': '布伦特原油',
        'GC.COMEX': 'COMEX黄金',
      }
      return names[symbol] || symbol
    },
    getIndicatorName(indicator) {
      const names = {
        'gdp': 'GDP',
        'cpi': 'CPI',
        'ppi': 'PPI',
        'pmi': 'PMI',
        'm2': 'M2',
        'softr': '社融',
      }
      return names[indicator] || indicator
    },
  },
}
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.page-header .btn {
  padding: 10px 20px;
  border-radius: 20px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.page-header .btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.section {
  margin-bottom: 24px;
  padding: 24px;
  border-radius: 20px;
  transition: all 0.3s ease;
}

.section:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  font-size: 20px;
  font-weight: 700;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-header h2::before {
  content: '';
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.date {
  font-size: 14px;
  opacity: 0.7;
  background: rgba(255,255,255,0.2);
  padding: 4px 12px;
  border-radius: 12px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
}

.metric-item {
  text-align: center;
  padding: 16px;
  background: rgba(255,255,255,0.15);
  border-radius: 16px;
  transition: all 0.3s ease;
  border: 2px solid transparent;
}

.metric-item:hover {
  transform: scale(1.05);
  border-color: rgba(255,255,255,0.3);
}

.metric-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  opacity: 0.9;
  margin-bottom: 8px;
}

.metric-value {
  display: block;
  font-size: 22px;
  font-weight: 700;
}

.metric-unit {
  font-size: 12px;
  opacity: 0.7;
}

.metric-change {
  display: block;
  font-size: 13px;
  font-weight: 600;
  margin-top: 6px;
  padding: 2px 8px;
  border-radius: 8px;
  background: rgba(255,255,255,0.2);
}

.metric-item.up .metric-value { 
  color: #4ade80; 
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-item.down .metric-value { 
  color: #f87171; 
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-change.up { 
  color: #4ade80; 
  background: rgba(74, 222, 128, 0.2);
}
.metric-change.down { 
  color: #f87171; 
  background: rgba(248, 113, 113, 0.2);
}

/* 市场情绪 - 紫粉渐变 */
.sentiment-card {
  background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f472b6 100%);
  color: white;
  box-shadow: 0 8px 32px rgba(168, 85, 247, 0.3);
  border: 2px solid rgba(255,255,255,0.2);
}

.sentiment-card .section-header h2::before {
  background: #fbbf24;
  animation: pulse 2s infinite;
}

/* 资金流向 - 绿松石渐变 */
.money-flow-card {
  background: linear-gradient(135deg, #14b8a6 0%, #2dd4bf 50%, #5eead4 100%);
  color: white;
  box-shadow: 0 8px 32px rgba(20, 184, 166, 0.3);
  border: 2px solid rgba(255,255,255,0.2);
}

.money-flow-card .section-header h2::before {
  background: #fbbf24;
  animation: pulse 2s infinite;
}

/* 全球宏观 - 橙粉渐变 */
.global-macro-card {
  background: linear-gradient(135deg, #f97316 0%, #fb923c 50%, #fdba74 100%);
  color: white;
  box-shadow: 0 8px 32px rgba(249, 115, 22, 0.3);
  border: 2px solid rgba(255,255,255,0.2);
}

.global-macro-card .section-header h2::before {
  background: #fff;
  animation: pulse 2s infinite;
}

/* 宏观经济 - 蓝紫渐变 */
.macro-card {
  background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
  color: white;
  box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
  border: 2px solid rgba(255,255,255,0.2);
}

.macro-card .section-header h2::before {
  background: #fbbf24;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.3); opacity: 0.7; }
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #9ca3af;
}

.empty-state .icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-state p {
  font-size: 16px;
}
</style>
