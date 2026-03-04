<template>
  <div class="macro-indicator card">
    <div class="indicator-header">
      <h3 class="indicator-title">
        <span class="title-icon">🏛️</span>
        宏观指标
      </h3>
      <span class="indicator-period">{{ currentPeriod }}</span>
    </div>

    <!-- 指标网格 -->
    <div class="indicators-grid">
      <div 
        v-for="item in indicators" 
        :key="item.key" 
        class="indicator-item"
      >
        <div class="indicator-icon">{{ item.icon }}</div>
        <div class="indicator-info">
          <span class="indicator-label">{{ item.label }}</span>
          <span class="indicator-value" :class="getValueClass(item)">
            {{ formatValue(item) }}
          </span>
          <span class="indicator-unit">{{ item.unit }}</span>
        </div>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getMarketMacroLatest } from '@/api/news'

const loading = ref(false)
const currentPeriod = ref('')
const indicators = ref([
  { 
    key: 'gdp', 
    label: 'GDP增速', 
    icon: '📊', 
    value: null, 
    unit: '%', 
    format: 'growth' 
  },
  { 
    key: 'pmi', 
    label: 'PMI', 
    icon: '🏭', 
    value: null, 
    unit: '', 
    format: 'number' 
  },
  { 
    key: 'cpi', 
    label: 'CPI', 
    icon: '🛒', 
    value: null, 
    unit: '%', 
    format: 'percent' 
  },
  { 
    key: 'm2', 
    label: 'M2增速', 
    icon: '💵', 
    value: null, 
    unit: '%', 
    format: 'growth' 
  },
  { 
    key: 'shibor', 
    label: 'Shibor', 
    icon: '🏦', 
    value: null, 
    unit: '%', 
    format: 'percent' 
  },
  { 
    key: 'usd_cny', 
    label: '美元/人民币', 
    icon: '💱', 
    value: null, 
    unit: '', 
    format: 'currency' 
  },
])

// 加载数据
async function loadData() {
  loading.value = true
  try {
    const indicatorKeys = indicators.value.map(i => i.key)
    
    for (const key of indicatorKeys) {
      try {
        const res = await getMarketMacroLatest(key)
        if (res.code === 0 && res.data) {
          const item = indicators.value.find(i => i.key === key)
          if (item) {
            item.value = res.data.value
            currentPeriod.value = res.data.period || res.data.publish_date || ''
          }
        }
      } catch (e) {
        console.warn(`加载 ${key} 失败:`, e)
      }
    }
  } catch (e) {
    console.error('加载宏观指标失败:', e)
  } finally {
    loading.value = false
  }
}

// 格式化值
function formatValue(item) {
  if (item.value === null || item.value === undefined) {
    return '--'
  }
  
  switch (item.format) {
    case 'growth':
    case 'percent':
      return item.value.toFixed(2)
    case 'number':
      return item.value.toFixed(1)
    case 'currency':
      return item.value.toFixed(4)
    default:
      return item.value
  }
}

// 获取值样式类
function getValueClass(item) {
  if (item.value === null) return ''
  
  switch (item.key) {
    case 'gdp':
    case 'pmi':
    case 'm2':
      // 越大越好
      return item.value > 0 ? 'positive' : 'negative'
    case 'cpi':
      // 温和通胀最好
      return item.value > 3 ? 'warning' : ''
    default:
      return ''
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.macro-indicator {
  height: 100%;
  position: relative;
}

.indicator-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.indicator-title {
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

.indicator-period {
  font-size: 12px;
  color: var(--text-muted);
  background: var(--bg-primary);
  padding: 4px 10px;
  border-radius: 10px;
}

.indicators-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.indicator-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px;
  background: var(--bg-primary);
  border-radius: 14px;
  transition: all 0.3s ease;
}

.indicator-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.indicator-icon {
  font-size: 24px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
  border-radius: 10px;
}

.indicator-info {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.indicator-label {
  font-size: 12px;
  color: var(--text-muted);
}

.indicator-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.3;
}

.indicator-value.positive {
  color: var(--success);
}

.indicator-value.negative {
  color: var(--danger);
}

.indicator-value.warning {
  color: var(--warning);
}

.indicator-unit {
  font-size: 11px;
  color: var(--text-muted);
}

.loading-state {
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
  .indicators-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
