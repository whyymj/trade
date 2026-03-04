<template>
  <div class="page-container">
    <div class="page-header">
      <h1 class="page-title">🏭 基金行业分析</h1>
      <p class="page-desc">基于 AI 分析基金的行业配置</p>
    </div>

    <div class="search-bar card">
      <input 
        v-model="fundCode" 
        type="text" 
        class="search-input"
        placeholder="输入基金代码，如: 000001"
        @keyup.enter="analyzeFund"
      />
      <button class="btn btn-primary" @click="analyzeFund" :disabled="loading">
        {{ loading ? '分析中...' : '🔍 分析' }}
      </button>
    </div>

    <div v-if="loading && !result" class="loading-card card">
      <div class="loading-icon">🧠</div>
      <p>AI 正在分析基金行业配置...</p>
    </div>

    <template v-if="result && result.length > 0">
      <div class="result-card card">
        <div class="result-header">
          <span class="tag tag-success">✅ 分析完成</span>
          <span class="fund-name">{{ fundName }}</span>
        </div>
        
        <div class="industry-list">
          <div 
            v-for="(item, index) in result" 
            :key="index"
            class="industry-item"
          >
            <div class="industry-info">
              <span class="industry-name">{{ item.industry }}</span>
              <span class="industry-code">{{ item.industry_code }}</span>
            </div>
            <div class="confidence-bar">
              <div 
                class="confidence-fill"
                :style="{ width: item.confidence + '%' }"
                :class="getConfidenceClass(item.confidence)"
              ></div>
            </div>
            <span 
              class="confidence-text"
              :class="getConfidenceClass(item.confidence)"
            >
              {{ item.confidence.toFixed(1) }}%
            </span>
          </div>
        </div>
      </div>

      <div class="tips-card card">
        <div class="tips-title">💡 说明</div>
        <ul class="tips-list">
          <li>行业置信度 ≥ 80%：高置信度行业，基金主要配置</li>
          <li>行业置信度 50%-80%：中置信度行业，可能有部分配置</li>
          <li>行业置信度 &lt; 50%：低置信度行业，仅供参考</li>
        </ul>
      </div>
    </template>

    <div v-if="error" class="error-card card">
      <span class="error-icon">⚠️</span>
      <p>{{ error }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const fundCode = ref(route.query.code || '')
const fundName = ref('')
const result = ref(null)
const loading = ref(false)
const error = ref('')

const analyzeFund = async () => {
  if (!fundCode.value.trim()) {
    error.value = '请输入基金代码'
    return
  }
  
  loading.value = true
  error.value = ''
  result.value = null
  
  try {
    const res = await fetch(`/api/fund-industry/analyze/${fundCode.value}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    const data = await res.json()
    
    if (data.code === 0) {
      result.value = data.data
      if (result.value.length > 0) {
        fundName.value = `基金代码: ${fundCode.value}`
      }
    } else {
      error.value = data.message || '分析失败'
    }
  } catch (e) {
    error.value = '网络错误，请稍后重试'
  } finally {
    loading.value = false
  }
}

const getConfidenceClass = (confidence) => {
  if (confidence >= 80) return 'high'
  if (confidence >= 50) return 'medium'
  return 'low'
}
</script>

<style scoped>
.page-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-title {
  font-size: 28px;
  color: var(--primary);
  margin-bottom: 8px;
}

.page-desc {
  color: var(--text-secondary);
  font-size: 14px;
}

.search-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
}

.search-input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid var(--border-color);
  border-radius: 12px;
  font-size: 16px;
  transition: border-color 0.3s;
}

.search-input:focus {
  outline: none;
  border-color: var(--primary);
}

.loading-card {
  text-align: center;
  padding: 48px;
}

.loading-icon {
  font-size: 48px;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.result-card {
  margin-bottom: 24px;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.fund-name {
  font-size: 16px;
  color: var(--text-secondary);
}

.industry-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.industry-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  background: var(--bg-primary);
  border-radius: 12px;
}

.industry-info {
  min-width: 120px;
}

.industry-name {
  font-weight: 600;
  color: var(--text-primary);
  display: block;
}

.industry-code {
  font-size: 12px;
  color: var(--text-muted);
}

.confidence-bar {
  flex: 1;
  height: 12px;
  background: #eee;
  border-radius: 6px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.5s ease;
}

.confidence-fill.high {
  background: linear-gradient(90deg, #7ED957, #52C41A);
}

.confidence-fill.medium {
  background: linear-gradient(90deg, #FFB347, #FAAD14);
}

.confidence-fill.low {
  background: linear-gradient(90deg, #FF6B6B, #FF4D4F);
}

.confidence-text {
  min-width: 60px;
  text-align: right;
  font-weight: 600;
}

.confidence-text.high {
  color: #52C41A;
}

.confidence-text.medium {
  color: #FAAD14;
}

.confidence-text.low {
  color: #FF4D4F;
}

.tips-card {
  background: #f0f9ff;
  border: 1px solid #91d5ff;
}

.tips-title {
  font-weight: 600;
  color: #1890ff;
  margin-bottom: 12px;
}

.tips-list {
  margin: 0;
  padding-left: 20px;
  color: var(--text-secondary);
  font-size: 14px;
}

.tips-list li {
  margin-bottom: 6px;
}

.error-card {
  background: #fff2f0;
  border: 1px solid #ffccc7;
  text-align: center;
}

.error-icon {
  font-size: 32px;
}

.error-card p {
  color: #ff4d4f;
  margin: 12px 0 0;
}
</style>
