<template>
  <div class="page-container">
    <div class="page-header">
      <h1 class="page-title">🔗 基金新闻关联</h1>
      <p class="page-desc">查看基金与相关行业新闻的关联分析</p>
    </div>

    <div class="search-bar card">
      <input 
        v-model="fundCode" 
        type="text" 
        class="search-input"
        placeholder="输入基金代码，如: 000001"
        @keyup.enter="searchFund"
      />
      <button class="btn btn-primary" @click="searchFund" :disabled="loading">
        {{ loading ? '搜索中...' : '🔍 查看关联' }}
      </button>
    </div>

    <div v-if="loading && !summary" class="loading-card card">
      <div class="loading-icon">🔗</div>
      <p>正在分析基金与新闻的关联...</p>
    </div>

    <template v-if="summary">
      <div class="summary-card card">
        <div class="summary-header">
          <div>
            <h2 class="fund-name">{{ summary.fund_name || summary.fund_code }}</h2>
            <span class="fund-code">{{ summary.fund_code }}</span>
          </div>
          <div class="sentiment-badge" :class="summary.sentiment">
            {{ getSentimentText(summary.sentiment) }}
          </div>
        </div>

        <div class="summary-stats">
          <div class="stat-item">
            <span class="stat-value">{{ summary.news_count }}</span>
            <span class="stat-label">相关新闻</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ summary.industries?.length || 0 }}</span>
            <span class="stat-label">配置行业</span>
          </div>
        </div>

        <div class="industries-section">
          <h3 class="section-title">📊 行业配置</h3>
          <div class="industry-tags">
            <span 
              v-for="ind in summary.industries" 
              :key="ind.industry"
              class="industry-tag"
            >
              {{ ind.industry }} ({{ ind.confidence }}%)
            </span>
          </div>
        </div>
      </div>

      <div class="news-section">
        <h3 class="section-title">📰 相关新闻</h3>
        <div v-if="summary.latest_news && summary.latest_news.length > 0" class="news-list">
          <div 
            v-for="(news, index) in summary.latest_news" 
            :key="index"
            class="news-item card"
          >
            <div class="news-header">
              <span class="tag tag-primary">{{ news.industry }}</span>
              <span class="match-score" :class="getScoreClass(news.match_score)">
                匹配度 {{ (news.match_score * 100).toFixed(0) }}%
              </span>
            </div>
            <a :href="news.url" target="_blank" class="news-title">{{ news.title }}</a>
            <div class="news-meta">
              <span>{{ news.source }}</span>
            </div>
          </div>
        </div>
        <div v-else class="empty-state card">
          <span class="empty-icon">📭</span>
          <p>暂无相关新闻</p>
        </div>
      </div>
    </template>

    <div v-if="error" class="error-card card">
      <span class="error-icon">⚠️</span>
      <p>{{ error }}</p>
    </div>

    <div class="tips-card card">
      <div class="tips-title">💡 使用说明</div>
      <ul class="tips-list">
        <li>输入基金代码查看该基金的相关行业新闻</li>
        <li>正相关新闻越多，说明基金持仓行业近期热度较高</li>
        <li>情绪分析基于新闻数量：正面新闻多则情绪积极</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const fundCode = ref(route.query.code || '')
const summary = ref(null)
const loading = ref(false)
const error = ref('')

const searchFund = async () => {
  if (!fundCode.value.trim()) {
    error.value = '请输入基金代码'
    return
  }
  
  loading.value = true
  error.value = ''
  summary.value = null
  
  try {
    const res = await fetch(`/api/fund-news/summary/${fundCode.value}?days=7`)
    const data = await res.json()
    
    if (data.code === 0) {
      summary.value = data.data
    } else {
      error.value = data.message || '获取失败'
    }
  } catch (e) {
    error.value = '网络错误，请稍后重试'
  } finally {
    loading.value = false
  }
}

const getSentimentText = (sentiment) => {
  const map = {
    positive: '😊 积极',
    negative: '😟 消极',
    neutral: '😐 中性'
  }
  return map[sentiment] || '😐 中性'
}

const getScoreClass = (score) => {
  if (score >= 0.8) return 'high'
  if (score >= 0.5) return 'medium'
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

.summary-card {
  margin-bottom: 24px;
}

.summary-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
}

.fund-name {
  font-size: 20px;
  margin-bottom: 4px;
}

.fund-code {
  font-size: 14px;
  color: var(--text-muted);
}

.sentiment-badge {
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
}

.sentiment-badge.positive {
  background: #f6ffed;
  color: #52c41a;
}

.sentiment-badge.negative {
  background: #fff2f0;
  color: #ff4d4f;
}

.sentiment-badge.neutral {
  background: #fafafa;
  color: #666;
}

.summary-stats {
  display: flex;
  gap: 32px;
  padding: 16px;
  background: var(--bg-primary);
  border-radius: 12px;
  margin-bottom: 20px;
}

.stat-item {
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 28px;
  font-weight: 700;
  color: var(--primary);
}

.stat-label {
  font-size: 12px;
  color: var(--text-muted);
}

.industries-section {
  margin-top: 20px;
}

.section-title {
  font-size: 16px;
  margin-bottom: 12px;
  color: var(--text-primary);
}

.industry-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.industry-tag {
  padding: 6px 12px;
  background: var(--bg-primary);
  border-radius: 16px;
  font-size: 13px;
  color: var(--text-secondary);
}

.news-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.news-item {
  transition: transform 0.3s;
}

.news-item:hover {
  transform: translateX(4px);
}

.news-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.match-score {
  font-weight: 600;
  font-size: 13px;
}

.match-score.high {
  color: #52c41a;
}

.match-score.medium {
  color: #faad14;
}

.match-score.low {
  color: #999;
}

.news-title {
  display: block;
  font-size: 15px;
  font-weight: 500;
  color: var(--text-primary);
  text-decoration: none;
  margin-bottom: 8px;
  line-height: 1.5;
}

.news-title:hover {
  color: var(--primary);
}

.news-meta {
  font-size: 12px;
  color: var(--text-muted);
}

.tips-card {
  background: #f0f9ff;
  border: 1px solid #91d5ff;
  margin-top: 24px;
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

.empty-state {
  text-align: center;
  padding: 48px;
}

.empty-icon {
  font-size: 48px;
}

.empty-state p {
  color: var(--text-secondary);
  margin-top: 12px;
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
