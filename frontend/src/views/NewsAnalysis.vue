<template>
  <div class="page-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1 class="page-title">
        <span class="title-icon">📊</span>
        新闻分析报告
      </h1>
      <button 
        class="btn btn-primary" 
        @click="triggerAnalysis" 
        :disabled="analyzing"
      >
        <span v-if="analyzing">⏳</span>
        <span v-else>🤖</span>
        {{ analyzing ? '分析中...' : '手动触发分析' }}
      </button>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>加载分析报告中...</p>
    </div>

    <!-- 分析报告内容 -->
    <template v-else-if="analysis">
      <!-- 市场情绪判断 -->
      <div class="sentiment-card card">
        <div class="sentiment-header">
          <span class="sentiment-icon">{{ getSentimentIcon(analysis.market_impact) }}</span>
          <h2>市场情绪判断</h2>
        </div>
        <div class="sentiment-badge" :class="analysis.market_impact">
          {{ getSentimentLabel(analysis.market_impact) }}
        </div>
      </div>

      <!-- 今日分析要点 -->
      <div class="analysis-section card">
        <h2 class="section-title">
          <span class="section-icon">💡</span>
          今日分析要点
        </h2>
        <div class="analysis-content">
          <pre class="summary-text">{{ analysis.summary }}</pre>
        </div>
      </div>

      <!-- 详细分析 -->
      <div class="analysis-section card" v-if="analysis.deep_analysis">
        <h2 class="section-title">
          <span class="section-icon">📝</span>
          详细分析
        </h2>
        <div class="analysis-content markdown-content" v-html="parseMarkdown(analysis.deep_analysis)"></div>
      </div>

      <!-- 投资建议 -->
      <div class="advice-section card">
        <h2 class="section-title">
          <span class="section-icon">💰</span>
          投资建议
        </h2>
        <div class="advice-content">
          <p>{{ analysis.investment_advice }}</p>
        </div>
      </div>

      <!-- 相关新闻 -->
      <div class="related-section card" v-if="relatedNews.length > 0">
        <h2 class="section-title">
          <span class="section-icon">📰</span>
          相关新闻
        </h2>
        <div class="related-list">
          <div 
            v-for="news in relatedNews" 
            :key="news.id" 
            class="related-item"
            @click="goToDetail(news.id)"
          >
            <span class="related-category">{{ news.category }}</span>
            <span class="related-title">{{ news.title }}</span>
          </div>
        </div>
      </div>
    </template>

    <!-- 无数据状态 -->
    <div v-else class="empty-state card">
      <div class="empty-icon">📊</div>
      <h3>暂无分析报告</h3>
      <p>点击上方按钮手动触发分析</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getLatestAnalysis, getNewsList, analyzeNews } from '@/api/news'

const router = useRouter()

// 数据状态
const loading = ref(true)
const analyzing = ref(false)
const analysis = ref(null)
const relatedNews = ref([])

// 获取分析报告
async function loadAnalysis() {
  loading.value = true
  try {
    const res = await getLatestAnalysis()
    if (res.code === 0 && res.data) {
      analysis.value = res.data
      // 加载相关新闻
      loadRelatedNews()
    }
  } catch (e) {
    console.error('加载分析报告失败:', e)
  } finally {
    loading.value = false
  }
}

// 加载相关新闻
async function loadRelatedNews() {
  try {
    const res = await getNewsList({ days: 3, limit: 5 })
    if (res.code === 0) {
      relatedNews.value = res.data?.list?.slice(0, 5) || res.data?.slice(0, 5) || []
    }
  } catch (e) {
    console.error('加载相关新闻失败:', e)
  }
}

// 手动触发分析
async function triggerAnalysis() {
  analyzing.value = true
  try {
    const res = await analyzeNews({ days: 1 })
    if (res.code === 0) {
      analysis.value = res.data
      // 提示成功
    }
  } catch (e) {
    console.error('分析失败:', e)
  } finally {
    analyzing.value = false
  }
}

// 获取情绪标签
function getSentimentLabel(impact) {
  const map = {
    bullish: '看涨',
    bearish: '看跌',
    neutral: '中性',
  }
  return map[impact] || '中性'
}

// 获取情绪图标
function getSentimentIcon(impact) {
  const map = {
    bullish: '🐂',
    bearish: '🐻',
    neutral: '➡️',
  }
  return map[impact] || '➡️'
}

// 解析 Markdown
function parseMarkdown(text) {
  if (!text) return ''
  return text
    .replace(/^## (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h5>$1</h5>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
}

// 跳转详情
function goToDetail(newsId) {
  router.push('/news/' + encodeURIComponent(newsId))
}

onMounted(() => {
  loadAnalysis()
})
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 28px;
  font-weight: 700;
  color: var(--primary);
  margin: 0;
}

.title-icon {
  font-size: 32px;
}

.loading-state {
  text-align: center;
  padding: 80px 20px;
  color: var(--text-muted);
}

.loading-spinner {
  width: 50px;
  height: 50px;
  margin: 0 auto 20px;
  border: 4px solid var(--border-color);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 情绪卡片 */
.sentiment-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin-bottom: 24px;
}

.sentiment-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.sentiment-header h2 {
  margin: 0;
  font-size: 20px;
}

.sentiment-icon {
  font-size: 36px;
}

.sentiment-badge {
  padding: 8px 24px;
  border-radius: 24px;
  font-size: 18px;
  font-weight: 700;
}

.sentiment-badge.bullish {
  background: rgba(126, 217, 87, 0.3);
  color: #7ed957;
}

.sentiment-badge.bearish {
  background: rgba(255, 107, 107, 0.3);
  color: #ff6b6b;
}

.sentiment-badge.neutral {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

/* 分析区块 */
.analysis-section {
  margin-bottom: 24px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 16px;
}

.section-icon {
  font-size: 20px;
}

.analysis-content {
  background: var(--bg-primary);
  border-radius: 12px;
  padding: 20px;
  line-height: 1.8;
}

.summary-text {
  font-size: 15px;
  color: var(--text-secondary);
  white-space: pre-wrap;
  font-family: inherit;
  margin: 0;
}

.markdown-content {
  color: var(--text-secondary);
  font-size: 14px;
}

.markdown-content :deep(h4) {
  font-size: 16px;
  color: var(--text-primary);
  margin: 16px 0 8px;
}

.markdown-content :deep(h5) {
  font-size: 14px;
  color: var(--text-primary);
  margin: 12px 0 6px;
}

.markdown-content :deep(li) {
  margin: 6px 0;
  padding-left: 8px;
}

/* 投资建议 */
.advice-section {
  margin-bottom: 24px;
  background: linear-gradient(135deg, #fff9f0 0%, #fff3e0 100%);
  border: 2px solid var(--warning);
}

.advice-content {
  background: white;
  border-radius: 12px;
  padding: 20px;
}

.advice-content p {
  font-size: 16px;
  font-weight: 600;
  color: var(--warning);
  margin: 0;
  line-height: 1.6;
}

/* 相关新闻 */
.related-section {
  margin-bottom: 24px;
}

.related-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.related-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: var(--bg-primary);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.related-item:hover {
  background: rgba(108, 155, 255, 0.1);
  transform: translateX(4px);
}

.related-category {
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  background: var(--primary);
  color: white;
  flex-shrink: 0;
}

.related-title {
  font-size: 14px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 80px 20px;
}

.empty-icon {
  font-size: 80px;
  margin-bottom: 20px;
}

.empty-state h3 {
  font-size: 20px;
  color: var(--text-primary);
  margin: 0 0 12px;
}

.empty-state p {
  color: var(--text-muted);
  margin: 0;
}
</style>
