<template>
  <div class="page-container">
    <div class="page-header">
      <h1 class="page-title">📰 新闻行业分类</h1>
      <p class="page-desc">AI 自动将财经新闻分类到对应行业</p>
    </div>

    <div class="action-bar card">
      <button class="btn btn-primary" @click="classifyToday" :disabled="loading">
        {{ loading ? '分类中...' : '🚀 分类今日新闻' }}
      </button>
      <span class="stats-text" v-if="stats">今日已分类 {{ stats.classified_count }} 条</span>
    </div>

    <div class="tabs">
      <button 
        class="tab" 
        :class="{ active: activeTab === 'stats' }"
        @click="activeTab = 'stats'; loadStats()"
      >
        📊 行业统计
      </button>
      <button 
        class="tab" 
        :class="{ active: activeTab === 'today' }"
        @click="activeTab = 'today'; loadToday()"
      >
        📰 今日新闻
      </button>
    </div>

    <div v-if="activeTab === 'stats'" class="stats-section">
      <div v-if="loadingStats" class="loading">加载中...</div>
      <template v-else-if="industryStats && industryStats.length > 0">
        <div class="stats-grid">
          <div 
            v-for="item in industryStats" 
            :key="item.industry_code"
            class="stat-card card"
            @click="viewIndustry(item.industry_code)"
          >
            <div class="stat-header">
              <span class="industry-name">{{ item.industry }}</span>
              <span class="tag tag-primary">{{ item.industry_code }}</span>
            </div>
            <div class="stat-count">{{ item.count }}</div>
            <div class="stat-confidence">平均置信度: {{ (item.avg_confidence * 100).toFixed(1) }}%</div>
          </div>
        </div>
      </template>
      <div v-else class="empty-state card">
        <span class="empty-icon">📭</span>
        <p>暂无分类数据，请先点击"分类今日新闻"</p>
      </div>
    </div>

    <div v-if="activeTab === 'today'" class="news-section">
      <div v-if="loadingToday" class="loading">加载中...</div>
      <template v-else-if="todayNews && todayNews.length > 0">
        <div class="news-list">
          <div 
            v-for="item in todayNews" 
            :key="item.news_id"
            class="news-item card"
          >
            <div class="news-header">
              <span class="tag" :class="getConfidenceClass(item.confidence)">
                {{ item.industry }}
              </span>
              <span class="news-score">{{ (item.confidence * 100).toFixed(0) }}%</span>
            </div>
            <a :href="item.url" target="_blank" class="news-title">{{ item.title }}</a>
            <div class="news-meta">
              <span>{{ item.source }}</span>
            </div>
          </div>
        </div>
      </template>
      <div v-else class="empty-state card">
        <span class="empty-icon">📭</span>
        <p>今日暂无分类新闻</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const activeTab = ref('stats')
const loading = ref(false)
const loadingStats = ref(false)
const loadingToday = ref(false)
const stats = ref(null)
const industryStats = ref([])
const todayNews = ref([])

onMounted(() => {
  loadStats()
})

const loadStats = async () => {
  loadingStats.value = true
  try {
    const res = await fetch('/api/news-classification/stats?days=7')
    const data = await res.json()
    if (data.code === 0) {
      industryStats.value = data.data
    }
  } catch (e) {
    console.error(e)
  } finally {
    loadingStats.value = false
  }
}

const loadToday = async () => {
  loadingToday.value = true
  try {
    const res = await fetch('/api/news-classification/today')
    const data = await res.json()
    if (data.code === 0) {
      todayNews.value = data.data
    }
  } catch (e) {
    console.error(e)
  } finally {
    loadingToday.value = false
  }
}

const classifyToday = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/news-classification/classify-today', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    const data = await res.json()
    if (data.code === 0) {
      stats.value = data.data
      loadStats()
      loadToday()
    } else {
      alert(data.message || '分类失败')
    }
  } catch (e) {
    alert('网络错误')
  } finally {
    loading.value = false
  }
}

const viewIndustry = (code) => {
  router.push(`/news/industry/${code}`)
}

const getConfidenceClass = (confidence) => {
  if (confidence >= 0.8) return 'tag-success'
  if (confidence >= 0.5) return 'tag-warning'
  return 'tag-secondary'
}
</script>

<style scoped>
.page-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 24px;
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

.action-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.stats-text {
  color: var(--text-secondary);
  font-size: 14px;
}

.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
}

.tab {
  padding: 12px 24px;
  border: none;
  background: var(--bg-secondary);
  border-radius: 12px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.tab.active {
  background: var(--primary);
  color: white;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}

.stat-card {
  cursor: pointer;
  transition: transform 0.3s;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.stat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.industry-name {
  font-weight: 600;
}

.stat-count {
  font-size: 32px;
  font-weight: 700;
  color: var(--primary);
}

.stat-confidence {
  font-size: 12px;
  color: var(--text-muted);
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

.news-score {
  font-weight: 600;
  color: var(--primary);
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
</style>
