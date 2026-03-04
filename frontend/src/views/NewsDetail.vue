<template>
  <div class="page-container">
    <!-- 返回按钮 -->
    <div class="nav-bar">
      <a class="back-btn" @click="router.push('/news')">
        ← 返回新闻首页
      </a>
      <div class="nav-links">
        <a @click="router.push('/news/list')">📋 新闻列表</a>
        <a @click="router.push('/news/analysis')">📊 分析报告</a>
      </div>
    </div>
    
    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>加载新闻详情...</p>
    </div>
    
    <!-- 错误状态 -->
    <div v-else-if="error" class="error-state card">
      <div class="error-icon">😵</div>
      <p>{{ error }}</p>
      <button class="btn btn-primary" @click="loadNews">
        重试
      </button>
    </div>
    
    <!-- 新闻详情 -->
    <template v-else-if="news">
      <div class="news-detail card">
        <!-- 分类标签 -->
        <div class="news-meta">
          <span class="news-category" :class="getCategoryClass(news.category)">
            {{ news.category || '默认' }}
          </span>
          <span class="news-source">{{ news.source }}</span>
          <span class="news-time">{{ formatTime(news.published_at) }}</span>
        </div>
        
        <!-- 来源链接 -->
        <div class="source-note">
          <span>📢 来源: </span>
          <a :href="news.url" target="_blank" rel="noopener">
            {{ news.source }}
            <span class="external-icon">↗</span>
          </a>
        </div>
        
        <!-- 标题 -->
        <h1 class="news-title">{{ news.title }}</h1>
        
        <!-- 内容 -->
        <div class="news-content">
          {{ news.content }}
        </div>
        
        <!-- 操作按钮 -->
        <div class="news-actions">
          <a :href="news.url" target="_blank" class="btn btn-primary">
            查看原文 →
          </a>
        </div>
      </div>

      <!-- 相关推荐 -->
      <div class="related-section" v-if="relatedNews.length > 0">
        <h3 class="related-title">
          <span class="title-icon">📰</span>
          相关推荐
        </h3>
        <div class="related-list">
          <div 
            v-for="item in relatedNews" 
            :key="item.id" 
            class="related-card card"
            @click="goToDetail(item.id)"
          >
            <span class="related-category" :class="getCategoryClass(item.category)">
              {{ item.category || '默认' }}
            </span>
            <h4 class="related-item-title">{{ item.title }}</h4>
            <div class="related-meta">
              <span>{{ item.source }}</span>
              <span>{{ formatTime(item.published_at) }}</span>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { getNewsDetail, getNewsList } from '@/api/news'

const router = useRouter()
const route = useRoute()

const loading = ref(true)
const error = ref(null)
const news = ref(null)
const relatedNews = ref([])

// 加载新闻详情
async function loadNews() {
  const newsId = route.params.id
  
  if (!newsId) {
    error.value = '新闻ID不存在'
    loading.value = false
    return
  }
  
  loading.value = true
  error.value = null
  
  try {
    const res = await getNewsDetail(newsId)
    if (res.code === 0) {
      news.value = res.data
      // 加载相关新闻
      loadRelatedNews()
    } else {
      error.value = res.message || '加载失败'
    }
  } catch (e) {
    error.value = e.message || '加载失败'
  } finally {
    loading.value = false
  }
}

// 加载相关新闻
async function loadRelatedNews() {
  try {
    // 获取同类别的新闻
    const category = news.value?.category
    const res = await getNewsList({ 
      category: category || undefined,
      limit: 4 
    })
    
    if (res.code === 0) {
      let list = res.data?.list || res.data || []
      // 过滤掉当前新闻
      relatedNews.value = list
        .filter(item => item.id !== news.value?.id)
        .slice(0, 4)
    }
  } catch (e) {
    console.error('加载相关新闻失败:', e)
  }
}

// 跳转详情
function goToDetail(newsId) {
  router.push('/news/' + encodeURIComponent(newsId))
}

// news/' + encode格式化时间
function formatTime(timeStr) {
  if (!timeStr) return ''
  const date = new Date(timeStr)
  return date.toLocaleDateString('zh-CN', { 
    month: 'short', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 获取分类样式
function getCategoryClass(category) {
  const map = {
    '宏观': 'macro',
    '政策': 'policy',
    '行业': 'industry',
    '全球': 'global',
  }
  return map[category] || ''
}

onMounted(() => {
  loadNews()
})
</script>

<style scoped>
.page-container {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
}

.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--primary);
  text-decoration: none;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.back-btn:hover {
  text-decoration: underline;
  transform: translateX(-4px);
}

.nav-links {
  display: flex;
  gap: 16px;
}

.nav-links a {
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: color 0.3s ease;
}

.nav-links a:hover {
  color: var(--primary);
}

/* 加载状态 */
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

/* 错误状态 */
.error-state {
  text-align: center;
  padding: 60px 20px;
}

.error-icon {
  font-size: 64px;
  margin-bottom: 20px;
}

.error-state p {
  color: var(--text-muted);
  margin-bottom: 20px;
}

/* 新闻详情 */
.news-detail {
  padding: 28px;
  margin-bottom: 24px;
}

.news-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.news-category {
  padding: 6px 14px;
  border-radius: 14px;
  font-size: 12px;
  font-weight: 600;
  background: #f0f0f0;
  color: #666;
}

.news-category.macro { background: #e3f2fd; color: #1976d2; }
.news-category.policy { background: #f3e5f5; color: #7b1fa2; }
.news-category.industry { background: #e8f5e9; color: #388e3c; }
.news-category.global { background: #fff3e0; color: #f57c00; }

.news-source {
  color: var(--text-secondary);
  font-size: 14px;
}

.news-time {
  color: var(--text-muted);
  font-size: 13px;
}

.source-note {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 20px;
  padding: 14px;
  background: var(--bg-primary);
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.source-note a {
  color: var(--primary);
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.source-note a:hover {
  text-decoration: underline;
}

.external-icon {
  font-size: 12px;
}

.news-title {
  font-size: 26px;
  font-weight: 700;
  margin-bottom: 24px;
  line-height: 1.4;
  color: var(--text-primary);
}

.news-content {
  font-size: 16px;
  line-height: 1.9;
  color: var(--text-primary);
  white-space: pre-wrap;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border-color);
}

.news-actions {
  margin-top: 24px;
  padding-top: 24px;
}

/* 相关推荐 */
.related-section {
  margin-top: 24px;
}

.related-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.title-icon {
  font-size: 20px;
}

.related-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.related-card {
  padding: 18px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.related-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover);
}

.related-category {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 10px;
}

.related-item-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 10px;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.related-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-muted);
}

@media (max-width: 768px) {
  .nav-bar {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }
  
  .related-list {
    grid-template-columns: 1fr;
  }
  
  .news-title {
    font-size: 22px;
  }
}
</style>
