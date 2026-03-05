<template>
  <div class="page-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1 class="page-title">
        <span class="title-icon">📰</span>
        财经新闻
      </h1>
    </div>

    <!-- 筛选区域 -->
    <div class="filter-section card">
      <!-- 分类筛选 -->
      <div class="filter-group">
        <span class="filter-label">📂 分类</span>
        <div class="filter-buttons">
          <button
            v-for="cat in categories"
            :key="cat.value"
            :class="['filter-btn', { active: currentCategory === cat.value }]"
            @click="handleCategoryChange(cat.value)"
          >
            {{ cat.label }}
          </button>
        </div>
      </div>

      <!-- 来源筛选 -->
      <div class="filter-group">
        <span class="filter-label">🌐 来源</span>
        <div class="filter-buttons">
          <button
            v-for="src in sources"
            :key="src"
            :class="['filter-btn', { active: currentSource === src }]"
            @click="handleSourceChange(src)"
          >
            {{ src }}
          </button>
        </div>
      </div>
    </div>

    <!-- 新闻列表 -->
    <div class="news-section">
      <div class="news-list">
        <div
          v-for="news in newsList"
          :key="news.id"
          class="news-card card"
          @click="goToDetail(news.id)"
        >
          <div class="news-card-header">
            <span class="news-category" :class="getCategoryClass(news.category)">
              {{ news.category || '默认' }}
            </span>
            <span class="news-source">{{ news.source }}</span>
            <span class="news-time">{{ formatTime(news.published_at) }}</span>
          </div>
          <h3 class="news-card-title">{{ news.title }}</h3>
          <p class="news-card-desc" v-if="news.content">
            {{ truncateContent(news.content) }}
          </p>
          <div class="news-card-footer">
            <span class="read-more">阅读全文 →</span>
          </div>
        </div>
        
        <div v-if="newsList.length === 0 && !loading" class="empty-state card">
          <div class="empty-icon">📭</div>
          <p>暂无新闻数据</p>
          <button class="btn btn-primary" @click="loadNews">
            <span>🔄</span> 刷新试试
          </button>
        </div>
      </div>

      <!-- 加载中 -->
      <div v-if="loading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载中...</p>
      </div>

      <!-- 分页 -->
      <div v-if="total > pageSize" class="pagination">
        <button
          class="btn btn-outline"
          :disabled="currentPage === 1"
          @click="handlePageChange(currentPage - 1)"
        >
          ← 上一页
        </button>
        <span class="page-info">
          第 {{ currentPage }} / {{ totalPage }} 页
        </span>
        <button
          class="btn btn-outline"
          :disabled="currentPage >= totalPage"
          @click="handlePageChange(currentPage + 1)"
        >
          下一页 →
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getNewsList } from '@/api/news'
import { formatTime, getCategoryClass, truncateContent } from '@/utils/formatters'

const router = useRouter()

// 分类选项
const categories = [
  { label: '全部', value: '' },
  { label: '宏观', value: '宏观' },
  { label: '行业', value: '行业' },
  { label: '全球', value: '全球' },
]

// 数据状态
const newsList = ref([])
const loading = ref(false)
const currentCategory = ref('')
const currentSource = ref('')
const sources = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

// 计算属性
const totalPage = computed(() => Math.ceil(total.value / pageSize.value))

// 加载新闻列表
async function loadNews() {
  loading.value = true
  try {
    const params = {
      category: currentCategory.value || undefined,
      source: currentSource.value || undefined,
      limit: pageSize.value,
      offset: (currentPage.value - 1) * pageSize.value,
    }
    const res = await getNewsList(params)
    if (res.code === 0) {
      newsList.value = res.data?.list || res.data || []
      total.value = res.data?.total || newsList.value.length
      
      // 提取来源列表
      if (newsList.value.length > 0 && sources.value.length === 0) {
        const uniqueSources = [...new Set(newsList.value.map(n => n.source).filter(Boolean))]
        sources.value = ['全部', ...uniqueSources]
      }
    }
  } catch (e) {
    console.error('加载新闻失败:', e)
  } finally {
    loading.value = false
  }
}

// 分类筛选
function handleCategoryChange(category) {
  currentCategory.value = category
  currentPage.value = 1
  loadNews()
}

// 来源筛选
function handleSourceChange(source) {
  currentSource.value = source === '全部' ? '' : source
  currentPage.value = 1
  loadNews()
}

// 分页
function handlePageChange(page) {
  currentPage.value = page
  loadNews()
  // 滚动到顶部
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

// 跳转详情
function goToDetail(newsId) {
  router.push('/news/' + encodeURIComponent(newsId))
}

onMounted(() => {
  loadNews()
})
</script>

<style scoped>
.page-header {
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

.filter-section {
  margin-bottom: 24px;
}

.filter-group {
  margin-bottom: 16px;
}

.filter-group:last-child {
  margin-bottom: 0;
}

.filter-label {
  display: inline-block;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.filter-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.filter-btn {
  padding: 8px 18px;
  border-radius: 20px;
  border: 2px solid var(--border-color);
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.filter-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}

.filter-btn.active {
  background: var(--primary);
  border-color: var(--primary);
  color: white;
  box-shadow: 0 4px 12px rgba(108, 155, 255, 0.3);
}

.news-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.news-card {
  cursor: pointer;
  transition: all 0.3s ease;
}

.news-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover);
}

.news-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.news-category {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

.news-category.macro {
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
  color: #1976d2;
}

.news-category.industry {
  background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
  color: #388e3c;
}

.news-category.global {
  background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
  color: #f57c00;
}

.news-category.policy {
  background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
  color: #7b1fa2;
}

.news-source {
  font-size: 13px;
  color: var(--text-muted);
}

.news-time {
  font-size: 12px;
  color: var(--text-muted);
  margin-left: auto;
}

.news-card-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 10px;
  line-height: 1.4;
}

.news-card-desc {
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.6;
  margin: 0 0 12px;
}

.news-card-footer {
  display: flex;
  justify-content: flex-end;
}

.read-more {
  font-size: 13px;
  color: var(--primary);
  font-weight: 500;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.empty-state p {
  color: var(--text-muted);
  margin-bottom: 20px;
}

.loading-state {
  text-align: center;
  padding: 40px;
  color: var(--text-muted);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  margin: 0 auto 16px;
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

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid var(--border-color);
}

.page-info {
  font-size: 14px;
  color: var(--text-secondary);
}
</style>
