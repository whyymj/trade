<template>
  <div class="page-container">
    <div class="page-header">
      <h1 class="page-title">基金列表</h1>
      <div class="header-actions">
        <button class="icon-btn" @click="goToPredict" title="预测中心">🎯</button>
        <button class="icon-btn" @click="goToNews" title="财经新闻">📰</button>
        <button class="icon-btn" @click="goToMarket" title="市场数据">📊</button>
        <button class="icon-btn" @click="goToFundIndustry" title="行业分析">🏭</button>
        <button class="icon-btn" @click="goToFundNews" title="新闻关联">🔗</button>
      </div>
    </div>
    
    <div class="search-row">
      <input 
        v-model="searchKeyword" 
        class="input" 
        placeholder="搜索基金代码或名称..."
        @keyup.enter="handleSearch"
      />
      <button class="btn btn-primary" @click="handleSearch">搜索</button>
      <button class="btn btn-success" @click="showAddModal = true">+ 添加基金</button>
    </div>
    
    <div class="filter-row">
      <span class="filter-label">行业:</span>
      <button 
        class="filter-tag" 
        :class="{ active: currentIndustryTag === '' }"
        @click="filterByIndustry('')"
      >
        全部
      </button>
      <button 
        v-for="tag in allIndustryTags" 
        :key="tag"
        class="filter-tag"
        :class="{ active: currentIndustryTag === tag }"
        @click="filterByIndustry(tag)"
      >
        {{ tag }}
      </button>
    </div>
    
    <div v-if="loading" class="loading">加载中...</div>
    <div v-else-if="funds.length === 0" class="empty">暂无基金数据</div>
    <div v-else class="grid grid-3">
      <div 
        v-for="fund in funds" 
        :key="fund.fund_code" 
        class="card fund-card"
        @click="goToDetail(fund.fund_code)"
      >
        <div class="fund-card-header">
          <div class="fund-name">{{ fund.fund_name }}</div>
          <button 
            class="star-btn" 
            @click.stop="toggleWatch(fund)"
          >
            {{ fund.watchlist ? '★' : '☆' }}
          </button>
        </div>
        <div class="fund-code">{{ fund.fund_code }}</div>
        
        <div class="fund-tags">
          <span class="tag tag-primary">{{ fund.fund_type || '混合型' }}</span>
          <span 
            v-for="tag in (fund.industry_tags || []).slice(0, 2)" 
            :key="tag" 
            class="tag tag-secondary"
          >
            {{ tag }}
          </span>
          <span v-if="fund.analysis_status === 'analyzing'" class="tag tag-warning">分析中...</span>
        </div>
        
        <div class="fund-price">
          <span class="nav-value">{{ fund.latest_nav?.toFixed(4) || '--' }}</span>
          <span 
            class="change-value"
            :class="(fund.daily_return || 0) >= 0 ? 'positive' : 'negative'"
          >
            {{ (fund.daily_return || 0) >= 0 ? '+' : '' }}{{ (fund.daily_return || 0).toFixed(2) }}%
          </span>
        </div>
        
        <button 
          class="btn btn-danger btn-sm"
          style="width: 100%; margin-top: 8px;"
          @click.stop="handleDelete(fund.fund_code)"
        >
          删除
        </button>
      </div>
    </div>
    
    <div v-if="totalPages > 1" class="flex-center" style="margin-top: 24px; gap: 12px;">
      <button 
        class="btn btn-outline" 
        :disabled="page <= 1"
        @click="changePage(page - 1)"
      >
        上一页
      </button>
      <span>{{ page }} / {{ totalPages }}</span>
      <button 
        class="btn btn-outline" 
        :disabled="page >= totalPages"
        @click="changePage(page + 1)"
      >
        下一页
      </button>
    </div>
    
    <div v-if="showAddModal" class="modal-overlay" @click.self="showAddModal = false">
      <div class="modal-content">
        <h3 class="modal-title">添加基金</h3>
        
        <div v-if="addProgress.length > 0" class="progress-list">
          <div 
            v-for="(item, idx) in addProgress" 
            :key="idx"
            class="progress-item"
            :class="item.status"
          >
            <span class="progress-icon">
              {{ item.status === 'pending' ? '⏳' : item.status === 'loading' ? '🔄' : item.status === 'success' ? '✅' : '❌' }}
            </span>
            <span class="progress-text">{{ item.message }}</span>
          </div>
        </div>
        
        <div v-else class="form-group">
          <label class="form-label">基金代码（多个用逗号/空格/换行分隔）</label>
          <textarea v-model="fundCodesInput" class="input" rows="4" placeholder="如: 000311, 161039, 270023"></textarea>
        </div>
        
        <div class="modal-actions">
          <button class="btn btn-outline" @click="closeAddModal" :disabled="addLoading">
            {{ addLoading ? '分析中...' : '关闭' }}
          </button>
          <button 
            v-if="addProgress.length === 0"
            class="btn btn-primary" 
            @click="handleAddBatch" 
            :disabled="addLoading"
          >
            {{ addLoading ? '添加中...' : '添加并自动分析' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getFundList, addFund, deleteFund, watchFund, getLatestNav } from '@/api/fund'

const router = useRouter()

const loading = ref(false)
const page = ref(1)
const size = ref(20)
const total = ref(0)
const searchKeyword = ref('')
const currentIndustryTag = ref('')
const showAddModal = ref(false)
const fundCodesInput = ref('')
const addLoading = ref(false)
const addProgress = ref([])
const funds = ref([])

const allIndustryTags = computed(() => {
  const tags = new Set()
  funds.value.forEach(f => {
    (f.industry_tags || []).forEach(t => tags.add(t))
  })
  return Array.from(tags).slice(0, 10)
})

onMounted(() => {
  loadFunds()
})

const totalPages = ref(1)

async function loadFunds() {
  loading.value = true
  try {
    const params = {
      page: page.value,
      size: size.value,
      keyword: searchKeyword.value || undefined,
      industry_tag: currentIndustryTag.value || undefined
    }
    const res = await getFundList(params)
    const list = res.data || []
    
    // 先显示列表，不等待净值数据
    funds.value = list
    total.value = res.total || 0
    totalPages.value = Math.ceil(total.value / size.value)
    
    // 然后异步获取净值
    for (const fund of list) {
      try {
        const navRes = await getLatestNav(fund.fund_code)
        fund.latest_nav = navRes.unit_nav
        fund.daily_return = navRes.daily_return ? navRes.daily_return * 100 : 0
      } catch (e) {
        console.warn(`获取 ${fund.fund_code} 净值失败:`, e.message)
      }
    }
  } catch (e) {
    console.error('加载基金列表失败:', e)
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  page.value = 1
  loadFunds()
}

function filterByIndustry(tag) {
  currentIndustryTag.value = tag
  page.value = 1
  loadFunds()
}

function changePage(newPage) {
  page.value = newPage
  loadFunds()
}

function goToDetail(code) {
  router.push(`/fund/${code}`)
}

function goToPredict() { router.push('/predict') }
function goToNews() { router.push('/news') }
function goToMarket() { router.push('/market') }
function goToFundIndustry() { router.push('/fund-industry') }
function goToFundNews() { router.push('/fund-news') }

async function handleAddBatch() {
  const input = fundCodesInput.value.trim()
  if (!input) {
    ElMessage.warning('请输入基金代码')
    return
  }
  
  const codes = input.split(/[,，\s\n]+/).map(c => c.trim()).filter(c => c)
  
  if (codes.length === 0) {
    ElMessage.warning('请输入有效的基金代码')
    return
  }
  
  addLoading.value = true
  addProgress.value = []
  
  for (const code of codes) {
    addProgress.value.push({ message: `添加基金 ${code}...`, status: 'loading' })
    
    try {
      await addFund({ fund_code: code })
      addProgress.value[addProgress.value.length - 1] = { message: `✅ ${code} 添加成功，正在分析行业标签...`, status: 'success' }
    } catch (e) {
      addProgress.value[addProgress.value.length - 1] = { message: `❌ ${code} 添加失败: ${e.message}`, status: 'error' }
    }
  }
  
  addLoading.value = false
  
  setTimeout(() => {
    closeAddModal()
    loadFunds()
  }, 1500)
}

function closeAddModal() {
  showAddModal.value = false
  addProgress.value = []
  fundCodesInput.value = ''
}

async function handleDelete(code) {
  try {
    await ElMessageBox.confirm('确定要删除该基金吗?', '确认删除', { type: 'warning' })
  } catch {
    return
  }
  
  try {
    await deleteFund(code)
    ElMessage.success('删除成功')
    await loadFunds()
  } catch (e) {
    ElMessage.error('删除失败: ' + e.message)
  }
}

async function toggleWatch(fund) {
  try {
    await watchFund(fund.fund_code, !fund.watchlist)
    fund.watchlist = !fund.watchlist
  } catch (e) {
    ElMessage.error('操作失败: ' + e.message)
  }
}
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.header-actions {
  display: flex;
  gap: 8px;
}

.icon-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 8px;
  background: var(--bg-secondary);
  cursor: pointer;
  font-size: 18px;
  transition: all 0.2s;
}

.icon-btn:hover {
  background: var(--primary);
  transform: scale(1.1);
}

.search-row {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.search-row .input {
  flex: 1;
  max-width: 400px;
}

.filter-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.filter-label {
  font-size: 14px;
  color: var(--text-muted);
}

.filter-tag {
  padding: 4px 12px;
  border: 1px solid var(--border-color);
  border-radius: 16px;
  background: white;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.filter-tag:hover {
  border-color: var(--primary);
}

.filter-tag.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

.fund-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.star-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #f5c842;
}

.fund-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 8px 0;
}

.fund-price {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-top: 8px;
}

.nav-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.change-value {
  font-size: 14px;
  font-weight: 500;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.progress-list {
  max-height: 300px;
  overflow-y: auto;
  margin-bottom: 16px;
}

.progress-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 4px;
  margin-bottom: 4px;
  font-size: 13px;
}

.progress-item.loading {
  background: #f0f9ff;
}

.progress-item.success {
  background: #f6ffed;
}

.progress-item.error {
  background: #fff2f0;
}

.progress-icon {
  font-size: 14px;
}
</style>
