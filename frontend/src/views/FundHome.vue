<template>
  <div class="page-container">
    <h1 class="page-title">基金列表</h1>
    
    <div class="search-bar">
      <input 
        v-model="searchKeyword" 
        class="input" 
        placeholder="搜索基金代码或名称..."
        @keyup.enter="handleSearch"
      />
      <button class="btn btn-primary" @click="handleSearch">搜索</button>
      <button class="btn btn-success" @click="showAddModal = true">+ 添加基金</button>
      <button class="btn" style="background: #409eff; color: white;" @click="goToPredict">🎯 预测</button>
      <button class="btn" style="background: #e6a23c; color: white;" @click="goToNews">📰 新闻</button>
      <button class="btn" style="background: #9c27b0; color: white;" @click="goToMarket">📊 市场</button>
      <button class="btn" style="background: #00bcd4; color: white;" @click="goToFundIndustry">🏭 行业</button>
      <button class="btn" style="background: #ff9800; color: white;" @click="goToFundNews">🔗 关联</button>
    </div>
    
    <div class="filter-bar">
      <button 
        class="filter-btn" 
        :class="{ active: currentType === '' }"
        @click="filterByType('')"
      >
        全部
      </button>
      <button 
        v-for="type in fundTypes" 
        :key="type"
        class="filter-btn"
        :class="{ active: currentType === type }"
        @click="filterByType(type)"
      >
        {{ type }}
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
        <div class="fund-name">{{ fund.fund_name }}</div>
        <div class="fund-code">{{ fund.fund_code }}</div>
        <div class="fund-type">{{ fund.fund_type || '混合型' }}</div>
        <div class="flex-between">
          <div>
            <span class="fund-nav">{{ fund.latest_nav?.toFixed(4) || '--' }}</span>
            <span 
              class="fund-change"
              :class="(fund.daily_return || 0) >= 0 ? 'positive' : 'negative'"
            >
              {{ (fund.daily_return || 0) >= 0 ? '+' : '' }}{{ (fund.daily_return || 0).toFixed(2) }}%
            </span>
          </div>
          <div>
            <button 
              class="btn btn-outline" 
              style="padding: 6px 12px; font-size: 12px;"
              @click.stop="toggleWatch(fund)"
            >
              {{ fund.watchlist ? '★' : '☆' }}
            </button>
            <button 
              class="btn btn-danger" 
              style="padding: 6px 12px; font-size: 12px; margin-left: 8px;"
              @click.stop="handleDelete(fund.fund_code)"
            >
              删除
            </button>
          </div>
        </div>
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
        <div class="form-group">
          <label class="form-label">基金代码（多个用逗号/空格/换行分隔）</label>
          <textarea v-model="fundCodesInput" class="input" rows="4" placeholder="如: 000311, 161039, 270023"></textarea>
        </div>
        <div class="modal-actions">
          <button class="btn btn-outline" @click="showAddModal = false">取消</button>
          <button class="btn btn-primary" @click="handleAddBatch" :disabled="addLoading">
            {{ addLoading ? '添加中...' : '添加' }}
          </button>
        </div>
        <div v-if="addResult" style="margin-top: 12px; font-size: 13px;">
          <span v-if="addResult.success > 0" style="color: green;">成功添加 {{ addResult.success }} 个基金</span>
          <span v-if="addResult.failed > 0" style="color: red; margin-left: 10px;">失败 {{ addResult.failed }} 个</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getFundList, addFund, deleteFund, watchFund, getLatestNav } from '@/api/fund'

const router = useRouter()

const loading = ref(false)
const page = ref(1)
const size = ref(20)
const total = ref(0)
const searchKeyword = ref('')
const currentType = ref('')
const showAddModal = ref(false)
const fundCodesInput = ref('')
const addLoading = ref(false)
const addResult = ref(null)
const funds = ref([])

const fundTypes = ['股票型', '混合型', '债券型', '指数型', '货币型']

watch(showAddModal, (val) => {
  if (val) {
    fundCodesInput.value = ''
    addResult.value = null
  }
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
      fund_type: currentType.value || undefined
    }
    const res = await getFundList(params)
    const list = res.data || []
    
    for (const fund of list) {
      if (searchKeyword.value) {
        const kw = searchKeyword.value.toLowerCase()
        if (!fund.fund_code.toLowerCase().includes(kw) && 
            !fund.fund_name.toLowerCase().includes(kw)) {
          continue
        }
      }
      try {
        const navRes = await getLatestNav(fund.fund_code)
        fund.latest_nav = navRes.unit_nav
        fund.daily_return = navRes.daily_return ? navRes.daily_return * 100 : 0
      } catch (e) {
        fund.latest_nav = null
        fund.daily_return = 0
      }
    }
    
    funds.value = searchKeyword.value 
      ? list.filter(f => f.fund_code.includes(searchKeyword.value) || f.fund_name.includes(searchKeyword.value))
      : list
    total.value = res.total || 0
    totalPages.value = Math.ceil(total.value / size.value)
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

function filterByType(type) {
  currentType.value = type
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

function goToPredict() {
  router.push('/predict')
}

function goToNews() {
  router.push('/news')
}

function goToMarket() {
  router.push('/market')
}

function goToFundIndustry() {
  router.push('/fund-industry')
}

function goToFundNews() {
  router.push('/fund-news')
}

async function handleAddBatch() {
  const input = fundCodesInput.value.trim()
  if (!input) {
    alert('请输入基金代码')
    return
  }
  
  // 解析基金代码：支持逗号、空格、换行分隔
  const codes = input.split(/[,，\s\n]+/).map(c => c.trim()).filter(c => c)
  
  if (codes.length === 0) {
    alert('请输入有效的基金代码')
    return
  }
  
  addLoading.value = true
  addResult.value = null
  
  let success = 0
  let failed = 0
  
  for (const code of codes) {
    try {
      await addFund({ fund_code: code })
      success++
    } catch (e) {
      failed++
      console.error(`添加 ${code} 失败:`, e)
    }
  }
  
  addResult.value = { success, failed }
  addLoading.value = false
  
  if (success > 0) {
    fundCodesInput.value = ''
    await loadFunds()
  }
}

async function handleDelete(code) {
  console.log('handleDelete called, code:', code)
  if (!confirm('确定要删除该基金吗?')) return
  
  try {
    console.log('Sending DELETE request for:', code)
    const res = await deleteFund(code)
    console.log('Response:', res)
    alert('删除成功: ' + JSON.stringify(res))
    await loadFunds()
  } catch (e) {
    console.error('Error:', e)
    alert('删除失败: ' + e.message)
  }
}

async function toggleWatch(fund) {
  try {
    await watchFund(fund.fund_code, !fund.watchlist)
    fund.watchlist = !fund.watchlist
  } catch (e) {
    alert('操作失败: ' + e.message)
  }
}

onMounted(() => {
  loadFunds()
})
</script>
