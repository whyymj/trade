<template>
  <div class="container">
    <h1>数据管理</h1>
    <p class="subtitle">配置股票列表、日期范围，全量同步或单只抓取</p>

    <div class="nav-row">
      <el-link type="primary" :underline="false" router to="/">← 返回曲线</el-link>
      <el-link type="primary" :underline="false" router to="/data-range">按日期范围查询</el-link>
    </div>

    <el-card shadow="never" class="card">
      <template #header><span>当前配置</span></template>
      <el-form label-width="90px" label-position="left">
        <el-form-item label="开始日期">
          <el-input v-model="config.start_date" placeholder="YYYYMMDD 或空" clearable style="width: 200px" />
        </el-form-item>
        <el-form-item label="结束日期">
          <el-input v-model="config.end_date" placeholder="YYYYMMDD 或空" clearable style="width: 200px" />
        </el-form-item>
        <el-form-item label="复权">
          <el-select v-model="config.adjust" style="width: 160px">
            <el-option value="qfq" label="前复权" />
            <el-option value="hfq" label="后复权" />
            <el-option value="" label="不复权" />
          </el-select>
        </el-form-item>
        <el-form-item label="数据目录">
          <el-input v-model="config.output_dir" placeholder="如 data" style="width: 200px" />
        </el-form-item>
        <el-form-item>
          <el-button :loading="savingConfig" @click="saveConfig">保存配置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="card">
      <template #header><span>股票列表（config.stocks）</span></template>
      <ul class="stock-list">
        <li v-for="code in config.stocks" :key="code" class="stock-item">
          <span class="code">{{ code }}</span>
          <el-input
            :model-value="remarkByCode[code] ?? ''"
            placeholder="说明（可选）"
            maxlength="200"
            show-word-limit
            style="width: 280px; max-width: 100%"
            @update:model-value="setRemark(code, $event)"
            @blur="saveRemark(code)"
          />
          <el-button type="danger" size="small" :loading="removing === code" @click="removeStock(code)">
            移除
          </el-button>
        </li>
      </ul>
      <el-empty v-if="!config.stocks || !config.stocks.length" description="暂无股票，请添加" :image-size="60" />
      <el-form label-width="0" class="add-row">
        <el-input
          v-model="newCode"
          placeholder="如 600519 或 09678.HK"
          maxlength="10"
          style="width: 200px; margin-right: 12px"
          @input="onCodeInput"
        />
        <el-button type="primary" :loading="addingStock" :disabled="!newCode.trim()" @click="addStock">
          抓取并加入
        </el-button>
      </el-form>
    </el-card>

    <el-card shadow="never" class="card actions">
      <template #header><span>批量操作</span></template>
      <div class="btn-group">
        <el-button type="primary" :loading="syncing" @click="syncAll">
          全量同步（清空后按配置拉取）
        </el-button>
        <el-button :loading="updatingAll" @click="updateAll">
          增量更新（补全至今日）
        </el-button>
      </div>
      <p class="hint">全量同步会先清空库再按配置拉取；增量更新只拉取库内最后交易日到今天的新数据。</p>
    </el-card>

    <div v-loading="appStore.loading" class="loading-wrap" />
    <el-alert v-if="appStore.errorMessage" type="error" :title="appStore.errorMessage" show-icon closable class="error-alert" />
  </div>
</template>

<script setup>
import { ref, reactive, watch, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useAppStore } from '@/stores/app'
import {
  apiGetConfig,
  apiUpdateConfig,
  apiSyncAll,
  apiRemoveStock,
  apiAddStock,
  apiUpdateAll,
  apiUpdateStockRemark,
} from '@/api/stock'
import { useStockStore } from '@/stores/stock'

const appStore = useAppStore()
const stockStore = useStockStore()

const remarkByCode = ref({})
watch(
  () => stockStore.fileList,
  (list) => {
    const next = { ...remarkByCode.value }
    for (const f of list || []) next[f.filename] = f.remark ?? ''
    remarkByCode.value = next
  },
  { immediate: true, deep: true }
)

const config = reactive({
  start_date: '',
  end_date: '',
  adjust: 'qfq',
  output_dir: 'data',
  stocks: [],
})
const newCode = ref('')
const savingConfig = ref(false)
const syncing = ref(false)
const updatingAll = ref(false)
const addingStock = ref(false)
const removing = ref('')

function onCodeInput(val) {
  const v = (val || '').trim().toUpperCase()
  if (/\.HK$/.test(v)) {
    newCode.value = v.replace(/[^\d.]/g, '').replace(/(\d{5})\.HK.*/, '$1.HK').slice(0, 10)
  } else {
    newCode.value = (val || '').replace(/\D/g, '').slice(0, 6)
  }
}

async function loadConfig() {
  appStore.setError('')
  try {
    const data = await apiGetConfig()
    config.start_date = data.start_date ?? ''
    config.end_date = data.end_date ?? ''
    config.adjust = data.adjust ?? 'qfq'
    config.output_dir = data.output_dir ?? 'data'
    config.stocks = Array.isArray(data.stocks) ? [...data.stocks] : []
  } catch (e) {
    appStore.setError(e.message || '加载配置失败')
  }
}

async function saveConfig() {
  savingConfig.value = true
  appStore.setError('')
  try {
    const out = await apiUpdateConfig({
      start_date: config.start_date || undefined,
      end_date: config.end_date || undefined,
      adjust: config.adjust || undefined,
      output_dir: config.output_dir || undefined,
      stocks: config.stocks,
    })
    if (out.ok) {
      ElMessage.success(out.message || '已保存')
    } else {
      ElMessage.error(out.message || '保存失败')
    }
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.message || e.data?.message))
  } finally {
    savingConfig.value = false
  }
}

async function addStock() {
  const code = newCode.value.trim()
  if (!code) return
  addingStock.value = true
  appStore.setError('')
  try {
    const data = await apiAddStock(code)
    if (data?.ok) {
      ElMessage.success(data.message || '已加入')
      newCode.value = ''
      await loadConfig()
      await stockStore.fetchList()
    } else {
      ElMessage.error(data?.message || '失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
  } finally {
    addingStock.value = false
  }
}

async function removeStock(code) {
  removing.value = code
  appStore.setError('')
  try {
    const out = await apiRemoveStock(code)
    if (out.ok) {
      ElMessage.success(out.message || '已移除')
      await loadConfig()
      await stockStore.fetchList()
    } else {
      ElMessage.error(out.message || '失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
  } finally {
    removing.value = ''
  }
}

async function syncAll() {
  syncing.value = true
  appStore.setError('')
  appStore.setLoading(true)
  try {
    const data = await apiSyncAll()
    if (data?.ok && data?.results) {
      const ok = data.results.filter((r) => r.ok).length
      const fail = data.results.filter((r) => !r.ok).length
      ElMessage.success(`全量同步完成：成功 ${ok}，失败 ${fail}`)
      await loadConfig()
      await stockStore.fetchList()
    } else {
      ElMessage.error('全量同步失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
  } finally {
    appStore.setLoading(false)
    syncing.value = false
  }
}

async function updateAll() {
  updatingAll.value = true
  appStore.setError('')
  appStore.setLoading(true)
  try {
    const data = await apiUpdateAll()
    if (data?.ok && data?.results) {
      const ok = data.results.filter((r) => r.ok).length
      const fail = data.results.filter((r) => !r.ok).length
      ElMessage.success(`更新完成：成功 ${ok}，失败 ${fail}`)
      await loadConfig()
      await stockStore.fetchList()
    } else {
      ElMessage.error('更新失败')
    }
  } catch (e) {
    ElMessage.error('请求失败: ' + (e.message || e.data?.message))
  } finally {
    appStore.setLoading(false)
    updatingAll.value = false
  }
}

function setRemark(code, value) {
  remarkByCode.value = { ...remarkByCode.value, [code]: value }
}

async function saveRemark(code) {
  const remark = remarkByCode.value[code] ?? ''
  try {
    await apiUpdateStockRemark(code, remark)
    await stockStore.fetchList()
  } catch (e) {
    ElMessage.error('保存说明失败: ' + (e.message || ''))
  }
}

onMounted(async () => {
  await loadConfig()
  await stockStore.fetchList()
})
</script>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  font-weight: 600;
  margin-bottom: 8px;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.subtitle {
  color: var(--el-text-color-secondary);
  margin-bottom: 16px;
}
.nav-row {
  margin-bottom: 20px;
  display: flex;
  gap: 16px;
}
.nav-row .el-link {
  font-size: 14px;
}
.card {
  margin-bottom: 20px;
}
.add-row {
  margin-top: 16px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.stock-list {
  list-style: none;
  padding: 0;
  margin: 0 0 12px 0;
}
.stock-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.stock-item .code {
  min-width: 88px;
  color: var(--el-text-color-regular);
}
.btn-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.actions .hint {
  margin-top: 8px;
}
.hint {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.loading-wrap {
  min-height: 2px;
  margin-top: 16px;
}
.error-alert {
  margin-top: 16px;
}
</style>
