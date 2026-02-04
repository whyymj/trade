<template>
  <div class="container">
    <div class="nav-row">
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/chart')">股票曲线</el-link>
    </div>
    <h1>股票列表</h1>
    <p class="subtitle">管理已配置股票，点击「查看」在新标签页打开曲线</p>

    <div class="toolbar">
      <el-button type="primary" @click="openAddDialog">新增股票</el-button>
      <el-button :loading="updatingAll" @click="openUpdatePeriodDialog">
        一次性全部更新
      </el-button>
    </div>

    <el-card shadow="never" class="list-card">
      <el-table :data="stockStore.fileList" stripe style="width: 100%">
        <el-table-column prop="filename" label="股票代码" min-width="120" />
        <el-table-column prop="displayName" label="股票名称" min-width="140" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.displayName || '—' }}
          </template>
        </el-table-column>
        <el-table-column prop="lastUpdateDate" label="最后更新日期" width="130">
          <template #default="{ row }">
            {{ row.lastUpdateDate || '—' }}
          </template>
        </el-table-column>
        <el-table-column prop="remark" label="说明" min-width="200" show-overflow-tooltip />
        <el-table-column label="操作" width="160" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="openChart(row)">查看</el-button>
            <el-button type="primary" link @click="openEditDialog(row)">修改</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!stockStore.fileList.length && !appStore.loading" description="暂无股票，请点击「新增股票」添加" :image-size="80" />
    </el-card>

    <el-dialog
      v-model="addDialogVisible"
      title="新增股票"
      width="400px"
      :close-on-click-modal="false"
      @closed="resetAddForm"
    >
      <el-form :model="addForm" label-width="80px">
        <el-form-item label="股票代码" required>
          <el-input
            v-model="addForm.code"
            placeholder="如 600519 或 09678.HK"
            maxlength="10"
            clearable
            @input="onAddCodeInput"
          />
        </el-form-item>
        <el-form-item label="股票说明">
          <el-input
            v-model="addForm.remark"
            type="textarea"
            placeholder="可选，便于区分多只股票"
            :rows="3"
            maxlength="200"
            show-word-limit
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="addDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="addingStock" @click="handleAddStockSubmit">
          确定
        </el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="updatePeriodDialogVisible"
      title="选择更新周期"
      width="360px"
      :close-on-click-modal="true"
    >
      <p class="update-period-tip">选择要拉取的日线数据时间范围：</p>
      <el-radio-group v-model="updatePeriodValue" class="update-period-options">
        <el-radio label="last">最后更新日期至今</el-radio>
        <el-radio label="3y">近 3 年</el-radio>
        <el-radio label="5y">近 5 年</el-radio>
        <el-radio label="10y">近 10 年</el-radio>
      </el-radio-group>
      <template #footer>
        <el-button @click="updatePeriodDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="updatingAll" @click="confirmUpdateAll">
          开始更新
        </el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="editDialogVisible"
      title="修改股票信息"
      width="400px"
      :close-on-click-modal="false"
      @closed="resetEditForm"
    >
      <el-form :model="editForm" label-width="80px">
        <el-form-item label="股票代码">
          <el-input v-model="editForm.symbol" disabled />
        </el-form-item>
        <el-form-item label="股票名称">
          <el-input
            v-model="editForm.name"
            placeholder="选填，如贵州茅台"
            maxlength="128"
            show-word-limit
            clearable
          />
        </el-form-item>
        <el-form-item label="股票说明">
          <el-input
            v-model="editForm.remark"
            type="textarea"
            placeholder="选填，便于区分多只股票"
            :rows="3"
            maxlength="200"
            show-word-limit
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="savingRemark" @click="handleEditSubmit">
          保存
        </el-button>
      </template>
    </el-dialog>

    <div v-loading="appStore.loading" class="loading-wrap" />
    <el-alert v-if="appStore.errorMessage" type="error" :title="appStore.errorMessage" show-icon closable class="error-alert" @close="appStore.setError('')" />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useStockStore } from '@/stores/stock'
import { useAppStore } from '@/stores/app'
import { apiUpdateStockRemark } from '@/api/stock'

const router = useRouter()
const stockStore = useStockStore()
const appStore = useAppStore()

const addDialogVisible = ref(false)
const addForm = reactive({ code: '', remark: '' })
const addingStock = ref(false)
const updatingAll = ref(false)

const UPDATE_PERIOD_OPTIONS = [
  { id: '1m', label: '近 1 个月', payload: { months: 1 } },
  { id: '3y', label: '近 3 年', payload: { years: 3 } },
  { id: '5y', label: '近 5 年', payload: { years: 5 } },
  { id: '10y', label: '近 10 年', payload: { years: 10 } },
]
const updatePeriodDialogVisible = ref(false)
const updatePeriodValue = ref('last')

const editDialogVisible = ref(false)
const editForm = reactive({ symbol: '', name: '', remark: '' })
const savingRemark = ref(false)

function openUpdatePeriodDialog() {
  updatePeriodValue.value = 'last'
  updatePeriodDialogVisible.value = true
}

function getUpdatePeriodPayload() {
  const opt = UPDATE_PERIOD_OPTIONS.find((o) => o.id === updatePeriodValue.value)
  return opt ? opt.payload : { fromLastUpdate: true }
}

async function confirmUpdateAll() {
  await handleUpdateAll(getUpdatePeriodPayload())
  updatePeriodDialogVisible.value = false
}

function openChart(row) {
  const href = router.resolve({ path: '/chart', query: { symbol: row.filename } }).href
  const url = href.startsWith('http') ? href : `${window.location.origin}${href}`
  window.open(url, '_blank')
}

function openEditDialog(row) {
  editForm.symbol = row.filename || ''
  editForm.name = (row.displayName ?? '').trim()
  editForm.remark = (row.remark ?? '').trim()
  editDialogVisible.value = true
}

function resetEditForm() {
  editForm.symbol = ''
  editForm.name = ''
  editForm.remark = ''
}

async function handleEditSubmit() {
  if (!editForm.symbol) return
  savingRemark.value = true
  appStore.setError('')
  try {
    await apiUpdateStockRemark(editForm.symbol, editForm.remark.trim(), editForm.name.trim())
    ElMessage.success('已保存')
    editDialogVisible.value = false
    await stockStore.fetchList()
  } catch (e) {
    ElMessage.error('保存失败: ' + (e.message || ''))
  } finally {
    savingRemark.value = false
  }
}

function onAddCodeInput(val) {
  const v = (val || '').trim().toUpperCase()
  if (/\.HK$/.test(v)) {
    addForm.code = v
      .replace(/[^\d.]/g, '')
      .replace(/(\d{5})\.HK.*/, '$1.HK')
      .slice(0, 10)
  } else {
    addForm.code = (val || '').replace(/\D/g, '').slice(0, 6)
  }
}

function openAddDialog() {
  addForm.code = ''
  addForm.remark = ''
  addDialogVisible.value = true
}

function resetAddForm() {
  addForm.code = ''
  addForm.remark = ''
}

function isStockCodeValid(code) {
  if (!code || !code.trim()) return false
  const c = code.trim().toUpperCase()
  return (
    (c.length === 6 && /^\d{6}$/.test(c)) ||
    (c.length === 5 && /^\d{5}$/.test(c)) ||
    /^\d{5}\.HK$/.test(c)
  )
}

async function handleAddStockSubmit() {
  const code = addForm.code.trim()
  if (!isStockCodeValid(code)) {
    ElMessage.warning('请输入 A股6位 或 港股5位/xxxxx.HK')
    return
  }
  addingStock.value = true
  appStore.setError('')
  try {
    const data = await stockStore.addStock(code)
    if (data?.ok) {
      const remark = (addForm.remark || '').trim()
      if (remark) {
        try {
          await apiUpdateStockRemark(code, remark)
        } catch (_) {
          ElMessage.warning('已加入配置，但说明保存失败')
        }
      }
      ElMessage.success(data.message || '已加入配置')
      addDialogVisible.value = false
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

async function handleUpdateAll(options = { fromLastUpdate: true }) {
  updatingAll.value = true
  appStore.setError('')
  appStore.setLoading(true)
  try {
    const data = await stockStore.updateAll(options)
    if (data?.ok && data?.results) {
      const ok = data.results.filter((r) => r.ok).length
      const fail = data.results.filter((r) => !r.ok).length
      ElMessage.success('更新完成：成功 ' + ok + '，失败 ' + fail)
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

onMounted(async () => {
  await stockStore.fetchList()
})
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  text-align: center;
  font-weight: 600;
  margin-bottom: 8px;
  background: linear-gradient(90deg, #00d9ff, #00ff88);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.subtitle {
  text-align: center;
  color: #8892a0;
  margin-bottom: 8px;
}
.nav-row { margin-bottom: 16px; display: flex; align-items: center; gap: 12px; }
.nav-link { margin-left: 8px; }
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.list-card {
  margin-bottom: 20px;
}
.list-card .el-empty {
  padding: 24px;
}
.loading-wrap {
  min-height: 2px;
  margin-top: 16px;
}
.error-alert {
  margin-top: 16px;
}
.update-period-tip {
  color: #606266;
  margin: 0 0 16px;
  font-size: 14px;
}
 
</style>
