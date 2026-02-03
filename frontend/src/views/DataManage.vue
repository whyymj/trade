<template>
  <div class="container">
    <h1>数据管理</h1>
    <p class="subtitle">配置股票列表、日期范围，全量同步或单只抓取</p>

    <div class="nav-row">
      <router-link to="/" class="link">← 返回曲线</router-link>
    </div>

    <div class="card">
      <div class="card-title">当前配置</div>
      <div class="form-row">
        <label>开始日期</label>
        <input v-model="config.start_date" type="text" placeholder="YYYYMMDD 或空" />
      </div>
      <div class="form-row">
        <label>结束日期</label>
        <input v-model="config.end_date" type="text" placeholder="YYYYMMDD 或空" />
      </div>
      <div class="form-row">
        <label>复权</label>
        <select v-model="config.adjust">
          <option value="qfq">前复权</option>
          <option value="hfq">后复权</option>
          <option value="">不复权</option>
        </select>
      </div>
      <div class="form-row">
        <label>数据目录</label>
        <input v-model="config.output_dir" type="text" placeholder="如 data" />
      </div>
      <button type="button" class="btn btn-secondary" :disabled="savingConfig" @click="saveConfig">
        保存配置
      </button>
    </div>

    <div class="card">
      <div class="card-title">股票列表（config.stocks）</div>
      <ul class="stock-list">
        <li v-for="code in config.stocks" :key="code" class="stock-item">
          <span>{{ code }}</span>
          <button
            type="button"
            class="btn btn-small btn-danger"
            :disabled="removing === code"
            @click="removeStock(code)"
          >
            移除
          </button>
        </li>
      </ul>
      <div v-if="!config.stocks || !config.stocks.length" class="hint">暂无股票，请添加</div>
      <div class="form-row add-row">
        <input
          v-model="newCode"
          type="text"
          placeholder="如 600519 或 09678.HK"
          maxlength="10"
          @input="onCodeInput"
        />
        <button
          type="button"
          class="btn btn-primary"
          :disabled="addingStock || !newCode.trim()"
          @click="addStock"
        >
          抓取并加入
        </button>
      </div>
    </div>

    <div class="card actions">
      <div class="card-title">批量操作</div>
      <div class="btn-group">
        <button
          type="button"
          class="btn btn-primary"
          :disabled="syncing"
          @click="syncAll"
        >
          全量同步（清空后按配置拉取）
        </button>
        <button
          type="button"
          class="btn btn-secondary"
          :disabled="updatingAll"
          @click="updateAll"
        >
          一键更新（覆盖已有）
        </button>
      </div>
      <p class="hint">全量同步会先清空数据目录再拉取；一键更新只覆盖已有股票文件。</p>
    </div>

    <div v-show="appStore.loading" class="loading">加载中…</div>
    <div v-show="appStore.errorMessage" class="error">{{ appStore.errorMessage }}</div>
    <div v-show="appStore.toastVisible" class="toast show">{{ appStore.toastMessage }}</div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import {
  apiGetConfig,
  apiUpdateConfig,
  apiSyncAll,
  apiRemoveStock,
  apiAddStock,
  apiUpdateAll,
} from '@/api/stock'
import { useStockStore } from '@/stores/stock'

const appStore = useAppStore()
const stockStore = useStockStore()

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

function onCodeInput(e) {
  const v = e.target.value.trim().toUpperCase()
  if (/\.HK$/.test(v)) {
    newCode.value = v.replace(/[^\d.]/g, '').replace(/(\d{5})\.HK.*/, '$1.HK').slice(0, 10)
  } else {
    newCode.value = e.target.value.replace(/\D/g, '').slice(0, 6)
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
      appStore.showToast(out.message || '已保存')
    } else {
      appStore.showToast(out.message || '保存失败')
    }
  } catch (e) {
    appStore.showToast('保存失败: ' + (e.message || e.data?.message))
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
      appStore.showToast(data.message || '已加入')
      newCode.value = ''
      await loadConfig()
      await stockStore.fetchList()
    } else {
      appStore.showToast(data?.message || '失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
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
      appStore.showToast(out.message || '已移除')
      await loadConfig()
      await stockStore.fetchList()
    } else {
      appStore.showToast(out.message || '失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
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
      appStore.showToast(`全量同步完成：成功 ${ok}，失败 ${fail}`)
      await loadConfig()
      await stockStore.fetchList()
    } else {
      appStore.showToast('全量同步失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
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
      appStore.showToast(`更新完成：成功 ${ok}，失败 ${fail}`)
      await loadConfig()
      await stockStore.fetchList()
    } else {
      appStore.showToast('更新失败')
    }
  } catch (e) {
    appStore.showToast('请求失败: ' + (e.message || e.data?.message))
  } finally {
    appStore.setLoading(false)
    updatingAll.value = false
  }
}

onMounted(() => {
  loadConfig()
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
  color: #8892a0;
  margin-bottom: 16px;
}
.nav-row {
  margin-bottom: 20px;
}
.link {
  color: #00d9ff;
  text-decoration: none;
  font-size: 14px;
}
.link:hover {
  text-decoration: underline;
}
.card {
  background: rgba(30, 41, 59, 0.6);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.card-title {
  font-size: 15px;
  color: #94a3b8;
  margin-bottom: 16px;
}
.form-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.form-row label {
  width: 90px;
  color: #a0aec0;
  font-size: 14px;
}
.form-row input,
.form-row select {
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2d3748;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 14px;
  min-width: 160px;
}
.add-row {
  margin-top: 12px;
}
.stock-list {
  list-style: none;
  padding: 0;
  margin: 0 0 12px 0;
}
.stock-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
}
.btn-small {
  padding: 4px 10px;
  font-size: 12px;
}
.btn-danger {
  background: #7f1d1d;
  color: #fecaca;
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
  color: #64748b;
  font-size: 13px;
}
.loading,
.error {
  text-align: center;
  padding: 16px;
  color: #94a3b8;
}
.error {
  color: #f87171;
}
.toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 24px;
  border-radius: 8px;
  background: #1e293b;
  border: 1px solid #334155;
  color: #e2e8f0;
  font-size: 14px;
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.3s;
}
.toast.show {
  opacity: 1;
}
.btn {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn:hover:not(:disabled) {
  opacity: 0.9;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary {
  background: linear-gradient(135deg, #00d9ff, #00a8cc);
  color: #0f172a;
}
.btn-secondary {
  background: #334155;
  color: #e2e8f0;
}
</style>
