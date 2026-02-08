<template>
  <div class="container">
    <div class="nav-row">
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/')">股票列表</el-link>
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/lstm')">LSTM 训练/预测</el-link>
    </div>
    <h1>集成多因子预测</h1>
    <p class="subtitle">多因子绩效报告（IC / Rank IC）+ XGBoost / LightGBM / 随机森林 集成训练与权重优化</p>

    <el-card shadow="never" class="form-card">
      <template #header>
        <span class="card-header-title">参数</span>
      </template>
      <el-form :model="form" label-width="100px" class="params-form">
        <el-form-item label="股票">
          <el-select
            v-model="form.symbol"
            placeholder="请选择股票"
            filterable
            clearable
            style="width: 220px"
          >
            <el-option
              v-for="item in stockStore.fileList"
              :key="item.filename"
              :label="`${item.filename} ${item.displayName || ''}`.trim()"
              :value="item.filename"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="开始日期">
          <el-date-picker
            v-model="form.start"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="开始日期"
            style="width: 160px"
          />
        </el-form-item>
        <el-form-item label="结束日期">
          <el-date-picker
            v-model="form.end"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="结束日期"
            style="width: 160px"
          />
        </el-form-item>
        <el-form-item label="未来收益(天)">
          <el-input-number v-model="form.forward_days" :min="1" :max="20" style="width: 120px" />
        </el-form-item>
        <el-form-item label="滚动窗口">
          <el-input-number v-model="form.rolling_window" :min="5" :max="60" style="width: 120px" />
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="form-card section-card">
      <template #header>
        <div class="card-header-inner">
          <span class="card-header-title">因子绩效报告</span>
          <span class="card-header-desc">基于 200+ 量化因子的 IC、Rank IC、多空收益与换手率分析</span>
        </div>
      </template>
      <el-button
        type="primary"
        :loading="reportLoading"
        :disabled="!form.symbol || !form.start || !form.end"
        @click="handleFactorReport"
      >
        生成因子报告
      </el-button>
      <div v-if="reportError" class="error-msg">{{ reportError }}</div>
      <div v-if="reportResult" class="report-result">
        <div class="report-meta">
          <span>因子数 {{ reportResult.n_factors }}，观测数 {{ reportResult.n_observations }}</span>
        </div>
        <div v-if="reportResult.top_ic_factors?.length" class="top-factors">
          <div class="sub-title">IC 前 15 因子</div>
          <el-table :data="reportResult.top_ic_factors" size="small" stripe max-height="240">
            <el-table-column label="因子" min-width="180">
              <template #default="{ row }">{{ Array.isArray(row) ? row[0] : row.factor }}</template>
            </el-table-column>
            <el-table-column label="IC" width="100">
              <template #default="{ row }">{{ Array.isArray(row) && row[1] != null ? Number(row[1]).toFixed(4) : '—' }}</template>
            </el-table-column>
          </el-table>
        </div>
        <div v-if="reportResult.report_md" class="report-md-wrap">
          <div class="sub-title">报告全文</div>
          <pre class="report-md">{{ reportResult.report_md }}</pre>
        </div>
      </div>
    </el-card>

    <el-card shadow="never" class="form-card section-card">
      <template #header>
        <div class="card-header-inner">
          <span class="card-header-title">集成训练</span>
          <span class="card-header-desc">XGBoost + LightGBM + 随机森林，RFE 特征选择与最优权重分配</span>
        </div>
      </template>
      <el-button
        type="success"
        :loading="trainLoading"
        :disabled="!form.symbol || !form.start || !form.end"
        @click="handleTrain"
      >
        开始集成训练
      </el-button>
      <div v-if="trainError" class="error-msg">{{ trainError }}</div>
      <div v-if="trainResult" class="train-result">
        <el-descriptions :column="2" border size="small" class="train-metrics">
          <el-descriptions-item label="样本数">{{ trainResult.n_samples ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="使用特征数">{{ trainResult.n_features_used ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="验证 AUC">{{ trainResult.val_auc != null ? trainResult.val_auc.toFixed(4) : '—' }}</el-descriptions-item>
          <el-descriptions-item label="验证准确率">{{ trainResult.val_accuracy != null ? (trainResult.val_accuracy * 100).toFixed(2) + '%' : '—' }}</el-descriptions-item>
          <el-descriptions-item label="集成权重">XGB: {{ (trainResult.ensemble_weights?.[0] ?? 0).toFixed(3) }}，LGB: {{ (trainResult.ensemble_weights?.[1] ?? 0).toFixed(3) }}，RF: {{ (trainResult.ensemble_weights?.[2] ?? 0).toFixed(3) }}</el-descriptions-item>
          <el-descriptions-item label="保存路径" :span="2">{{ trainResult.artifacts_saved_to || '—' }}</el-descriptions-item>
        </el-descriptions>
        <div v-if="trainResult.rf_feature_importance?.length" class="rf-importance">
          <div class="sub-title">RF 特征重要性（前 20）</div>
          <el-table :data="trainResult.rf_feature_importance" size="small" stripe max-height="320">
            <el-table-column prop="feature" label="特征" min-width="160" show-overflow-tooltip />
            <el-table-column prop="importance" label="重要性" width="100">
              <template #default="{ row }">{{ (row.importance ?? 0).toFixed(4) }}</template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useStockStore } from '@/stores/stock'
import { apiEnsembleFactorReport, apiEnsembleTrain } from '@/api/stock'

const route = useRoute()
const stockStore = useStockStore()

const form = reactive({
  symbol: '',
  start: '',
  end: '',
  forward_days: 5,
  rolling_window: 20,
})

const reportLoading = ref(false)
const reportError = ref('')
const reportResult = ref(null)

const trainLoading = ref(false)
const trainError = ref('')
const trainResult = ref(null)

function defaultDateRange() {
  const end = new Date()
  const start = new Date()
  start.setFullYear(start.getFullYear() - 2)
  form.end = end.toISOString().slice(0, 10)
  form.start = start.toISOString().slice(0, 10)
}

async function handleFactorReport() {
  if (!form.symbol || !form.start || !form.end) return
  reportError.value = ''
  reportResult.value = null
  reportLoading.value = true
  try {
    const res = await apiEnsembleFactorReport({
      symbol: form.symbol,
      start: form.start,
      end: form.end,
      forward_days: form.forward_days,
      rolling_window: form.rolling_window,
    })
    reportResult.value = res
  } catch (e) {
    reportError.value = e.message || e.error || '请求失败'
  } finally {
    reportLoading.value = false
  }
}

async function handleTrain() {
  if (!form.symbol || !form.start || !form.end) return
  trainError.value = ''
  trainResult.value = null
  trainLoading.value = true
  try {
    const res = await apiEnsembleTrain({
      symbol: form.symbol,
      start: form.start,
      end: form.end,
      forward_days: form.forward_days,
      train_ratio: 0.7,
      use_rfe: true,
      rfe_min_features: 15,
      rfe_cv: 3,
      early_stopping_rounds: 30,
      optimize_weights: true,
    })
    trainResult.value = res
  } catch (e) {
    trainError.value = e.message || e.data?.error || '训练失败'
  } finally {
    trainLoading.value = false
  }
}

onMounted(async () => {
  await stockStore.fetchList()
  if (route.query.symbol) {
    form.symbol = route.query.symbol
  }
  if (stockStore.fileList.length && !form.symbol) {
    form.symbol = stockStore.fileList[0].filename
  }
  defaultDateRange()
})
</script>

<style scoped>
.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 20px 16px;
}
.nav-row {
  margin-bottom: 16px;
}
.nav-link {
  margin-right: 16px;
}
.subtitle {
  color: #909399;
  font-size: 14px;
  margin: 0 0 20px 0;
}
.form-card {
  margin-bottom: 20px;
}
.section-card {
  margin-top: 20px;
}
.card-header-inner {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.card-header-title {
  font-weight: 600;
}
.card-header-desc {
  font-size: 12px;
  color: #909399;
}
.params-form {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
}
.error-msg {
  color: #f56c6c;
  margin-top: 12px;
}
.report-result,
.train-result {
  margin-top: 16px;
}
.report-meta,
.sub-title {
  margin-bottom: 8px;
  font-size: 13px;
  color: #909399;
}
.sub-title {
  margin-top: 16px;
}
.top-factors {
  margin-bottom: 12px;
}
.report-md-wrap {
  margin-top: 12px;
}
.report-md {
  font-size: 12px;
  background: #1e1e1e;
  padding: 12px;
  border-radius: 6px;
  max-height: 400px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
.train-metrics {
  margin-top: 12px;
}
.rf-importance {
  margin-top: 16px;
}
</style>
