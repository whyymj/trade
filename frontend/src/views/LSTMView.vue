<template>
  <div class="container">
    <div class="nav-row">
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/')">股票列表</el-link>
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/chart')">股票曲线</el-link>
    </div>
    <h1>LSTM 训练与预测</h1>
    <p class="subtitle">控制训练参数、查看训练流水与版本、监控与告警、结果验证</p>

    <el-tabs v-model="activeTab" type="border-card" class="main-tabs">
      <!-- 训练与预测 -->
      <el-tab-pane label="训练与预测" name="train">
        <el-card shadow="never" class="form-card">
          <div class="form-title">数据周期与训练选项</div>
          <el-form :model="trainForm" label-width="120px" class="train-form compact">
            <el-form-item label="数据周期">
              <el-button-group class="range-btns">
                <el-button size="small" :loading="recommendedRangeLoading" @click="applyRecommendedRangePreset(1)">推荐 1 年</el-button>
                <el-button size="small" :loading="recommendedRangeLoading" @click="applyRecommendedRangePreset(2)">推荐 2 年</el-button>
              </el-button-group>
              <span v-if="dateRangeHint" class="range-hint">{{ dateRangeHint }}</span>
              <span class="range-dates">{{ trainForm.start && trainForm.end ? `${trainForm.start} ~ ${trainForm.end}` : '' }}</span>
            </el-form-item>
            <el-form-item label="训练选项">
              <div class="train-options-row">
                <el-checkbox v-model="trainForm.do_cv_tune">交叉验证与超参优化</el-checkbox>
                <el-checkbox v-model="trainForm.do_shap">SHAP</el-checkbox>
                <el-checkbox v-model="trainForm.do_plot">曲线图</el-checkbox>
                <el-checkbox v-model="trainForm.fast_training">快速训练</el-checkbox>
              </div>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">股票列表（表格选择）</div>
          <div class="table-toolbar">
            <el-button size="small" :loading="stocksStatusLoading" @click="loadStocksTrainingStatusForTrain">加载训练时间</el-button>
            <el-button type="primary" size="small" :loading="trainSelectedLoading" :disabled="!selectedTrainRows.length" @click="handleTrainSelected">
              一键训练选中项 ({{ selectedTrainRows.length }})
            </el-button>
            <el-button type="success" size="small" :loading="trainAllLoading" @click="handleTrainAll">一键训练全部</el-button>
          </div>
          <el-table
            ref="trainTableRef"
            v-loading="trainTableLoading"
            :data="trainTableData"
            size="small"
            stripe
            max-height="320"
            class="train-stock-table"
            @selection-change="onTrainTableSelectionChange"
          >
            <el-table-column type="selection" width="44" />
            <el-table-column prop="displayName" label="股票名称" min-width="100" show-overflow-tooltip />
            <el-table-column prop="symbol" label="股票代码" width="92" />
            <el-table-column label="一年最后训练" width="165">
              <template #default="{ row }">{{ row.last_train_1y || '—' }}</template>
            </el-table-column>
            <el-table-column label="二年最后训练" width="165">
              <template #default="{ row }">{{ row.last_train_2y || '—' }}</template>
            </el-table-column>
            <el-table-column label="操作" width="160" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" size="small" link :loading="trainingSymbol === row.symbol" @click="handleTrainOne(row.symbol)">
                  训练
                </el-button>
                <el-button type="success" size="small" link :loading="predicting && predictSymbol === row.symbol" @click="handlePredictFor(row.symbol)">
                  预测
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card v-if="trainingResultsList.length" shadow="never" class="form-card result-list-card">
          <div class="form-title">训练结果列表</div>
          <el-table :data="trainingResultsList" size="small" stripe max-height="320" class="result-list-table">
            <el-table-column prop="displayName" label="股票" width="100" show-overflow-tooltip />
            <el-table-column prop="symbol" label="代码" width="88" />
            <el-table-column label="版本" width="145">
              <template #default="{ row }">{{ row.result?.metadata?.version_id ?? (row.error ? '—' : '-') }}</template>
            </el-table-column>
            <el-table-column label="样本数" width="72">
              <template #default="{ row }">{{ row.result?.n_samples ?? '—' }}</template>
            </el-table-column>
            <el-table-column label="准确率" width="80">
              <template #default="{ row }">{{ formatPct(row.result?.metrics?.accuracy) }}</template>
            </el-table-column>
            <el-table-column label="F1" width="72">
              <template #default="{ row }">{{ formatPct(row.result?.metrics?.f1) }}</template>
            </el-table-column>
            <el-table-column label="MSE" width="88">
              <template #default="{ row }">{{ row.result?.metrics?.mse != null ? row.result.metrics.mse.toFixed(4) : '—' }}</template>
            </el-table-column>
            <el-table-column label="状态" width="70">
              <template #default="{ row }">
                <el-tag v-if="row.error" type="danger" size="small">失败</el-tag>
                <el-tag v-else type="success" size="small">成功</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" fixed="right">
              <template #default="{ row }">
                <el-button v-if="!row.error" type="success" size="small" link :loading="predicting && predictSymbol === row.symbol" @click="handlePredictFor(row.symbol)">
                  预测
                </el-button>
                <span v-else class="error-msg" :title="row.error">{{ (row.error || '').slice(0, 12) }}…</span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never" class="form-card predict-card">
          <div class="form-title">预测结果{{ predictSymbol ? `（${predictSymbol}）` : '' }}</div>
          <el-form inline>
            <el-form-item label="选项">
              <el-checkbox v-model="predictUseFallback">LSTM 不可用时回退</el-checkbox>
              <el-checkbox v-model="predictTriggerTrainAsync">预测后触发训练</el-checkbox>
            </el-form-item>
            <el-form-item v-if="!predictResult">
              <span class="hint">从上方表格或结果列表点击「预测」执行</span>
            </el-form-item>
          </el-form>
          <div v-if="predictResult" class="predict-result">
            <el-descriptions :column="2" border size="small" class="summary-desc">
              <el-descriptions-item label="方向">
                <span :class="predictResult.direction === 1 ? 'dir-up' : 'dir-down'">
                  {{ predictResult.direction_label ?? (predictResult.direction === 1 ? '涨' : '跌') }}
                </span>
              </el-descriptions-item>
              <el-descriptions-item label="预测来源">
                <el-tag size="small">{{ predictResult.source || 'lstm' }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="预测涨跌幅">{{ formatPct(predictResult.magnitude, 4) }}</el-descriptions-item>
              <el-descriptions-item label="上涨概率">{{ formatPct(predictResult.prob_up) }}</el-descriptions-item>
              <el-descriptions-item label="下跌概率">{{ formatPct(predictResult.prob_down) }}</el-descriptions-item>
              <el-descriptions-item v-if="predictResult.model_health" label="模型健康" :span="2">
                <el-tag :type="predictResult.model_health.healthy ? 'success' : 'warning'" size="small">
                  {{ predictResult.model_health.healthy ? '正常' : '异常' }}
                </el-tag>
                <span v-if="predictResult.model_health.message" class="health-msg">{{ predictResult.model_health.message }}</span>
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- 训练流水与版本 -->
      <el-tab-pane label="训练流水与版本" name="runs">
        <el-card shadow="never" class="form-card">
          <div class="form-title">全部股票与训练时间</div>
          <el-button :loading="stocksStatusLoading" @click="loadStocksTrainingStatus">加载</el-button>
          <el-table v-loading="stocksStatusLoading" :data="stocksTrainingStatus" size="small" stripe max-height="400" class="stocks-status-table">
            <el-table-column prop="displayName" label="股票名称" min-width="120" show-overflow-tooltip />
            <el-table-column prop="symbol" label="股票代码" width="100" />
            <el-table-column label="一年数据最后训练时间" width="200">
              <template #default="{ row }">
                {{ row.last_train_1y || '—' }}
              </template>
            </el-table-column>
            <el-table-column label="二年数据最后训练时间" width="200">
              <template #default="{ row }">
                {{ row.last_train_2y || '—' }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">训练流水</div>
          <el-form inline>
            <el-form-item label="股票">
              <el-select
                v-model="runsSymbol"
                placeholder="全部"
                clearable
                filterable
                style="width: 180px"
              >
                <el-option
                  v-for="item in stockStore.fileList"
                  :key="item.filename"
                  :label="`${item.filename} ${item.displayName || ''}`"
                  :value="item.filename"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="条数">
              <el-input-number v-model="runsLimit" :min="10" :max="200" :step="10" style="width: 100px" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="runsLoading" @click="loadTrainingRuns">查询</el-button>
            </el-form-item>
          </el-form>
          <el-table v-loading="runsLoading" :data="trainingRuns" size="small" stripe max-height="320" class="runs-table">
            <el-table-column prop="version_id" label="版本" width="140" />
            <el-table-column prop="symbol" label="股票" width="90" />
            <el-table-column prop="training_type" label="类型" width="90" />
            <el-table-column prop="trigger_type" label="触发" width="100" />
            <el-table-column prop="data_start" label="数据起" width="100" />
            <el-table-column prop="data_end" label="数据止" width="100" />
            <el-table-column prop="validation_deployed" label="已部署" width="70">
              <template #default="{ row }">
                <el-tag :type="row.validation_deployed ? 'success' : 'info'" size="small">
                  {{ row.validation_deployed ? '是' : '否' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="duration_seconds" label="耗时(秒)" width="85" />
            <el-table-column prop="created_at" label="创建时间" width="165" />
          </el-table>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">模型版本与回滚</div>
          <el-button :loading="versionsLoading" @click="loadVersions">刷新版本列表</el-button>
          <el-table v-loading="versionsLoading" :data="versionsList" size="small" stripe class="versions-table">
            <el-table-column prop="version_id" label="版本 ID" width="160" />
            <el-table-column prop="training_time" label="训练时间" width="180" />
            <el-table-column prop="data_start" label="数据起" width="100" />
            <el-table-column prop="data_end" label="数据止" width="100" />
            <el-table-column prop="validation_score" label="验证分" width="90" />
            <el-table-column label="操作" width="100">
              <template #default="{ row }">
                <el-button
                  v-if="currentVersionId !== row.version_id"
                  type="warning"
                  size="small"
                  @click="handleRollback(row.version_id)"
                >
                  回滚
                </el-button>
                <el-tag v-else type="success" size="small">当前</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">更新预测准确性</div>
          <p class="hint">根据实际行情回填预测准确性（预测日+5 交易日后可计算），供性能衰减判断。</p>
          <el-form inline>
            <el-form-item label="股票">
              <el-select
                v-model="accuracySymbol"
                placeholder="选择股票"
                filterable
                style="width: 200px"
              >
                <el-option
                  v-for="item in stockStore.fileList"
                  :key="item.filename"
                  :label="`${item.filename} ${item.displayName || ''}`"
                  :value="item.filename"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="截止日期">
              <el-input v-model="accuracyAsOfDate" placeholder="可选，默认今天" clearable style="width: 140px" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="accuracyLoading" @click="handleUpdateAccuracy">回填</el-button>
            </el-form-item>
          </el-form>
          <span v-if="accuracyResult != null" class="result-inline">已更新 {{ accuracyResult.updated_count }} 条</span>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">检查/执行训练触发</div>
          <p class="hint">检查周五周度、月末完整、性能衰减等触发；勾选“执行”时满足条件则自动训练。</p>
          <el-form inline>
            <el-form-item label="股票">
              <el-select
                v-model="triggersSymbol"
                placeholder="选择股票"
                filterable
                style="width: 200px"
              >
                <el-option
                  v-for="item in stockStore.fileList"
                  :key="item.filename"
                  :label="`${item.filename} ${item.displayName || ''}`"
                  :value="item.filename"
                />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-checkbox v-model="triggersRun">满足条件时执行训练</el-checkbox>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="triggersLoading" @click="handleCheckTriggers">检查触发</el-button>
            </el-form-item>
          </el-form>
          <div v-if="triggersResult" class="triggers-result">
            <div>触发结果：</div>
            <pre class="metrics-pre">{{ JSON.stringify(triggersResult.triggers || {}, null, 2) }}</pre>
            <div v-if="triggersResult.training">训练结果：<pre class="metrics-pre">{{ JSON.stringify(triggersResult.training, null, 2) }}</pre></div>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- 监控与告警 -->
      <el-tab-pane label="监控与告警" name="monitor">
        <el-card shadow="never" class="form-card">
          <div class="form-title">监控状态</div>
          <el-button :loading="monitoringLoading" @click="loadMonitoring">刷新</el-button>
          <div v-if="monitoringData" class="monitoring-grid">
            <el-descriptions :column="2" border size="small" class="summary-desc">
              <el-descriptions-item label="当前版本">{{ monitoringData.current_version_id ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="最后训练时间">{{ monitoringData.last_training_time ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="数据范围">{{ (monitoringData.data_start || '') + ' ~ ' + (monitoringData.data_end || '-') }}</el-descriptions-item>
              <el-descriptions-item label="验证分数">{{ monitoringData.validation_score ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="近 7 日预测次数">{{ monitoringData.prediction_count_7d ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="MAE">{{ monitoringData.mae ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="RMSE">{{ monitoringData.rmse ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="方向准确率">{{ formatPct(monitoringData.direction_accuracy) }}</el-descriptions-item>
              <el-descriptions-item label="近期平均误差">{{ monitoringData.accuracy_recent_avg_error ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="历史平均误差">{{ monitoringData.accuracy_historical_avg_error ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="训练失败次数">{{ monitoringData.training_failure_count ?? 0 }}</el-descriptions-item>
              <el-descriptions-item v-if="monitoringData.performance_decay" label="性能衰减" :span="2">
                <pre class="metrics-pre small">{{ JSON.stringify(monitoringData.performance_decay, null, 2) }}</pre>
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">性能衰减检测</div>
          <el-form inline>
            <el-form-item label="阈值倍数">
              <el-input-number v-model="decayThreshold" :min="1" :max="3" :step="0.1" style="width: 100px" />
            </el-form-item>
            <el-form-item label="最近 N 条">
              <el-input-number v-model="decayNRecent" :min="5" :max="50" style="width: 100px" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="decayLoading" @click="handlePerformanceDecay">执行检测</el-button>
            </el-form-item>
          </el-form>
          <div v-if="decayResult" class="decay-result">
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="是否触发">{{ decayResult.triggered ? '是' : '否' }}</el-descriptions-item>
              <el-descriptions-item label="近期平均误差">{{ decayResult.recent_avg_error ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="历史平均误差">{{ decayResult.historical_avg_error ?? '-' }}</el-descriptions-item>
              <el-descriptions-item label="检测时间">{{ decayResult.detected_at ?? '-' }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>

        <el-card shadow="never" class="form-card">
          <div class="form-title">告警</div>
          <el-button :loading="alertsLoading" @click="loadAlerts">检查告警</el-button>
          <el-button type="danger" :loading="alertsFiring" :disabled="!alertsList.length" @click="fireAlerts">
            发送告警通知 ({{ alertsList.length }})
          </el-button>
          <el-table :data="alertsList" size="small" stripe class="alerts-table">
            <el-table-column prop="type" label="类型" width="180" />
            <el-table-column prop="severity" label="级别" width="80" />
            <el-table-column prop="message" label="说明" />
            <el-table-column prop="at" label="时间" width="165" />
          </el-table>
          <div v-if="alertsFired" class="fired-info">
            <el-tag type="success">已记录</el-tag>
            <span v-if="alertsFired.webhook_sent"> 已发送 webhook</span>
          </div>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon closable class="error-alert" @close="errorMessage = ''" />
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useStockStore } from '@/stores/stock'
import {
  apiLstmTrain,
  apiLstmTrainAll,
  apiLstmLastPrediction,
  apiLstmPredict,
  apiLstmStocksTrainingStatus,
  apiLstmTrainingRuns,
  apiLstmVersions,
  apiLstmRecommendedRange,
  apiLstmRollback,
  apiLstmCheckTriggers,
  apiLstmUpdateAccuracy,
  apiLstmMonitoring,
  apiLstmPerformanceDecay,
  apiLstmAlerts,
} from '@/api/stock'

const route = useRoute()
const stockStore = useStockStore()

const activeTab = ref('train')

// 训练
const trainForm = ref({
  start: '',
  end: '',
  do_cv_tune: true,
  do_shap: true,
  do_plot: true,
  fast_training: false,
})
const recommendedRangeLoading = ref(false)
const dateRangeHint = ref('')
const trainTableRef = ref(null)
const selectedTrainRows = ref([])
const trainTableData = ref([])
const trainTableLoading = ref(false)
const trainingSymbol = ref('')
const trainSelectedLoading = ref(false)
const trainAllLoading = ref(false)
const trainingResultsList = ref([])

// 预测
const predictSymbol = ref('')
const predictUseFallback = ref(false)
const predictTriggerTrainAsync = ref(false)
const predicting = ref(false)
const predictResult = ref(null)

// 全部股票训练状态
const stocksStatusLoading = ref(false)
const stocksTrainingStatus = ref([])

// 训练流水
const runsSymbol = ref('')
const runsLimit = ref(50)
const runsLoading = ref(false)
const trainingRuns = ref([])

// 版本
const versionsLoading = ref(false)
const versionsList = ref([])
const currentVersionId = ref(null)

// 更新准确性
const accuracySymbol = ref('')
const accuracyAsOfDate = ref('')
const accuracyLoading = ref(false)
const accuracyResult = ref(null)

// 触发
const triggersSymbol = ref('')
const triggersRun = ref(false)
const triggersLoading = ref(false)
const triggersResult = ref(null)

// 监控
const monitoringLoading = ref(false)
const monitoringData = ref(null)

// 性能衰减
const decayThreshold = ref(1.5)
const decayNRecent = ref(20)
const decayLoading = ref(false)
const decayResult = ref(null)

// 告警
const alertsLoading = ref(false)
const alertsFiring = ref(false)
const alertsList = ref([])
const alertsFired = ref(null)

const errorMessage = ref('')

/** 合并股票列表与训练时间，供训练页表格使用 */
function buildTrainTableData() {
  const files = stockStore.fileList || []
  const statusMap = {}
  for (const s of stocksTrainingStatus.value) {
    if (s.symbol) statusMap[s.symbol] = { last_train_1y: s.last_train_1y, last_train_2y: s.last_train_2y }
  }
  trainTableData.value = files.map((f) => ({
    symbol: f.filename,
    displayName: f.displayName || f.filename || '',
    last_train_1y: statusMap[f.filename]?.last_train_1y ?? null,
    last_train_2y: statusMap[f.filename]?.last_train_2y ?? null,
  }))
}

function onTrainTableSelectionChange(rows) {
  selectedTrainRows.value = rows || []
}

function formatPct(val, decimals = 2) {
  if (val == null) return '-'
  const n = Number(val)
  if (Number.isNaN(n)) return '-'
  return (n * 100).toFixed(decimals) + '%'
}

function setError(msg) {
  errorMessage.value = msg || ''
}

/** 拉取推荐周期并填入表单。preset: 1 | 2 */
async function applyRecommendedRangePreset(preset) {
  recommendedRangeLoading.value = true
  dateRangeHint.value = ''
  try {
    const res = await apiLstmRecommendedRange({ years: preset })
    if (res?.start) trainForm.value.start = res.start
    if (res?.end) trainForm.value.end = res.end
    if (res?.hint) dateRangeHint.value = res.hint
    if (res?.start && res?.end) ElMessage.success('已填入推荐周期')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '获取推荐周期失败'
    ElMessage.error(msg)
  } finally {
    recommendedRangeLoading.value = false
  }
}

/** 进入页面时若起止为空则自动加载推荐 1 年 */
async function maybeAutoLoadRecommendedRange() {
  if ((trainForm.value.start || '').trim() && (trainForm.value.end || '').trim()) return
  recommendedRangeLoading.value = true
  dateRangeHint.value = ''
  try {
    const res = await apiLstmRecommendedRange({ years: 1 })
    if (res?.start) trainForm.value.start = res.start
    if (res?.end) trainForm.value.end = res.end
    if (res?.hint) dateRangeHint.value = res.hint
  } catch (_) {
    // 静默失败，用户可点「推荐 1 年」/「推荐 2 年」
  } finally {
    recommendedRangeLoading.value = false
  }
}

async function handleTrainOne(symbol) {
  const sym = (symbol || '').trim()
  if (!sym) return
  setError('')
  trainingSymbol.value = sym
  const displayName = (trainTableData.value.find((r) => r.symbol === sym) || {}).displayName || sym
  try {
    const body = {
      symbol: sym,
      start: (trainForm.value.start || '').trim() || undefined,
      end: (trainForm.value.end || '').trim() || undefined,
      do_cv_tune: trainForm.value.do_cv_tune,
      do_shap: trainForm.value.do_shap,
      do_plot: trainForm.value.do_plot,
      fast_training: trainForm.value.fast_training,
    }
    const res = await apiLstmTrain(body)
    if (res.error) {
      trainingResultsList.value = [{ symbol: sym, displayName, error: res.error }, ...trainingResultsList.value]
      ElMessage.warning(res.error)
    } else {
      trainingResultsList.value = [{ symbol: sym, displayName, result: res, error: null }, ...trainingResultsList.value]
      ElMessage.success(sym + ' 训练完成')
      await loadStocksTrainingStatusForTrain(true)
    }
  } catch (e) {
    const msg = e?.data?.error || e?.message || '训练失败'
    trainingResultsList.value = [{ symbol: sym, displayName, error: msg, result: null }, ...trainingResultsList.value]
    setError(msg)
    ElMessage.error(msg)
  } finally {
    trainingSymbol.value = ''
  }
}

async function handleTrainSelected() {
  const rows = selectedTrainRows.value
  if (!rows.length) {
    ElMessage.warning('请先勾选要训练的股票')
    return
  }
  setError('')
  trainSelectedLoading.value = true
  for (const row of rows) {
    await handleTrainOne(row.symbol)
  }
  trainSelectedLoading.value = false
  await loadStocksTrainingStatusForTrain(true)
  ElMessage.success('选中项训练已全部完成')
}

async function handleTrainAll() {
  if (!stockStore.fileList?.length) {
    ElMessage.warning('当前无股票数据，请先在股票列表或数据管理中添加')
    return
  }
  setError('')
  trainAllLoading.value = true
  try {
    const start = (trainForm.value.start || '').trim()
    const end = (trainForm.value.end || '').trim()
    const body = {
      start: start || undefined,
      end: end || undefined,
      years: !start && !end ? 1 : undefined,
      do_cv_tune: trainForm.value.do_cv_tune,
      do_shap: trainForm.value.do_shap,
      do_plot: trainForm.value.do_plot,
      fast_training: trainForm.value.fast_training,
    }
    const res = await apiLstmTrainAll(body)
    const nameMap = {}
    for (const r of trainTableData.value) {
      nameMap[r.symbol] = r.displayName || r.symbol
    }
    const newEntries = (res.results || []).map((item) => ({
      symbol: item.symbol,
      displayName: nameMap[item.symbol] || item.symbol,
      result: item.ok ? { metadata: { version_id: item.version_id }, n_samples: null, metrics: {} } : null,
      error: item.ok ? null : (item.error || '失败'),
    }))
    trainingResultsList.value = [...newEntries, ...trainingResultsList.value]
    await loadStocksTrainingStatusForTrain(true)
    ElMessage.success(`全部训练完成：成功 ${res.success_count ?? 0} 只，失败 ${res.fail_count ?? 0} 只`)
  } catch (e) {
    const msg = e?.data?.error || e?.message || '一键训练失败'
    setError(msg)
    ElMessage.error(msg)
  } finally {
    trainAllLoading.value = false
  }
}

function handlePredictFor(symbol) {
  predictSymbol.value = (symbol || '').trim()
  if (predictSymbol.value) handlePredict()
}

async function handlePredict() {
  const symbol = (predictSymbol.value || '').trim()
  if (!symbol) {
    ElMessage.warning('请从表格或结果列表点击「预测」选择股票')
    return
  }
  setError('')
  predicting.value = true
  predictResult.value = null
  try {
    const res = await apiLstmPredict(symbol, {
      use_fallback: predictUseFallback.value,
      trigger_train_async: predictTriggerTrainAsync.value,
    })
    predictResult.value = res
    ElMessage.success('预测完成')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '预测失败'
    setError(msg)
    ElMessage.error(msg)
  } finally {
    predicting.value = false
  }
}

async function loadStocksTrainingStatus() {
  stocksStatusLoading.value = true
  try {
    const res = await apiLstmStocksTrainingStatus()
    stocksTrainingStatus.value = res.stocks || []
    ElMessage.success('已加载 ' + (res.stocks?.length ?? 0) + ' 只股票')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '加载失败'
    ElMessage.error(msg)
  } finally {
    stocksStatusLoading.value = false
  }
}

async function loadStocksTrainingStatusForTrain(silent = false) {
  trainTableLoading.value = true
  try {
    const res = await apiLstmStocksTrainingStatus()
    stocksTrainingStatus.value = res.stocks || []
    buildTrainTableData()
    if (!silent) ElMessage.success('已加载训练时间')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '加载失败'
    ElMessage.error(msg)
  } finally {
    trainTableLoading.value = false
  }
}

/** 用最近一次（或若干次）训练流水填充训练结果列表，作为默认展示。 */
async function loadLastTrainingResultsIntoList() {
  try {
    const res = await apiLstmTrainingRuns({ limit: 10 })
    const runs = res.runs || []
    if (!runs.length) return
    const nameMap = {}
    for (const f of stockStore.fileList || []) {
      nameMap[f.filename] = f.displayName || f.filename || ''
    }
    const entries = runs.map((run) => ({
      symbol: run.symbol || '',
      displayName: nameMap[run.symbol] || run.symbol || '',
      result: {
        metadata: { version_id: run.version_id || null },
        n_samples: run.params?.n_samples ?? null,
        metrics: run.metrics || {},
      },
      error: null,
    }))
    trainingResultsList.value = entries
  } catch (_) {
    // 流水不可用时保持列表为空
  }
}

async function loadTrainingRuns() {
  runsLoading.value = true
  try {
    const res = await apiLstmTrainingRuns({
      symbol: (runsSymbol.value || '').trim() || undefined,
      limit: runsLimit.value,
    })
    trainingRuns.value = res.runs || []
  } catch (e) {
    const msg = e?.data?.error || e?.message || '查询失败'
    ElMessage.error(msg)
  } finally {
    runsLoading.value = false
  }
}

async function loadVersions() {
  versionsLoading.value = true
  try {
    const res = await apiLstmVersions()
    currentVersionId.value = res.current_version_id ?? null
    versionsList.value = res.versions || []
  } catch (e) {
    const msg = e?.data?.error || e?.message || '获取版本失败'
    ElMessage.error(msg)
  } finally {
    versionsLoading.value = false
  }
}

async function handleRollback(versionId) {
  try {
    await apiLstmRollback(versionId)
    currentVersionId.value = versionId
    ElMessage.success('已回滚到 ' + versionId)
    loadVersions()
  } catch (e) {
    const msg = e?.data?.error || e?.message || '回滚失败'
    ElMessage.error(msg)
  }
}

async function handleUpdateAccuracy() {
  const symbol = (accuracySymbol.value || '').trim()
  if (!symbol) {
    ElMessage.warning('请选择股票')
    return
  }
  accuracyLoading.value = true
  accuracyResult.value = null
  try {
    const res = await apiLstmUpdateAccuracy({
      symbol,
      as_of_date: (accuracyAsOfDate.value || '').trim() || undefined,
    })
    accuracyResult.value = res
    ElMessage.success('已更新 ' + res.updated_count + ' 条')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '回填失败'
    ElMessage.error(msg)
  } finally {
    accuracyLoading.value = false
  }
}

async function handleCheckTriggers() {
  const symbol = (triggersSymbol.value || '').trim()
  if (!symbol) {
    ElMessage.warning('请选择股票')
    return
  }
  triggersLoading.value = true
  triggersResult.value = null
  try {
    const res = await apiLstmCheckTriggers({
      symbol,
      run: triggersRun.value,
    })
    triggersResult.value = res
    ElMessage.success(triggersRun.value && res.training ? '已检查并执行训练' : '已检查触发')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '检查触发失败'
    ElMessage.error(msg)
  } finally {
    triggersLoading.value = false
  }
}

async function loadMonitoring() {
  monitoringLoading.value = true
  try {
    const res = await apiLstmMonitoring()
    monitoringData.value = res
  } catch (e) {
    const msg = e?.data?.error || e?.message || '获取监控失败'
    ElMessage.error(msg)
  } finally {
    monitoringLoading.value = false
  }
}

async function handlePerformanceDecay() {
  decayLoading.value = true
  decayResult.value = null
  try {
    const res = await apiLstmPerformanceDecay({
      threshold: decayThreshold.value,
      n_recent: decayNRecent.value,
      log: true,
    })
    decayResult.value = res
    ElMessage.success('检测完成')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '检测失败'
    ElMessage.error(msg)
  } finally {
    decayLoading.value = false
  }
}

async function loadAlerts() {
  alertsLoading.value = true
  alertsFired.value = null
  try {
    const res = await apiLstmAlerts({})
    alertsList.value = res.alerts || []
    if (res.count === 0) ElMessage.info('当前无告警')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '获取告警失败'
    ElMessage.error(msg)
  } finally {
    alertsLoading.value = false
  }
}

async function fireAlerts() {
  if (!alertsList.value.length) return
  alertsFiring.value = true
  try {
    const res = await apiLstmAlerts({ fire: true })
    alertsFired.value = res.fired || {}
    ElMessage.success('已发送告警通知')
  } catch (e) {
    const msg = e?.data?.error || e?.message || '发送失败'
    ElMessage.error(msg)
  } finally {
    alertsFiring.value = false
  }
}

watch(
  () => route.query.symbol,
  (sym) => {
    if (sym) {
      trainForm.value.symbol = sym
      predictSymbol.value = sym
    }
  },
  { immediate: true }
)

onMounted(async () => {
  await stockStore.fetchList()
  buildTrainTableData()
  const firstSymbol = route.query.symbol || (stockStore.fileList.length > 0 ? stockStore.fileList[0].filename : '')
  predictSymbol.value = firstSymbol
  accuracySymbol.value = firstSymbol
  triggersSymbol.value = firstSymbol
  await maybeAutoLoadRecommendedRange()
  await loadLastTrainingResultsIntoList()
  if (predictSymbol.value) {
    try {
      const last = await apiLstmLastPrediction(predictSymbol.value)
      if (last && (last.direction !== undefined || last.magnitude !== undefined)) {
        predictResult.value = last
      }
    } catch (_) {
      // 无历史预测或接口不可用时忽略
    }
  }
})
</script>

<style scoped>
.container {
  max-width: 960px;
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
  margin-bottom: 24px;
}
.nav-row {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.nav-link {
  margin-left: 8px;
}
.main-tabs {
  margin-bottom: 20px;
}
.form-card,
.result-card,
.predict-card {
  margin-bottom: 20px;
}
.form-title {
  font-weight: 600;
  margin-bottom: 12px;
  color: #e8e8e8;
}
.train-form :deep(.el-form-item) {
  margin-bottom: 16px;
}
.train-form.compact :deep(.el-form-item) {
  margin-bottom: 12px;
}
.table-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 12px;
}
.train-stock-table,
.result-list-table {
  margin-top: 0;
}
.result-list-card {
  margin-top: 16px;
}
.error-msg {
  font-size: 12px;
  color: #f56c6c;
}
.range-btns {
  margin-left: 12px;
  vertical-align: middle;
}
.range-hint {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
.range-dates {
  margin-left: 12px;
  color: #c0c4cc;
  font-size: 12px;
}
.train-options-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px 24px;
}
.train-options-desc {
  margin-top: 10px;
  padding: 10px 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  font-size: 12px;
  color: #a0a8b4;
  line-height: 1.6;
}
.train-options-desc div {
  margin-bottom: 6px;
}
.train-options-desc div:last-child {
  margin-bottom: 0;
}
.train-options-desc strong {
  color: #c0c4cc;
}
.train-all-btn {
  margin-left: 12px;
}
.train-all-summary {
  margin-bottom: 12px;
  font-size: 14px;
  color: #e8e8e8;
}
.train-all-note {
  display: block;
  margin-top: 6px;
  font-size: 12px;
  color: #909399;
}
.train-all-table {
  margin-top: 8px;
}
.summary-desc {
  margin-bottom: 16px;
}
.validation-block,
.feature-importance {
  margin-top: 16px;
}
.plot-wrap {
  margin-top: 20px;
}
.plot-img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid #2c3e50;
}
.predict-result {
  margin-top: 16px;
}
.health-msg {
  margin-left: 8px;
  color: #909399;
}
.dir-up { color: #67c23a; }
.dir-down { color: #f56c6c; }
.metrics-pre {
  font-size: 12px;
  background: #1e1e1e;
  padding: 8px;
  border-radius: 4px;
  margin: 4px 0;
  max-height: 160px;
  overflow: auto;
}
.metrics-pre.small {
  max-height: 100px;
}
.hint {
  color: #909399;
  font-size: 12px;
  margin-bottom: 12px;
}
.stocks-status-table {
  margin-top: 12px;
}
.runs-table,
.versions-table,
.alerts-table {
  margin-top: 12px;
}
.triggers-result,
.decay-result {
  margin-top: 12px;
}
.result-inline {
  margin-left: 12px;
  color: #67c23a;
}
.monitoring-grid {
  margin-top: 12px;
}
.fired-info {
  margin-top: 8px;
}
.error-alert {
  margin-top: 16px;
}
</style>
