<template>
  <div class="container">
    <div class="nav-row">
      <el-link type="primary" underline="never" class="nav-link" @click="$router.push('/')">股票列表</el-link>
    </div>
    <h1>LSTM 训练与预测</h1>
    <p class="subtitle">按股票与 1/2/3 年分别训练与预测，查看流水、版本与监控</p>

    <el-tabs v-model="activeTab" type="border-card" class="main-tabs">
      <!-- 训练与预测 -->
      <el-tab-pane label="训练与预测" name="train">
        <!-- 训练（表格含最近一次训练结果） -->
        <el-card shadow="never" class="form-card section-card">
          <template #header>
            <div class="card-header-inner">
              <div>
                <span class="card-header-title">训练</span>
                <span class="card-header-desc">勾选股票后执行训练（每只股票将依次训练 1/2/3 年数据并分别保存模型）；表格中展示最近一次训练结果</span>
              </div>
            </div>
          </template>
          <div class="table-toolbar">
            <div class="toolbar-group">
              <span class="toolbar-label">训练年份</span>
              <el-select v-model="trainForm.trainYears" size="small" class="train-years-select" placeholder="选择年份">
                <el-option label="全部 (1+2+3 年)" value="all" />
                <el-option label="仅 1 年" :value="1" />
                <el-option label="仅 2 年" :value="2" />
                <el-option label="仅 3 年" :value="3" />
              </el-select>
            </div>
            <div class="toolbar-group">
              <el-button type="primary" size="small" :loading="trainSelectedLoading" :disabled="!selectedTrainRows.length || trainAllLoading" @click="handleTrainSelected">
                训练选中 ({{ selectedTrainRows.length }})
              </el-button>
              <el-button type="success" size="small" :loading="trainAllLoading" :disabled="trainSelectedLoading || clearTrainingLoading" @click="handleTrainAll">
                训练全部
              </el-button>
              <el-button type="warning" size="small" plain :loading="clearTrainingLoading" :disabled="!selectedTrainRows.length || trainAllLoading" @click="handleClearTraining">
                清理选中训练数据
              </el-button>
              <el-button
                v-if="trainingSymbols.length || trainAllLoading"
                type="danger"
                size="small"
                plain
                :disabled="stopTrainingLoading"
                @click="handleStopTraining"
              >
                停止训练
              </el-button>
            </div>
            <span v-if="trainAllLoading && trainAllProgress.total" class="train-all-progress">
              正在训练 {{ trainAllProgress.current }}/{{ trainAllProgress.total }} {{ trainAllProgress.displayName || trainAllProgress.symbol }}
            </span>
          </div>
          <el-table
            ref="trainTableRef"
            v-loading="trainTableLoading"
            :data="mergedTrainTableData"
            row-key="symbol"
            size="small"
            stripe
            max-height="420"
            class="train-stock-table"
            @selection-change="onTrainTableSelectionChange"
          >
            <el-table-column type="selection" width="44" />
            <el-table-column prop="displayName" label="股票名称" min-width="100" show-overflow-tooltip />
            <el-table-column prop="symbol" label="股票代码" width="92" />
            <el-table-column label="最后一次训练" width="165">
              <template #default="{ row }">{{ row.last_train || '—' }}</template>
            </el-table-column>
            <el-table-column label="已训年份" width="88" title="1/2/3 年是否齐全">
              <template #default="{ row }">
                <span v-if="(row.trained_years || []).length">{{ (row.trained_years || []).join('、') }} 年</span>
                <span v-else>—</span>
                <el-tag v-if="hasAllThreeYearsTrained(row.trained_years)" type="success" size="small">齐</el-tag>
                <el-tag v-else-if="(row.trained_years || []).length" type="warning" size="small">缺</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="版本" width="130" show-overflow-tooltip>
              <template #default="{ row }">{{ row.result?.metadata?.version_id ?? '—' }}</template>
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
            <el-table-column label="诊断" width="80" align="center">
              <template #default="{ row }">
                <el-tooltip v-if="row.result?.diagnostics && Object.keys(row.result.diagnostics).length" placement="top" :show-after="300">
                  <template #content>
                    <div class="diagnostics-tooltip">
                      <div v-for="([key, msg]) in diagnosticsOrdered(row.result.diagnostics)" :key="key" class="diagnostics-tooltip-item">{{ msg }}</div>
                    </div>
                  </template>
                  <el-tag v-if="hasDiagnosticsRisk(row.result.diagnostics)" type="warning" size="small">有风险</el-tag>
                  <el-tag v-else type="info" size="small">正常</el-tag>
                </el-tooltip>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column label="状态" width="100">
              <template #default="{ row }">
                <el-tag v-if="row.error" type="danger" size="small">失败</el-tag>
                <el-tag v-else-if="row.result" type="success" size="small">成功</el-tag>
                <span v-else>—</span>
                <span v-if="row.error" class="error-msg" :title="row.error">{{ (row.error || '').slice(0, 12) }}…</span>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="88" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" size="small" link :loading="trainingSymbols.includes(row.symbol)" @click="handleTrainOne(row.symbol)">
                  训练
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never" class="form-card section-card predict-card">
          <template #header>
            <div class="card-header-inner">
              <div>
                <span class="card-header-title">预测</span>
                <span class="card-header-desc">默认使用 1/2/3 年模型分别预测并记录；展开行可查看预测详情与拟合曲线</span>
              </div>
              <el-button size="small" class="plot-refresh-btn" @click="refreshPlotUrl">刷新曲线</el-button>
            </div>
          </template>
          <div class="predict-toolbar">
            <div class="toolbar-group">
              <el-checkbox v-model="predictUseFallback" size="small">LSTM 不可用时回退</el-checkbox>
              <el-checkbox v-model="predictTriggerTrainAsync" size="small">预测后触发训练</el-checkbox>
            </div>
            <div class="toolbar-group">
              <el-button type="primary" size="small" :loading="predictAllLoading" @click="handlePredictAll">预测全部</el-button>
              <span class="hint-inline">点击行内「预测」可对单只股票预测；点击行首展开箭头查看拟合曲线</span>
            </div>
          </div>
          <el-table :data="predictTableWithExpand" size="small" stripe max-height="420" row-key="symbol" class="predict-by-stock-table predict-expand-table">
            <el-table-column type="expand">
              <template #default="{ row }">
                <div class="expand-row-wrap">
                  <div
                    v-for="y in [1, 2, 3]"
                    :key="y"
                    class="expand-year-group"
                  >
                    <div class="expand-block-title">{{ y }} 年模型 · 预测结果与拟合曲线</div>
                    <div class="expand-year-content">
                      <div class="expand-block expand-predict-block">
                        <div class="expand-block-label">预测结果</div>
                        <div v-if="(row.resultByYears || {})[y]" class="expand-predict-detail">
                          <span>方向：<span :class="(row.resultByYears[y].direction === 1) ? 'dir-up' : 'dir-down'">{{ row.resultByYears[y].direction_label ?? (row.resultByYears[y].direction === 1 ? '涨' : '跌') }}</span></span>
                          <span>预测涨跌幅：{{ formatPct(row.resultByYears[y].magnitude, 4) }}</span>
                          <span>上涨概率：{{ formatPct(row.resultByYears[y].prob_up) }}</span>
                          <span>下跌概率：{{ formatPct(row.resultByYears[y].prob_down) }}</span>
                          <span>来源：{{ row.resultByYears[y].source || 'lstm' }}</span>
                        </div>
                        <div v-else class="expand-predict-empty">暂无该年份预测，请点击「预测」执行</div>
                      </div>
                      <div class="expand-block expand-curve-block">
                        <div class="expand-block-label">拟合曲线</div>
                        <div
                          v-if="!isYearTrained(row.trained_years, y)"
                          class="plot-placeholder plot-placeholder-untrained"
                        >未训练</div>
                        <div
                          v-else
                          class="expand-curve-cell plot-wrap-clickable"
                          @click="openPlotPreview(row, y)"
                        >
                          <LstmFitChart
                            :symbol="row.symbol"
                            :years="y"
                            :refresh-key="plotImageRefreshKey"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="displayName" label="股票名称" min-width="100" show-overflow-tooltip />
            <el-table-column prop="symbol" label="代码" width="92" />
            <el-table-column label="方向" width="72">
              <template #default="{ row }">
                <span v-if="row.result" :class="row.result.direction === 1 ? 'dir-up' : 'dir-down'">
                  {{ row.result.direction_label ?? (row.result.direction === 1 ? '涨' : '跌') }}
                </span>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column label="预测涨跌幅" width="100">
              <template #default="{ row }">{{ row.result != null ? formatPct(row.result.magnitude, 4) : '—' }}</template>
            </el-table-column>
            <el-table-column label="上涨概率" width="88">
              <template #default="{ row }">{{ row.result != null ? formatPct(row.result.prob_up) : '—' }}</template>
            </el-table-column>
            <el-table-column label="下跌概率" width="88">
              <template #default="{ row }">{{ row.result != null ? formatPct(row.result.prob_down) : '—' }}</template>
            </el-table-column>
            <el-table-column label="来源" width="80">
              <template #default="{ row }">
                <el-tag v-if="row.result" size="small">{{ row.result.source || 'lstm' }}</el-tag>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="88" fixed="right">
              <template #default="{ row }">
                <el-button type="success" size="small" link :loading="predicting && predictSymbol === row.symbol" @click="handlePredictFor(row.symbol)">
                  预测
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <!-- 拟合曲线放大展示：同时展示 1/2/3 年 -->
        <el-dialog
          v-model="plotPreviewVisible"
          :title="plotPreviewRow ? `${plotPreviewRow.displayName || plotPreviewRow.symbol} - 拟合曲线（1/2/3 年）` : '拟合曲线'"
          width="92%"
          top="3vh"
          class="plot-preview-dialog"
          destroy-on-close
          align-center
          @close="plotPreviewRow = null"
        >
          <div v-if="plotPreviewRow" class="plot-preview-content plot-preview-multi">
            <div v-for="y in [1, 2, 3]" :key="y" class="plot-preview-year-block">
              <div class="plot-preview-year-label">{{ y }} 年模型</div>
              <div v-if="!isYearTrained(plotPreviewRow.trained_years, y)" class="plot-placeholder plot-placeholder-untrained">未训练</div>
              <LstmFitChart
                v-else
                :symbol="plotPreviewRow.symbol"
                :years="y"
                :refresh-key="plotImageRefreshKey"
                class="plot-preview-chart"
              />
            </div>
          </div>
        </el-dialog>
      </el-tab-pane>

      <!-- 训练流水与版本 -->
      <el-tab-pane label="训练流水与版本" name="runs">
        <el-card shadow="never" class="form-card">
          <div class="form-title">全部股票与训练时间</div>
          <el-button :loading="stocksStatusLoading" @click="loadStocksTrainingStatus">加载</el-button>
          <el-table v-loading="stocksStatusLoading" :data="stocksTrainingStatus" size="small" stripe max-height="400" class="stocks-status-table">
            <el-table-column prop="displayName" label="股票名称" min-width="120" show-overflow-tooltip />
            <el-table-column prop="symbol" label="股票代码" width="100" />
            <el-table-column label="最后一次训练时间" width="200">
              <template #default="{ row }">
                {{ row.last_train || '—' }}
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
              <el-button :loading="dedupeRunsLoading" @click="handleDedupeTrainingRuns">数据库去重</el-button>
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
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox, ElNotification } from 'element-plus'
import { useStockStore } from '@/stores/stock'
import LstmFitChart from '@/components/LstmFitChart.vue'
import {
  apiLstmTrain,
  apiLstmTrainStop,
  apiLstmTrainAll,
  apiLstmClearTraining,
  apiLstmLastPrediction,
  apiLstmLastPredictions,
  apiLstmPredict,
  apiLstmPredictAll,
  apiLstmPlotData,
  apiLstmStocksTrainingStatus,
  apiLstmTrainingRuns,
  apiLstmTrainingRunsDedupe,
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
  trainYears: 'all', // 'all' | 1 | 2 | 3，便于测试时只训单年
  do_cv_tune: true,
  do_shap: false,
  do_plot: true,
  fast_training: false,
  use_improved_training: true,
  reg_loss_type: 'full',
})
const trainTableRef = ref(null)
const selectedTrainRows = ref([])
const trainTableData = ref([])
const trainTableLoading = ref(false)
/** 当前正在训练中的股票 symbol 列表，用于多行同时显示 loading（点击新训练不会取消上一只的 loading） */
const trainingSymbols = ref([])
const trainSelectedLoading = ref(false)
const trainAllLoading = ref(false)
const trainAllProgress = ref({ current: 0, total: 0, symbol: '', displayName: '' })
/** 用户点击「停止训练」时置为 true，train-all 循环会在此轮结束后退出 */
const stopRequested = ref(false)
const stopTrainingLoading = ref(false)
const clearTrainingLoading = ref(false)
const trainingResultsList = ref([])

/** 训练表格数据：股票列表合并最近一次训练结果（按 symbol 取 trainingResultsList 中最新一条） */
const mergedTrainTableData = computed(() => {
  const list = trainTableData.value
  const results = trainingResultsList.value
  return list.map((row) => {
    const latest = results.find((r) => r.symbol === row.symbol)
    return {
      ...row,
      result: latest?.result ?? null,
      error: latest?.error ?? null,
    }
  })
})

// 预测（按股票分别展示，表格采用展开行：行下展示预测结果详情与拟合曲线）
const predictSymbol = ref('')
const predictUseFallback = ref(false)
const predictTriggerTrainAsync = ref(false)
const predicting = ref(false)
const predictAllLoading = ref(false)
const predictResultsByStock = ref([])
const plotImageRefreshKey = ref(0)
/** 预测表格数据：合并预测结果与训练状态（trained_years），供展开行内拟合曲线使用 */
const predictTableWithExpand = computed(() => {
  const predList = predictResultsByStock.value
  const trainRows = trainTableData.value
  return predList.map((p) => ({
    ...p,
    trained_years: trainRows.find((r) => r.symbol === p.symbol)?.trained_years ?? [],
  }))
})
const plotPreviewVisible = ref(false)
const plotPreviewRow = ref(null)

// 全部股票训练状态
const stocksStatusLoading = ref(false)
const stocksTrainingStatus = ref([])

// 训练流水
const runsSymbol = ref('')
const runsLimit = ref(50)
const runsLoading = ref(false)
const dedupeRunsLoading = ref(false)
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

/** 是否已具备 1/2/3 年全部模型（用于区分部分年份失败的情况） */
function hasAllThreeYearsTrained(trainedYears) {
  const ty = Array.isArray(trainedYears) ? trainedYears : []
  return ty.length === 3 && ty.includes(1) && ty.includes(2) && ty.includes(3)
}

/** 诊断结果中是否包含风险项（除「诊断摘要」外的项视为风险） */
function hasDiagnosticsRisk(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') return false
  const riskKeys = ['波动率问题', '模型容量', '特征问题', '过度拟合', '输入错误']
  return riskKeys.some((k) => diagnostics[k])
}

/** 诊断条目排序：诊断摘要排在最前 */
function diagnosticsOrdered(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') return []
  const entries = Object.entries(diagnostics)
  return entries.sort((a, b) => (a[0] === '诊断摘要' ? -1 : b[0] === '诊断摘要' ? 1 : 0))
}

/** 合并股票列表与训练时间、已训练年份，供训练页表格使用 */
function buildTrainTableData() {
  const files = stockStore.fileList || []
  const statusMap = {}
  for (const s of stocksTrainingStatus.value) {
    if (s.symbol) statusMap[s.symbol] = { last_train: s.last_train, trained_years: s.trained_years || [] }
  }
  trainTableData.value = files.map((f) => ({
    symbol: f.filename,
    displayName: f.displayName || f.filename || '',
    last_train: statusMap[f.filename]?.last_train ?? null,
    trained_years: statusMap[f.filename]?.trained_years ?? [],
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

/** 获取指定年数的推荐日期范围（训练时内部使用）。years: 1|2|3 */
async function getRecommendedRangeForYears(years) {
  const res = await apiLstmRecommendedRange({ years })
  return { start: res?.start || '', end: res?.end || '' }
}

function isTrainTimeToday(lastTrain) {
  if (!lastTrain || typeof lastTrain !== 'string') return false
  const datePart = lastTrain.trim().slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(datePart)) return false
  const today = new Date()
  const y = today.getFullYear()
  const m = String(today.getMonth() + 1).padStart(2, '0')
  const d = String(today.getDate()).padStart(2, '0')
  return datePart === `${y}-${m}-${d}`
}

async function handleTrainOne(symbol) {
  const sym = (symbol || '').trim()
  if (!sym) return
  const row = trainTableData.value.find((r) => r.symbol === sym)
  const trainYears = trainForm.value.trainYears
  const isAllYears = trainYears === 'all'
  if (isAllYears && hasAllThreeYearsTrained(row?.trained_years)) {
    ElMessage.info(`${row?.displayName || sym} 1/2/3 年模型已齐全，无需重复训练`)
    return
  }
  setError('')
  if (!trainingSymbols.value.includes(sym)) {
    trainingSymbols.value = [...trainingSymbols.value, sym]
  }
  const displayName = (row || {}).displayName || sym
  try {
    const res = await apiLstmTrain({
      symbol: sym,
      all_years: isAllYears,
      ...(isAllYears ? {} : { years: Number(trainYears) }),
      do_cv_tune: trainForm.value.do_cv_tune,
      do_shap: trainForm.value.do_shap,
      do_plot: trainForm.value.do_plot,
      fast_training: trainForm.value.fast_training,
      use_improved_training: trainForm.value.use_improved_training,
      reg_loss_type: trainForm.value.reg_loss_type,
    })
    if (res.error) {
      trainingResultsList.value = [{ symbol: sym, displayName, error: res.error }, ...trainingResultsList.value.filter((e) => e.symbol !== sym)]
      ElMessage.warning(res.error)
    } else {
      trainingResultsList.value = [{ symbol: sym, displayName, result: res, error: null }, ...trainingResultsList.value.filter((e) => e.symbol !== sym)]
      const riskKeys = ['波动率问题', '模型容量', '特征问题', '过度拟合', '输入错误']
      const riskItems = res.diagnostics && riskKeys.filter((k) => res.diagnostics[k]).map((k) => res.diagnostics[k])
      if (riskItems?.length) {
        ElNotification({
          type: 'warning',
          title: '预测平淡诊断',
          message: riskItems.join('；'),
          duration: 8000,
        })
      }
    }
    await loadStocksTrainingStatusForTrain(true)
    if (res?.metadata?.training_stopped) {
      ElMessage.warning('训练已按请求停止，已保存当前进度')
    } else {
      ElMessage.success(isAllYears ? sym + ' 1/2/3 年训练已全部完成' : `${sym} ${trainYears} 年模型训练完成`)
    }
  } catch (e) {
    const msg = e?.data?.error || e?.message || '训练失败'
    const isNetworkError = /socket hang up|fetch failed|network|aborted|ECONNRESET/i.test(String(msg))
    const hint = isNetworkError ? '（连接中断，训练可能仍在服务端执行，请稍后刷新「训练流水」或重新加载页面查看）' : ''
    trainingResultsList.value = [{ symbol: sym, displayName, error: msg + hint, result: null }, ...trainingResultsList.value.filter((e) => e.symbol !== sym)]
    setError(msg + hint)
    ElMessage.error(msg + hint)
  } finally {
    trainingSymbols.value = trainingSymbols.value.filter((s) => s !== sym)
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
  stopRequested.value = false
  for (const row of rows) {
    if (stopRequested.value) break
    await handleTrainOne(row.symbol)
  }
  trainSelectedLoading.value = false
  await loadStocksTrainingStatusForTrain(true)
  if (stopRequested.value) {
    ElMessage.warning('训练已停止')
  } else {
    ElMessage.success('选中项训练已全部完成')
  }
}

async function handleTrainAll() {
  if (!stockStore.fileList?.length) {
    ElMessage.warning('当前无股票数据，请先在股票列表或数据管理中添加')
    return
  }
  const trainYears = trainForm.value.trainYears
  const isAllYears = trainYears === 'all'
  const toTrain = isAllYears
    ? trainTableData.value.filter((r) => !hasAllThreeYearsTrained(r.trained_years))
    : trainTableData.value
  if (toTrain.length === 0) {
    ElMessage.info(isAllYears ? '全部股票 1/2/3 年模型已齐全，无需重复训练' : '当前无股票可训练')
    return
  }
  const total = trainTableData.value.length
  if (isAllYears && toTrain.length < total) {
    ElMessage.info(`共 ${total} 只股票，${total - toTrain.length} 只 1/2/3 年已齐全已跳过，将训练 ${toTrain.length} 只`)
  } else if (!isAllYears && toTrain.length > 0) {
    ElMessage.info(`将仅训练 ${trainYears} 年模型，共 ${toTrain.length} 只股票`)
  }
  setError('')
  trainAllLoading.value = true
  stopRequested.value = false
  trainAllProgress.value = { current: 0, total: toTrain.length, symbol: '', displayName: '' }
  try {
    let trained = 0
    for (let i = 0; i < toTrain.length; i++) {
      if (stopRequested.value) {
        ElMessage.warning('已停止训练')
        break
      }
      const row = toTrain[i]
      trainAllProgress.value = {
        current: i + 1,
        total: toTrain.length,
        symbol: row.symbol,
        displayName: row.displayName || row.symbol || '',
      }
      await handleTrainOne(row.symbol)
      trained++
      if (stopRequested.value) break
    }
    await loadStocksTrainingStatusForTrain(true)
    if (stopRequested.value) {
      ElMessage.warning(`训练已停止，已完成 ${trained}/${toTrain.length} 只`)
    } else {
      ElMessage.success(`一键训练完成：本次共训练 ${toTrain.length} 只`)
    }
  } catch (e) {
    const msg = e?.data?.error || e?.message || '一键训练失败'
    setError(msg)
    ElMessage.error(msg)
  } finally {
    trainAllLoading.value = false
    stopRequested.value = false
    trainAllProgress.value = { current: 0, total: 0, symbol: '', displayName: '' }
  }
}

async function handleStopTraining() {
  stopTrainingLoading.value = true
  try {
    if (trainAllLoading.value && trainAllProgress.value.symbol) {
      const res = await apiLstmTrainStop({ symbol: trainAllProgress.value.symbol })
      stopRequested.value = true
      ElMessage.success(res.message || '已请求停止')
    } else if (trainingSymbols.value.length) {
      stopRequested.value = true
      const sym = trainingSymbols.value[0]
      await apiLstmTrainStop({ symbol: sym })
      ElMessage.success('已请求停止当前训练')
    }
  } catch (e) {
    ElMessage.error(e?.data?.message || e?.message || '请求停止失败')
  } finally {
    stopTrainingLoading.value = false
  }
}

async function handleClearTraining() {
  const rows = selectedTrainRows.value
  if (!rows?.length) {
    ElMessage.warning('请先勾选要清理训练数据的股票')
    return
  }
  const symbols = rows.map((r) => r.symbol).filter(Boolean)
  try {
    await ElMessageBox.confirm(
      `确定要清理选中 ${symbols.length} 只股票的训练数据吗？清理后可重新训练。`,
      '清理训练数据',
      { type: 'warning', confirmButtonText: '确定清理', cancelButtonText: '取消' }
    )
  } catch {
    return
  }
  clearTrainingLoading.value = true
  try {
    const res = await apiLstmClearTraining({ symbols })
    if (res.error) {
      ElMessage.error(res.error)
      return
    }
    await loadStocksTrainingStatusForTrain(true)
    plotImageRefreshKey.value += 1
    ElMessage.success(res.message || `已清理 ${symbols.length} 只股票的训练数据`)
  } catch (e) {
    const msg = e?.data?.error || e?.message || '清理失败'
    ElMessage.error(msg)
  } finally {
    clearTrainingLoading.value = false
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
  try {
    let res = null
    for (const y of [1, 2, 3]) {
      const r = await apiLstmPredict(symbol, {
        years: y,
        use_fallback: predictUseFallback.value,
        trigger_train_async: predictTriggerTrainAsync.value,
      })
      if (r && (r.direction != null || r.error)) res = r
    }
    if (!res) {
      ElMessage.warning('预测无结果')
      predicting.value = false
      return
    }
    await loadPredictionsByStock()
    ElMessage.success('预测完成（1/2/3 年）')
    plotImageRefreshKey.value = Date.now()
  } catch (e) {
    const msg = e?.data?.error || e?.message || '预测失败'
    setError(msg)
    ElMessage.error(msg)
  } finally {
    predicting.value = false
  }
}

async function handlePredictAll() {
  if (!(stockStore.fileList?.length)) {
    ElMessage.warning('当前无股票数据，请先在股票列表或数据管理中添加')
    return
  }
  setError('')
  predictAllLoading.value = true
  try {
    const res = await apiLstmPredictAll({
      all_years: true,
      use_fallback: predictUseFallback.value,
      trigger_train_async: predictTriggerTrainAsync.value,
    })
    const successCount = res.success_count ?? 0
    if (successCount > 0) await loadPredictionsByStock()
    const failCount = res.fail_count ?? 0
    if (successCount > 0) {
      ElMessage.success(`预测全部完成（1/2/3 年）：成功 ${successCount} 只，失败 ${failCount} 只`)
      plotImageRefreshKey.value = Date.now()
    } else if (failCount > 0) {
      const firstError = (res.results || []).find((r) => r && !r.ok && r.error)?.error || '未知原因'
      setError(firstError)
      ElMessage.warning(`预测全部失败（${failCount} 只）。示例原因：${firstError}`)
    } else {
      ElMessage.info('暂无预测结果')
    }
  } catch (e) {
    const msg = e?.data?.error || e?.message || '预测全部失败'
    setError(msg)
    ElMessage.error(msg)
  } finally {
    predictAllLoading.value = false
  }
}

/** 该年份是否已训练（有模型） */
function isYearTrained(trained_years, y) {
  return Array.isArray(trained_years) && trained_years.includes(y)
}
function refreshPlotUrl() {
  plotImageRefreshKey.value = Date.now()
}

function openPlotPreview(row, _year) {
  if (!row) return
  plotPreviewRow.value = row
  plotPreviewVisible.value = true
}

/** 按股票加载预测结果列表：合并股票列表与各股票最近一次预测。 */
async function loadPredictionsByStock() {
  const files = stockStore.fileList || []
  const nameMap = {}
  for (const f of files) {
    nameMap[f.filename] = f.displayName || f.filename || ''
  }
  const baseList = files.map((f) => ({
    symbol: f.filename,
    displayName: nameMap[f.filename] || f.filename,
    result: null,
    resultByYears: { 1: null, 2: null, 3: null },
  }))
  try {
    const res = await apiLstmLastPredictions()
    const preds = res.predictions || []
    const bySymbolYears = {}
    for (const p of preds) {
      if (!p.symbol) continue
      const y = p.years === 1 || p.years === 2 || p.years === 3 ? p.years : 1
      if (!bySymbolYears[p.symbol]) bySymbolYears[p.symbol] = { 1: null, 2: null, 3: null }
      bySymbolYears[p.symbol][y] = p
    }
    predictResultsByStock.value = baseList.map((r) => {
      const byYears = bySymbolYears[r.symbol] || { 1: null, 2: null, 3: null }
      const primary = byYears[1]
      return {
        ...r,
        result: primary,
        resultByYears: byYears,
      }
    })
  } catch (_) {
    predictResultsByStock.value = baseList
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

/** 用每只股票最近一次训练流水填充训练结果列表（刷新后恢复表格中的版本/指标）。dedupe=1 时接口按股票去重返回最新一条。 */
async function loadLastTrainingResultsIntoList() {
  try {
    const res = await apiLstmTrainingRuns({ limit: 200, dedupe: 1 })
    const runs = res.runs || []
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

async function handleDedupeTrainingRuns() {
  dedupeRunsLoading.value = true
  try {
    const res = await apiLstmTrainingRunsDedupe()
    const n = res.deleted ?? 0
    ElMessage.success(res.message || `已删除 ${n} 条重复记录`)
    await loadTrainingRuns()
    await loadLastTrainingResultsIntoList()
  } catch (e) {
    const msg = e?.data?.error || e?.message || '去重失败'
    ElMessage.error(msg)
  } finally {
    dedupeRunsLoading.value = false
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
    if (sym) predictSymbol.value = sym
  },
  { immediate: true }
)

/** 刷新后/进入页面时：拉取训练状态（last_train、trained_years）与最近训练结果，恢复表格展示。 */
async function refreshTrainingStatus() {
  await stockStore.fetchList()
  await loadStocksTrainingStatusForTrain(true)
  await loadLastTrainingResultsIntoList()
}

onMounted(async () => {
  await refreshTrainingStatus()
  const firstSymbol = route.query.symbol || (stockStore.fileList?.length > 0 ? stockStore.fileList[0].filename : '')
  predictSymbol.value = firstSymbol
  accuracySymbol.value = firstSymbol
  triggersSymbol.value = firstSymbol
  await loadPredictionsByStock()
  refreshPlotUrl()
})

watch(activeTab, (tab) => {
  if (tab === 'train') refreshTrainingStatus()
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
.card-header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.card-header-title {
  font-weight: 600;
  color: #e8e8e8;
  margin-right: 10px;
}
.card-header-desc {
  font-size: 12px;
  color: #909399;
}
.section-card {
  margin-bottom: 20px;
}
.table-toolbar,
.predict-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 20px;
  margin-bottom: 12px;
}
.toolbar-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.toolbar-label {
  font-size: 13px;
  color: #909399;
  margin-right: 4px;
}
.hint-inline {
  font-size: 12px;
  color: #909399;
  margin-left: 4px;
}
.train-all-progress {
  font-size: 13px;
  color: #67c23a;
  margin-left: 8px;
}
.plot-section .plot-cards-grid {
  margin-top: 0;
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
.diagnostics-tooltip {
  max-width: 320px;
  font-size: 12px;
}
.diagnostics-tooltip-item {
  margin-bottom: 4px;
  line-height: 1.4;
}
.diagnostics-tooltip-item:last-child {
  margin-bottom: 0;
}
.config-label {
  font-size: 12px;
  color: #909399;
  margin-right: 4px;
}
.config-select {
  width: 200px;
}
.train-years-select {
  width: 140px;
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
.plot-cards-title {
  margin-top: 8px;
  margin-bottom: 4px;
}
.plot-cards-hint {
  margin-bottom: 8px;
}
.plot-refresh-btn {
  margin-bottom: 12px;
}
.plot-cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}
.plot-card {
  margin-bottom: 0;
}
.plot-card :deep(.el-card__header) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
}
.plot-card-title {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.plot-card-symbol {
  font-size: 12px;
  color: #909399;
  margin-left: 8px;
  flex-shrink: 0;
}
.plot-card :deep(.el-card__body) {
  padding: 10px 14px;
}
.plot-years-row {
  display: flex;
  gap: 10px;
  margin-top: 0;
  flex-wrap: wrap;
}
.plot-year-cell {
  flex: 1;
  min-width: 0;
  text-align: center;
}
.plot-year-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}
.plot-wrap {
  margin-top: 0;
}
.plot-img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid #2c3e50;
  display: block;
}
.plot-placeholder {
  margin-top: 0;
  padding: 20px 12px;
  text-align: center;
  color: #909399;
  font-size: 12px;
  background: #1e1e1e;
  border-radius: 8px;
  border: 1px dashed #3a3a3a;
}
.plot-placeholder-untrained {
  color: #606266;
  border-color: #2a2a2a;
}
.plot-section-hint {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}
.plot-wrap-clickable {
  cursor: pointer;
}
.plot-wrap-clickable:hover .plot-img {
  outline: 2px solid var(--el-color-primary);
  outline-offset: 2px;
}
.plot-preview-dialog :deep(.el-dialog) {
  max-width: 1280px;
  margin: 0 auto;
}
.plot-preview-dialog :deep(.el-dialog__body) {
  padding: 16px 24px 28px;
  max-height: 82vh;
  overflow-y: auto;
  overflow-x: hidden;
}
.plot-preview-content {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  min-height: 240px;
}
.plot-preview-content.plot-preview-multi {
  flex-direction: column;
  gap: 28px;
}
.plot-preview-year-block {
  width: 100%;
  padding: 16px 20px;
  background: var(--el-fill-color-lighter);
  border-radius: 12px;
  border: 1px solid var(--el-border-color-lighter);
}
.plot-preview-year-block:last-child {
  margin-bottom: 0;
}
.plot-preview-year-label {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
  padding-bottom: 8px;
  color: var(--el-text-color-primary);
  border-bottom: 1px solid var(--el-border-color-light);
}
.plot-preview-chart :deep(.lstm-fit-chart-wrap) {
  min-height: 720px;
}
.plot-preview-chart :deep(.lstm-fit-chart) {
  height: 720px;
}
.plot-preview-year-block .plot-placeholder-untrained {
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}
.plot-preview-img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid #2c3e50;
  display: block;
}
.predict-options {
  margin-bottom: 12px;
}
.predict-by-stock-table {
  margin-top: 8px;
}
.predict-expand-table :deep(.el-table__expanded-cell) {
  padding: 12px 16px;
  background: var(--el-fill-color-light);
}
.expand-row-wrap {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.expand-block {
  padding: 10px 12px;
  background: var(--el-bg-color);
  border-radius: 8px;
  border: 1px solid var(--el-border-color-lighter);
}
.expand-block-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-primary);
  margin-bottom: 8px;
}
.expand-predict-detail {
  display: flex;
  flex-wrap: wrap;
  gap: 16px 24px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.expand-predict-empty {
  font-size: 13px;
  color: #909399;
}
.expand-plot-years-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 4px;
}
.expand-year-group {
  margin-bottom: 16px;
}
.expand-year-group:last-child {
  margin-bottom: 0;
}
.expand-year-group .expand-block-title {
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--el-text-color-primary);
}
.expand-year-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: stretch;
}
.expand-predict-block {
  width: 100%;
  flex-shrink: 0;
}
.expand-curve-block {
  width: 100%;
  flex-shrink: 0;
}
.expand-block-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}
.expand-curve-cell {
  min-height: 160px;
  width: 100%;
}
.expand-curve-cell :deep(.lstm-fit-chart-wrap),
.expand-curve-cell :deep(.lstm-fit-chart) {
  min-height: 160px;
}
.expand-plot-year-cell {
  flex: 1;
  min-width: 180px;
  text-align: center;
}
.expand-plot-year-cell .plot-year-label {
  margin-bottom: 4px;
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
