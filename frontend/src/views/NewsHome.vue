<template>
  <div class="news-home">
    <div class="header">
      <h1>📰 财经新闻</h1>
      <div class="actions">
        <button @click="goToNewsList" class="btn btn-outline">
          📋 新闻列表
        </button>
        <button @click="goToAnalysis" class="btn btn-outline">
          📊 分析报告
        </button>
        <button @click="syncNews" :disabled="syncing" class="btn">
          {{ syncing ? '同步中...' : '🔄 同步' }}
        </button>
        <button @click="analyzeNews" :disabled="analyzing" class="btn btn-primary">
          {{ analyzing ? '分析中...' : '🤖 分析' }}
        </button>
      </div>
    </div>

    <!-- 市场情绪核心指标 -->
    <div class="section sentiment-card" v-if="sentiment">
      <div class="sentiment-header">
        <h2>📈 市场情绪</h2>
        <span class="sentiment-badge" :class="getSentimentType(sentiment)">
          {{ getSentimentLabel(sentiment) }}
        </span>
      </div>
      <div class="sentiment-grid">
        <div class="sentiment-item">
          <span class="label">成交额</span>
          <span class="value">{{ formatVolume(sentiment.volume) }}</span>
        </div>
        <div class="sentiment-item up">
          <span class="label">涨停</span>
          <span class="value">{{ sentiment.up_count }}</span>
        </div>
        <div class="sentiment-item down">
          <span class="label">跌停</span>
          <span class="value">{{ sentiment.down_count }}</span>
        </div>
        <div class="sentiment-item">
          <span class="label">换手率</span>
          <span class="value">{{ (sentiment.turnover_rate || 0).toFixed(2) }}%</span>
        </div>
      </div>
    </div>

    <!-- 快速入口 -->
    <div class="quick-links">
      <div class="quick-link card" @click="goToNewsList">
        <span class="link-icon">📋</span>
        <span class="link-text">新闻列表</span>
        <span class="link-desc">分类筛选 浏览全部</span>
      </div>
      <div class="quick-link card" @click="goToAnalysis">
        <span class="link-icon">📊</span>
        <span class="link-text">分析报告</span>
        <span class="link-desc">AI分析 投资建议</span>
      </div>
    </div>

    <!-- 最新新闻 -->
    <div class="section">
      <div class="section-header">
        <h2>📰 最新新闻</h2>
        <button class="btn-link" @click="goToNewsList">查看更多 →</button>
      </div>
      <div class="news-list">
        <div v-for="news in newsList" :key="news.id" class="news-item card" @click="goToDetail(news.id)">
          <div class="news-header">
            <span class="news-category" :class="getCategoryClass(news.category)">
              {{ news.category || '默认' }}
            </span>
            <span class="news-source">{{ news.source }}</span>
            <span class="news-time">{{ formatTime(news.published_at) }}</span>
          </div>
          <div class="news-title">{{ news.title }}</div>
        </div>
        <div v-if="newsList.length === 0" class="empty">暂无新闻，点击同步获取</div>
      </div>
    </div>

    <!-- 分析报告 -->
    <div class="section" v-if="analysis">
      <div class="section-header">
        <h2>📊 今日分析</h2>
        <button class="btn-link" @click="goToAnalysis">查看详情 →</button>
      </div>
      <div class="analysis-card card">
        <div class="impact-badge" :class="analysis.market_impact">
          {{ getImpactLabel(analysis.market_impact) }}
        </div>
        <div class="analysis-content">
          <h3>核心要点</h3>
          <pre class="summary">{{ analysis.summary }}</pre>
          <h3>投资建议</h3>
          <div class="advice">{{ analysis.investment_advice }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { getNewsList, analyzeNews as apiAnalyzeNews, getLatestAnalysis, syncNews as apiSyncNews } from '@/api/news'
import { getMarketSentiment } from '@/api/news'

export default {
  name: 'NewsHome',
  setup() {
    const router = useRouter()
    return { router }
  },
  data() {
    return {
      newsList: [],
      sentiment: null,
      analysis: null,
      syncing: false,
      analyzing: false,
    }
  },
  mounted() {
    this.loadNews()
    this.loadSentiment()
    this.loadAnalysis()
  },
  methods: {
    async loadNews() {
      try {
        const res = await getNewsList({ limit: 8 })
        if (res.code === 0) {
          this.newsList = res.data?.list || res.data || []
        }
      } catch (e) {
        console.error(e)
      }
    },
    async loadSentiment() {
      try {
        const res = await getMarketSentiment(1)
        if (res.code === 0 && res.data) {
          this.sentiment = res.data
        }
      } catch (e) {
        console.error(e)
      }
    },
    async loadAnalysis() {
      try {
        const res = await getLatestAnalysis()
        if (res.code === 0 && res.data) {
          this.analysis = res.data
        }
      } catch (e) {
        console.error(e)
      }
    },
    async syncNews() {
      this.syncing = true
      try {
        const res = await apiSyncNews()
        if (res.code === 0) {
          this.$message.success('同步成功')
          this.loadNews()
        } else {
          this.$message.warning(res.data?.message || res.message || '频率限制')
        }
      } catch (e) {
        this.$message.error('同步失败')
      } finally {
        this.syncing = false
      }
    },
    async analyzeNews() {
      this.analyzing = true
      try {
        const res = await apiAnalyzeNews({ days: 1 })
        if (res.code === 0) {
          this.analysis = res.data
          this.$message.success('分析完成')
        }
      } catch (e) {
        this.$message.error('分析失败')
      } finally {
        this.analyzing = false
      }
    },
    formatVolume(vol) {
      if (!vol) return '-'
      if (vol >= 100000000) {
        return (vol / 100000000).toFixed(2) + '万亿'
      }
      return (vol / 10000).toFixed(1) + '亿'
    },
    formatTime(time) {
      if (!time) return ''
      return time.substring(5, 16)
    },
    goToDetail(newsId) {
      this.router.push('/news/' + encodeURIComponent(newsId))
    },
    goToNewsList() {
      this.router.push('/news/list')
    },
    goToAnalysis() {
      this.router.push('/news/analysis')
    },
    getImpactLabel(impact) {
      const map = { bullish: '看涨', bearish: '看跌', neutral: '中性' }
      return map[impact] || '中性'
    },
    getSentimentLabel(sentiment) {
      if (!sentiment) return '中性'
      const up = sentiment.up_count || 0
      const down = sentiment.down_count || 0
      if (up > down * 2) return '偏热'
      if (down > up * 2) return '偏冷'
      return '中性'
    },
    getSentimentType(sentiment) {
      if (!sentiment) return 'neutral'
      const up = sentiment.up_count || 0
      const down = sentiment.down_count || 0
      if (up > down * 2) return 'hot'
      if (down > up * 2) return 'cold'
      return 'neutral'
    },
    getCategoryClass(category) {
      const map = {
        '宏观': 'macro',
        '政策': 'policy',
        '行业': 'industry',
        '全球': 'global',
      }
      return map[category] || ''
    },
  }
}
</script>

<style scoped>
.news-home {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header h1 {
  margin: 0;
  color: var(--primary);
  font-size: 28px;
}

.actions {
  display: flex;
  gap: 10px;
}

.btn {
  padding: 10px 18px;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.btn-primary {
  background: var(--primary);
  color: white;
}

.btn-outline {
  background: transparent;
  border: 2px solid var(--primary);
  color: var(--primary);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.section {
  background: var(--card-bg, white);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: var(--shadow);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section h2 {
  margin: 0;
  font-size: 18px;
  color: var(--text-primary);
}

.btn-link {
  background: none;
  border: none;
  color: var(--primary);
  cursor: pointer;
  font-size: 14px;
}

.btn-link:hover {
  text-decoration: underline;
}

/* 快速入口 */
.quick-links {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.quick-link {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
}

.quick-link:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover);
}

.link-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.link-text {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.link-desc {
  font-size: 13px;
  color: var(--text-muted);
}

/* 市场情绪 */
.sentiment-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.sentiment-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.sentiment-header h2 {
  color: white;
}

.sentiment-badge {
  padding: 4px 14px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.2);
}

.sentiment-badge.hot {
  background: rgba(126, 217, 87, 0.4);
  color: #7ed957;
}

.sentiment-badge.cold {
  background: rgba(255, 107, 107, 0.4);
  color: #ff6b6b;
}

.sentiment-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.sentiment-item {
  text-align: center;
}

.sentiment-item .label {
  display: block;
  font-size: 13px;
  opacity: 0.85;
  margin-bottom: 6px;
}

.sentiment-item .value {
  font-size: 26px;
  font-weight: 700;
}

.sentiment-item.up .value {
  color: #7ed957;
}

.sentiment-item.down .value {
  color: #ff6b6b;
}

/* 新闻列表 */
.news-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.news-item {
  padding: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.news-item:hover {
  transform: translateX(4px);
  box-shadow: var(--shadow-hover);
}

.news-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.news-category {
  padding: 4px 12px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
  background: #e0e0e0;
  color: #666;
}

.news-category.macro { background: #e3f2fd; color: #1976d2; }
.news-category.policy { background: #f3e5f5; color: #7b1fa2; }
.news-category.industry { background: #e8f5e9; color: #388e3c; }
.news-category.global { background: #fff3e0; color: #f57c00; }

.news-source {
  font-size: 13px;
  color: var(--text-muted);
}

.news-time {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-muted);
}

.news-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-primary);
  line-height: 1.4;
}

/* 分析卡片 */
.analysis-card {
  background: var(--bg-primary);
  border-radius: 14px;
  padding: 20px;
}

.impact-badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: 18px;
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 16px;
}

.impact-badge.bullish { background: #e8f5e9; color: #388e3c; }
.impact-badge.bearish { background: #ffebee; color: #c62828; }
.impact-badge.neutral { background: #e0e0e0; color: #616161; }

.analysis-content h3 {
  font-size: 14px;
  margin: 16px 0 10px;
  color: #666;
}

.summary {
  background: white;
  padding: 14px;
  border-radius: 10px;
  font-size: 13px;
  white-space: pre-wrap;
  font-family: inherit;
  line-height: 1.6;
  margin: 0;
}

.advice {
  background: white;
  padding: 14px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--primary);
}

.empty {
  text-align: center;
  color: var(--text-muted);
  padding: 30px;
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    gap: 16px;
    align-items: flex-start;
  }
  
  .actions {
    flex-wrap: wrap;
  }
  
  .quick-links {
    grid-template-columns: 1fr;
  }
  
  .sentiment-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
