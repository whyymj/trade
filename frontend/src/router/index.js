import { createRouter, createWebHistory } from 'vue-router'
import '@/styles/cute.css'

/** 前端页面统一前缀，与后端 /api 等接口区分，避免路径冲突 */
export const APP_BASE = '/app'

const routes = [
  {
    path: '/',
    name: 'FundHome',
    component: () => import('@/views/FundHome.vue'),
    meta: { title: '基金列表' },
  },
  {
    path: '/fund/:code',
    name: 'FundDetail',
    component: () => import('@/views/FundDetail.vue'),
    meta: { title: '基金详情' },
  },
  {
    path: '/predict',
    name: 'FundPredict',
    component: () => import('@/views/FundPredict.vue'),
    meta: { title: '预测中心' },
  },
  {
    path: '/fund-industry',
    name: 'FundIndustry',
    component: () => import('@/views/FundIndustry.vue'),
    meta: { title: '基金行业分析' },
  },
  {
    path: '/news',
    name: 'NewsHome',
    component: () => import('@/views/NewsHome.vue'),
    meta: { title: '财经新闻' },
  },
  {
    path: '/news/list',
    name: 'NewsList',
    component: () => import('@/views/NewsList.vue'),
    meta: { title: '新闻列表' },
  },
  {
    path: '/news/analysis',
    name: 'NewsAnalysis',
    component: () => import('@/views/NewsAnalysis.vue'),
    meta: { title: '新闻分析' },
  },
  {
    path: '/news/classification',
    name: 'NewsClassification',
    component: () => import('@/views/NewsClassification.vue'),
    meta: { title: '新闻行业分类' },
  },
  {
    path: '/news/:id',
    name: 'NewsDetail',
    component: () => import('@/views/NewsDetail.vue'),
    meta: { title: '新闻详情' },
  },
  {
    path: '/market',
    name: 'MarketHome',
    component: () => import('@/views/MarketHome.vue'),
    meta: { title: '市场数据' },
  },
  {
    path: '/fund-news',
    name: 'FundNewsAssociation',
    component: () => import('@/views/FundNewsAssociation.vue'),
    meta: { title: '基金新闻关联' },
  },
]

const router = createRouter({
  history: createWebHistory(APP_BASE),
  routes,
})

router.beforeEach((to, _from, next) => {
  if (to.meta?.title) {
    document.title = to.meta.title + ' - 股票数据'
  }
  next()
})

export default router
