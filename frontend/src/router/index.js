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
