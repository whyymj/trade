import { createRouter, createWebHistory } from 'vue-router'

/** 前端页面统一前缀，与后端 /api 等接口区分，避免路径冲突 */
export const APP_BASE = '/app'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/StockChart.vue'),
    meta: { title: '股票列表' },
  },
  {
    path: '/chart',
    name: 'Chart',
    component: () => import('@/views/ChartView.vue'),
    meta: { title: '股票曲线' },
  },
  {
    path: '/lstm',
    name: 'LSTM',
    component: () => import('@/views/LSTMView.vue'),
    meta: { title: 'LSTM 训练与预测' },
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
