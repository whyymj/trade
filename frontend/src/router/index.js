import { createRouter, createWebHistory } from 'vue-router'

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
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, _from, next) => {
  if (to.meta?.title) {
    document.title = to.meta.title + ' - 股票数据'
  }
  next()
})

export default router
