import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/StockChart.vue'),
    meta: { title: '股票数据曲线' },
  },
  {
    path: '/data-manage',
    name: 'DataManage',
    component: () => import('@/views/DataManage.vue'),
    meta: { title: '数据管理' },
  },
  {
    path: '/data-range',
    name: 'DataRange',
    component: () => import('@/views/DataRange.vue'),
    meta: { title: '按日期范围查询' },
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
