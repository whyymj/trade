import { defineStore } from 'pinia'
import { ref } from 'vue'
import { apiList, apiData, apiAddStock, apiUpdateAll } from '@/api/stock'

export const useStockStore = defineStore('stock', () => {
  const fileList = ref([])
  const selectedFilename = ref('')
  const chartData = ref(null)

  async function fetchList() {
    const res = await apiList()
    fileList.value = res.files || []
    return fileList.value
  }

  async function fetchData(filename) {
    if (!filename) return null
    const decoded = decodeURIComponent(filename)
    const data = await apiData(decoded)
    chartData.value = data
    return data
  }

  function setSelected(filename) {
    selectedFilename.value = filename || ''
  }

  function clearChartData() {
    chartData.value = null
  }

  async function addStock(code) {
    return apiAddStock(code)
  }

  async function updateAll(options = { fromLastUpdate: true }) {
    return apiUpdateAll(options)
  }

  return {
    fileList,
    selectedFilename,
    chartData,
    fetchList,
    fetchData,
    setSelected,
    clearChartData,
    addStock,
    updateAll,
  }
})
