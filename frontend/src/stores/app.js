import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const loading = ref(false)
  const errorMessage = ref('')
  const toastMessage = ref('')
  const toastVisible = ref(false)

  function setLoading(v) {
    loading.value = !!v
  }

  function setError(msg) {
    errorMessage.value = msg || ''
  }

  function showToast(msg, duration = 2500) {
    toastMessage.value = msg || ''
    toastVisible.value = true
    if (duration > 0) {
      setTimeout(() => {
        toastVisible.value = false
      }, duration)
    }
  }

  return {
    loading,
    errorMessage,
    toastMessage,
    toastVisible,
    setLoading,
    setError,
    showToast,
  }
})
