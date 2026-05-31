import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type { GeneratedImage } from '../types'

export const useHistoryStore = defineStore('history', () => {
  // All persisted images loaded from the backend (newest first)
  const images = ref<GeneratedImage[]>([])
  // True while fetching or deleting
  const isLoading = ref(false)

  /** Fetch the full history from the backend and replace the local list. */
  function fetchHistory() {
    isLoading.value = true
    return diffusionApi.getHistory()
      .then((data) => {
        images.value = data
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to load generation history')
      })
      .finally(() => {
        isLoading.value = false
      })
  }

  /** Delete one image by ID and remove it from the local list immediately. */
  function deleteEntry(imageId: string) {
    return diffusionApi.deleteHistoryEntry(imageId)
      .then(() => {
        // Remove from local state without a full refetch for a snappy UI
        images.value = images.value.filter((img) => img.id !== imageId)
        useNotificationStore().push('success', 'History entry deleted')
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to delete history entry')
      })
  }

  /** Wipe all history entries on the backend and clear local state. */
  function clearAll() {
    return diffusionApi.clearAllHistory()
      .then(() => {
        images.value = []
        useNotificationStore().push('success', 'All history cleared')
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to clear history')
      })
  }

  return {
    images,
    isLoading,
    fetchHistory,
    deleteEntry,
    clearAll,
  }
})
