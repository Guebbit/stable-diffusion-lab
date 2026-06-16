import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type { ArtifactEntry } from '../types'

export const useHistoryStore = defineStore('history', () => {
  // All persisted artifacts loaded from the backend (newest first)
  const images = ref<ArtifactEntry[]>([])
  // True while fetching or deleting
  const isLoading = ref(false)

  /** Fetch the full artifact gallery from the backend. */
  function fetchHistory() {
    isLoading.value = true
    return diffusionApi.getArtifacts({ limit: 100 })
      .then((data) => {
        images.value = data.items
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to load generation history')
      })
      .finally(() => {
        isLoading.value = false
      })
  }

  /** Delete one artifact by ID and remove it from the local list immediately. */
  function deleteEntry(imageId: string) {
    return diffusionApi.deleteArtifact(imageId)
      .then(() => {
        images.value = images.value.filter((img) => img.id !== imageId)
        useNotificationStore().push('success', 'History entry deleted')
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to delete history entry')
      })
  }

  /** Delete all artifacts (files + DB records). */
  function clearHistory() {
    return diffusionApi.deleteAllArtifacts()
      .then(() => {
        images.value = []
        useNotificationStore().push('success', 'History cleared')
      })
      .catch(() => {
        useNotificationStore().push('error', 'Failed to clear history')
      })
  }

  /** Initialize store by fetching history from the backend. */
  function init() {
    fetchHistory()
  }

  return {
    images,
    isLoading,
    fetchHistory,
    deleteEntry,
    clearHistory,
    init,
  }
})
