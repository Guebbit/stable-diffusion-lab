/**
 * Model registry store — manages the centralized model catalog.
 *
 * Responsibilities:
 *  - Fetch all registered models (with download status)
 *  - Fetch only downloaded models (for generation form selects)
 *  - Add/remove models from the registry
 *  - Trigger model downloads
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type { ModelRegistryEntry, ModelRegistryAddRequest, ModelSource } from '../types'

// Polling config for download status checks
const POLL_INTERVAL_MS = 10_000  // Check every 10 seconds
const MAX_POLL_ATTEMPTS = 60     // Give up after ~10 minutes

export const useModelsStore = defineStore('models', () => {
  // All registered models (full catalog with download status)
  const registry = ref<ModelRegistryEntry[]>([])
  // Only models that are downloaded and ready for generation
  const downloadedModels = ref<ModelRegistryEntry[]>([])
  // Loading states
  const isLoading = ref(false)
  const isDownloading = ref<Set<string>>(new Set())

  // Computed: models grouped by source for convenience
  const huggingfaceModels = computed(() =>
    downloadedModels.value.filter(m => m.source === 'huggingface'),
  )
  const civitaiModels = computed(() =>
    downloadedModels.value.filter(m => m.source === 'civitai'),
  )

  /** Fetch the full model registry (used by the Models management page). */
  function fetchRegistry() {
    isLoading.value = true
    return diffusionApi.getModels()
      .then((models) => {
        registry.value = models
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to fetch model registry'
        useNotificationStore().push('error', msg)
      })
      .finally(() => {
        isLoading.value = false
      })
  }

  /** Fetch only downloaded models (used by generation form selects). */
  function fetchDownloadedModels() {
    return diffusionApi.getDownloadedModels()
      .then((models) => {
        downloadedModels.value = models
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to fetch downloaded models'
        useNotificationStore().push('error', msg)
      })
  }

  /** Register a new model in the catalog. */
  function addModel(payload: ModelRegistryAddRequest) {
    const notif = useNotificationStore()
    return diffusionApi.addModel(payload)
      .then((entry) => {
        registry.value = [...registry.value, entry]
        notif.push('success', `Model "${entry.name}" added to registry`)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to add model'
        notif.push('error', msg)
        throw err
      })
  }

  /** Remove a model from the registry. */
  function removeModel(modelId: string, source: ModelSource) {
    const notif = useNotificationStore()
    return diffusionApi.removeModel(modelId, source)
      .then(() => {
        registry.value = registry.value.filter(
          m => !(m.id === modelId && m.source === source),
        )
        downloadedModels.value = downloadedModels.value.filter(
          m => !(m.id === modelId && m.source === source),
        )
        notif.push('success', `Model "${modelId}" removed from registry`)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to remove model'
        notif.push('error', msg)
      })
  }

  /** Trigger a background download for a model. */
  function downloadModel(modelId: string, source: ModelSource) {
    const notif = useNotificationStore()
    const key = `${source}:${modelId}`
    isDownloading.value.add(key)

    notif.push('info', `Downloading model "${modelId}"…`)
    return diffusionApi.downloadModel(modelId, source)
      .then((res) => {
        notif.push('info', res.detail)
        // Poll for completion after a delay
        _pollDownloadStatus(modelId, source)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to start download'
        notif.push('error', msg)
        isDownloading.value.delete(key)
      })
  }

  /** Check if a specific model is currently being downloaded. */
  function isModelDownloading(modelId: string, source: ModelSource): boolean {
    return isDownloading.value.has(`${source}:${modelId}`)
  }

  /**
   * Poll the registry periodically to detect when a download finishes.
   * Stops after the model shows as downloaded or after MAX_POLL_ATTEMPTS.
   */
  function _pollDownloadStatus(modelId: string, source: ModelSource) {
    const key = `${source}:${modelId}`
    let attempts = 0

    const interval = setInterval(() => {
      attempts++
      diffusionApi.getModels()
        .then((models) => {
          registry.value = models
          const model = models.find(m => m.id === modelId && m.source === source)
          if (model?.downloaded) {
            clearInterval(interval)
            isDownloading.value.delete(key)
            useNotificationStore().push('success', `Model "${model.name}" downloaded successfully!`)
            // Refresh the downloaded models list
            fetchDownloadedModels()
          } else if (attempts >= MAX_POLL_ATTEMPTS) {
            clearInterval(interval)
            isDownloading.value.delete(key)
            useNotificationStore().push('warning', `Download polling timed out for "${modelId}"`)
          }
        })
        .catch(() => {
          // Silently retry on network errors during polling
        })
    }, POLL_INTERVAL_MS)
  }

  return {
    registry,
    downloadedModels,
    isLoading,
    isDownloading,
    huggingfaceModels,
    civitaiModels,
    fetchRegistry,
    fetchDownloadedModels,
    addModel,
    removeModel,
    downloadModel,
    isModelDownloading,
  }
})
