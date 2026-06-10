/**
 * Model registry store — manages the centralized model catalog.
 *
 * Responsibilities:
 *  - Fetch all registered models (with download status)
 *  - Add/remove models from the registry
 *  - Trigger model downloads (async job-based)
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type { ModelRegistryEntry, ModelRegistryAddRequest } from '../types'

// Polling config for download status checks
const POLL_INTERVAL_MS = 5_000   // Check every 5 seconds
const MAX_POLL_ATTEMPTS = 120    // Give up after ~10 minutes

export const useModelsStore = defineStore('models', () => {
  // All registered models (full catalog with download status)
  const registry = ref<ModelRegistryEntry[]>([])
  // Loading states
  const isLoading = ref(false)
  const isDownloading = ref<Set<string>>(new Set())

  // Computed: models grouped by source for convenience
  const huggingfaceModels = computed(() =>
    registry.value.filter(m => m.source === 'huggingface' && m.status === 'downloaded'),
  )
  const civitaiModels = computed(() =>
    registry.value.filter(m => m.source === 'civitai' && m.status === 'downloaded'),
  )
  // Downloaded models (ready for generation)
  const downloadedModels = computed(() =>
    registry.value.filter(m => m.status === 'downloaded'),
  )

  // Computed: models filtered by capability (for form dropdowns)
  const captioningModels = computed(() =>
    registry.value.filter(
      m => m.status === 'downloaded' && m.capabilities?.includes('captioning'),
    ),
  )
  const txt2imgModels = computed(() =>
    registry.value.filter(
      m => m.status === 'downloaded' && m.capabilities?.includes('txt2img'),
    ),
  )
  const img2imgModels = computed(() =>
    registry.value.filter(
      m => m.status === 'downloaded' && m.capabilities?.includes('img2img'),
    ),
  )

  /** Fetch the full model registry. */
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
  function removeModel(modelId: string) {
    const notif = useNotificationStore()
    return diffusionApi.removeModel(modelId)
      .then(() => {
        registry.value = registry.value.filter(m => m.model_id !== modelId)
        notif.push('success', `Model "${modelId}" removed from registry`)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to remove model'
        notif.push('error', msg)
      })
  }

  /** Trigger a background download for a model. */
  function downloadModel(modelId: string) {
    const notif = useNotificationStore()
    isDownloading.value = new Set(isDownloading.value)
    isDownloading.value.add(modelId)

    notif.push('info', `Downloading model "${modelId}"…`)
    return diffusionApi.downloadModel(modelId)
      .then((res) => {
        notif.push('info', res.message)
        // Poll registry for completion
        _pollDownloadStatus(modelId)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to start download'
        notif.push('error', msg)
        isDownloading.value.delete(modelId)
      })
  }

  /** Check if a specific model is currently being downloaded. */
  function isModelDownloading(modelId: string): boolean {
    return isDownloading.value.has(modelId)
  }

  /**
   * Poll the model registry until the model shows as downloaded or fails.
   */
  function _pollDownloadStatus(modelId: string) {
    let attempts = 0

    const interval = setInterval(() => {
      attempts++

      diffusionApi.getModels()
        .then((models) => {
          registry.value = models
          const model = models.find(m => m.model_id === modelId)
          
          if (!model) {
            // Model not found in registry, continue polling
            return
          }
          
          if (model.status === 'downloaded' || model.status === 'loaded' || model.status === 'loaded') {
            clearInterval(interval)
            isDownloading.value.delete(modelId)
            useNotificationStore().push('success', `Model "${model.name}" downloaded successfully!`)
          } else if (model.status === 'downloading' || model.status === 'download_paused') {
            // Model is actively downloading - show progress
            const progress = model.download_progress ?? 0
            useNotificationStore().push(
              'info',
              `Downloading "${model.name}": ${progress.toFixed(1)}%`,
            )
          } else if (model.status === 'error') {
            clearInterval(interval)
            isDownloading.value.delete(modelId)
            useNotificationStore().push('error', `Model "${model.name}" download failed`)
          } else if (attempts >= MAX_POLL_ATTEMPTS) {
            clearInterval(interval)
            isDownloading.value.delete(modelId)
            useNotificationStore().push('warning', `Download polling timed out for "${model.name}"`)
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
    captioningModels,
    txt2imgModels,
    img2imgModels,
    fetchRegistry,
    addModel,
    removeModel,
    downloadModel,
    isModelDownloading,
  }
})
