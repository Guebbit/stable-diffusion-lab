/**
 * Model registry store — manages the centralized model catalog.
 *
 * Responsibilities:
 *  - Fetch all registered models (with download status)
 *  - Add/remove models from the registry
 *  - Trigger model downloads (async job-based)
 *  - Track download progress via SSE (no polling)
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type { ModelRegistryEntry, ModelRegistryAddRequest } from '../types'

export const useModelsStore = defineStore('models', () => {
  const registry = ref<ModelRegistryEntry[]>([])
  const isLoading = ref(false)
  const isDownloading = ref<Set<string>>(new Set())
  const sseConnected = ref(false)

  // Per-model progress data received from SSE job.progress events
  const downloadProgress = ref<Map<string, { pct: number; file: string; fileIndex: number; totalFiles: number }>>(new Map())

  // SSE connection — one per store instance, managed by the view
  let _sseSource: EventSource | null = null
  let _reconnectTimer: ReturnType<typeof setTimeout> | null = null

  const huggingfaceModels = computed(() =>
    registry.value.filter(m => m.source === 'huggingface' && m.status === 'downloaded'),
  )
  const civitaiModels = computed(() =>
    registry.value.filter(m => m.source === 'civitai' && m.status === 'downloaded'),
  )
  const downloadedModels = computed(() =>
    registry.value.filter(m => m.status === 'downloaded'),
  )
  const allTags = computed(() => {
    const tagSet = new Set<string>()
    for (const m of registry.value) {
      for (const t of m.tags ?? []) tagSet.add(t)
    }
    return [...tagSet].sort()
  })

  /** Fetch the full model registry from the server. */
  function fetchRegistry() {
    isLoading.value = true
    return diffusionApi.getModels()
      .then((models) => {
        registry.value = models
        // If any model is in-progress (e.g. after a page refresh) ensure SSE is active
        const hasActiveDownload = models.some(
          m => m.status === 'downloading' || m.status === 'download_paused',
        )
        if (hasActiveDownload) {
          connectSSE()
          for (const m of models) {
            if (m.status === 'downloading' || m.status === 'download_paused') {
              isDownloading.value = new Set(isDownloading.value)
              isDownloading.value.add(m.model_id)
            }
          }
        }
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

  /** Delete downloaded files for a model and reset its status to not_downloaded. */
  function deleteModelFiles(modelId: string) {
    const notif = useNotificationStore()
    return diffusionApi.purgeModelFiles(modelId)
      .then(() => {
        const entry = registry.value.find(m => m.model_id === modelId)
        if (entry) entry.status = 'not_downloaded'
        notif.push('success', `Files deleted for "${modelId}"`)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to delete model files'
        notif.push('error', msg)
      })
  }

  /** Remove a model from the registry (database + disk files). Permanent. */
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

  /** Remove ALL models from the registry and disk. Permanent. */
  function removeAllModels() {
    const notif = useNotificationStore()
    return diffusionApi.removeAllModels()
      .then(() => {
        registry.value = []
        notif.push('success', 'All models removed from registry')
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to remove all models'
        notif.push('error', msg)
      })
  }

  /** Purge downloaded files for all models, resetting their status to not_downloaded. */
  function purgeAllModelFiles() {
    const notif = useNotificationStore()
    return diffusionApi.purgeAllModelFiles()
      .then(() => {
        registry.value = registry.value.map(m => ({ ...m, status: 'not_downloaded' as const }))
        notif.push('success', 'All model files purged')
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to purge all model files'
        notif.push('error', msg)
      })
  }

  /** Trigger a background download for a model. Progress arrives via SSE. */
  function downloadModel(modelId: string) {
    const notif = useNotificationStore()
    isDownloading.value = new Set(isDownloading.value)
    isDownloading.value.add(modelId)
    connectSSE()

    notif.push('info', `Queuing download for "${modelId}"…`)
    return diffusionApi.downloadModel(modelId)
      .then((res) => {
        notif.push('info', res.message)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to start download'
        notif.push('error', msg)
        isDownloading.value = new Set(isDownloading.value)
        isDownloading.value.delete(modelId)
      })
  }

  /** True while a model is being downloaded (current session OR server status). */
  function isModelDownloading(modelId: string): boolean {
    if (isDownloading.value.has(modelId)) return true
    const model = registry.value.find(m => m.model_id === modelId)
    return model?.status === 'downloading' || model?.status === 'download_paused'
  }

  // ── SSE connection ────────────────────────────────────────────────────────

  /** Open (or reuse) the SSE connection to the observability stream. */
  function connectSSE() {
    if (_sseSource && _sseSource.readyState !== EventSource.CLOSED) return
    if (_reconnectTimer !== null) {
      clearTimeout(_reconnectTimer)
      _reconnectTimer = null
    }

    _sseSource = new EventSource(
      `${window.location.origin}/sse/observability?subscribe=job,model`,
    )

    _sseSource.onopen = () => {
      sseConnected.value = true
    }

    _sseSource.onmessage = (evt: MessageEvent) => {
      try {
        _handleSSEEvent(JSON.parse(evt.data as string))
      } catch {
        // malformed frame, ignore
      }
    }

    _sseSource.onerror = () => {
      sseConnected.value = false
      _sseSource?.close()
      _sseSource = null
      // Reconnect only if downloads are still pending
      if (isDownloading.value.size > 0) {
        _reconnectTimer = setTimeout(connectSSE, 5000)
      }
    }
  }

  /** Close the SSE connection. Call from the view's onUnmounted. */
  function disconnectSSE() {
    if (_reconnectTimer !== null) {
      clearTimeout(_reconnectTimer)
      _reconnectTimer = null
    }
    _sseSource?.close()
    _sseSource = null
    sseConnected.value = false
  }

  /** Initialize store by fetching the model registry from the backend. */
  function init() {
    fetchRegistry()
  }

  function _handleSSEEvent(event: { event_type?: string; type?: string; payload?: Record<string, unknown> }) {
    const eventType = event.event_type ?? event.type
    const payload = event.payload ?? {}

    if (eventType === 'job.progress') {
      const modelId = payload.model_id as string | undefined
      if (!modelId) return
      const pct = (payload.progress_percent as number | undefined) ?? 0
      const file = (payload.file as string | undefined) ?? ''
      const fileIndex = (payload.file_index as number | undefined) ?? 0
      const totalFiles = (payload.total_files as number | undefined) ?? 0
      const next = new Map(downloadProgress.value)
      next.set(modelId, { pct, file, fileIndex, totalFiles })
      downloadProgress.value = next
      registry.value = registry.value.map(m =>
        m.model_id === modelId ? { ...m, status: 'downloading' } : m,
      )
    }

    else if (eventType === 'model.downloaded') {
      const modelId = payload.model_id as string | undefined
      if (!modelId) return
      const name = registry.value.find(m => m.model_id === modelId)?.name ?? modelId
      isDownloading.value = new Set(isDownloading.value)
      isDownloading.value.delete(modelId)
      const next = new Map(downloadProgress.value)
      next.delete(modelId)
      downloadProgress.value = next
      useNotificationStore().push('success', `Model "${name}" downloaded successfully!`)
      fetchRegistry()
    }

    else if (eventType === 'model.download_failed') {
      const modelId = payload.model_id as string | undefined
      if (!modelId) return
      const name = registry.value.find(m => m.model_id === modelId)?.name ?? modelId
      const errorMsg = (payload.error as string | undefined) ?? 'Download failed'
      isDownloading.value = new Set(isDownloading.value)
      isDownloading.value.delete(modelId)
      const next = new Map(downloadProgress.value)
      next.delete(modelId)
      downloadProgress.value = next
      useNotificationStore().push('error', `"${name}": ${errorMsg}`)
      fetchRegistry()
    }
  }

  return {
    registry,
    downloadedModels,
    isLoading,
    isDownloading,
    sseConnected,
    downloadProgress,
    huggingfaceModels,
    civitaiModels,
    allTags,
    fetchRegistry,
    addModel,
    deleteModelFiles,
    removeModel,
    removeAllModels,
    purgeAllModelFiles,
    downloadModel,
    isModelDownloading,
    connectSSE,
    disconnectSSE,
    init,
  }
})
