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
  const captioningModels = computed(() =>
    registry.value.filter(m => m.status === 'downloaded' && m.capabilities?.includes('captioning')),
  )
  const txt2imgModels = computed(() =>
    registry.value.filter(m => m.status === 'downloaded' && m.capabilities?.includes('txt2img')),
  )
  const img2imgModels = computed(() =>
    registry.value.filter(m => m.status === 'downloaded' && m.capabilities?.includes('img2img')),
  )

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

  function _handleSSEEvent(event: { event_type?: string; type?: string; payload?: Record<string, unknown> }) {
    const eventType = event.event_type ?? event.type
    const payload = event.payload ?? {}

    if (eventType === 'job.progress') {
      const modelId = payload.model_id as string | undefined
      const progressPct = payload.progress_percent as number | undefined
      if (!modelId || progressPct === undefined) return
      registry.value = registry.value.map(m =>
        m.model_id === modelId
          ? { ...m, status: 'downloading', download_progress: progressPct }
          : m,
      )
    }

    else if (eventType === 'model.downloaded') {
      const modelId = payload.model_id as string | undefined
      if (!modelId) return
      const name = registry.value.find(m => m.model_id === modelId)?.name ?? modelId
      isDownloading.value = new Set(isDownloading.value)
      isDownloading.value.delete(modelId)
      useNotificationStore().push('success', `Model "${name}" downloaded successfully!`)
      fetchRegistry()
    }

    else if (eventType === 'model.download_failed') {
      const modelId = payload.model_id as string | undefined
      if (!modelId) return
      const name = registry.value.find(m => m.model_id === modelId)?.name ?? modelId
      isDownloading.value = new Set(isDownloading.value)
      isDownloading.value.delete(modelId)
      useNotificationStore().push('error', `Model "${name}" download failed`)
      fetchRegistry()
    }
  }

  return {
    registry,
    downloadedModels,
    isLoading,
    isDownloading,
    sseConnected,
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
    connectSSE,
    disconnectSSE,
  }
})
