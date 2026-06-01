import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import type {
  BackendStatus,
  GenerationRequest,
  GenerationResponse,
  GeneratedImage,
  GenerationTask,
  ImageGenerationRequest,
  ModelSource,
  SketchToInkRequest,
  DownloadEvent,
} from '../types'

export const useDiffusionStore = defineStore('diffusion', () => {
  /** Backend health metadata and currently loaded model info. */
  const status = ref<BackendStatus | null>(null)
  /** Gallery items shown in the UI (newest first). */
  const generatedImages = ref<GeneratedImage[]>([])
  /** Shared loading flag for any generation endpoint. */
  const isGenerating = ref(false)
  /** Loading flag only for explicit model-load actions. */
  const isLoadingModel = ref(false)
  /** User-friendly error message displayed by the UI. */
  const error = ref<string | null>(null)

  // ─── Download Progress ──

  /** Currently downloading model info (modelId, source, progress). */
  const downloadingModelId = ref<string | null>(null)
  const downloadingModelSource = ref<ModelSource>('huggingface')
  const downloadProgress = ref<{ downloaded_bytes: number; total_bytes: number; percentage: number } | null>(null)
  /** Interval ID for the download progress polling (stored as number for cleanup). */
  const downloadProgressInterval = ref<number | null>(null)

  /** Fetch backend status and silently reset status if the request fails. */
  function fetchStatus() {
    // Track whether we were offline before this call so we can notify on reconnection
    const wasOffline = status.value === null
    return diffusionApi.getStatus()
      .then((s) => {
        // Notify on first connect and also when reconnecting after going offline
        if (wasOffline) {
          useNotificationStore().push('info', `Backend connected — device: ${s.device.toUpperCase()}`)
        }
        status.value = s
      })
      .catch(() => {
        if (status.value !== null) {
          // Only notify once when we first lose connection (not on every failed poll)
          useNotificationStore().push('warning', 'Backend offline — cannot reach the API')
        }
        status.value = null
      })
  }

  /**
   * Wrap generation calls with shared loading/error state handling.
   * Returns a Promise so callers can optionally chain further logic.
   */
  function runGeneration(
    request: () => Promise<GenerationResponse>,
    label: string,
  ): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', `${label} — generating…`)

    return request()
      .then((response) => {
        generatedImages.value = [...response.images, ...generatedImages.value]
        // Build a rich success notification with device/VRAM info
        const firstImage = response.images[0]
        let detail = `${label} done — ${response.images.length} image(s) in ${response.elapsed_seconds}s`
        if (firstImage?.device) {
          detail += ` on ${firstImage.device.toUpperCase()}`
        }
        if (firstImage?.vram_used_mb) {
          detail += ` (${firstImage.vram_used_mb} MB VRAM)`
        }
        notif.push('success', detail)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : `${label} failed`
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
      })
  }

  /** Load a model for the selected task and refresh backend status afterwards. */
  function loadModel(modelId: string, source: ModelSource, task: GenerationTask = 'text2img') {
    const notif = useNotificationStore()
    isLoadingModel.value = true
    error.value = null
    notif.push('info', `Loading model "${modelId}" (${task})…`)

    return diffusionApi.loadModel({ model_id: modelId, model_source: source, task })
      .then((res) => {
        notif.push('success', res.message)
        return fetchStatus()
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to load model'
        error.value = msg
        notif.push('error', `Model load failed: ${msg}`)
      })
      .finally(() => {
        isLoadingModel.value = false
      })
  }

  /** Trigger standard text-to-image generation and prepend returned images. */
  function generate(request: GenerationRequest): Promise<void> {
    return runGeneration(() => diffusionApi.generate(request), 'Text-to-image')
  }

  /** Trigger image-to-image generation and prepend returned images. */
  function generateFromImage(request: ImageGenerationRequest): Promise<void> {
    return runGeneration(() => diffusionApi.generateFromImage(request), 'Image-to-image')
  }

  /** Trigger sketch-to-ink generation and prepend returned images. */
  function generateSketchToInk(request: SketchToInkRequest): Promise<void> {
    return runGeneration(() => diffusionApi.generateSketchToInk(request), 'Sketch-to-ink')
  }

  /** Clear gallery state. */
  function clearImages() {
    generatedImages.value = []
  }

  /** Clear current UI error. */
  function clearError() {
    error.value = null
  }

  // ─── Download Events ──

  /**
   * Fetch all download events from the backend.
   */
  function getDownloadEvents(): Promise<DownloadEvent[]> {
    return diffusionApi.getDownloadEvents().then((events) => events || [])
  }

  /**
   * Clear all download events from the backend.
   */
  function clearDownloadEvents() {
    return diffusionApi.clearDownloadEvents()
  }

  /**
   * Start polling download progress for a model.
   */
  function startDownloadProgressPolling(modelId: string, source: ModelSource) {
    downloadingModelId.value = modelId
    downloadingModelSource.value = source
    downloadProgress.value = null

    // Clear any existing interval first
    if (downloadProgressInterval.value !== null) {
      clearInterval(downloadProgressInterval.value)
    }

    // Poll every 500ms for near-real-time progress
    downloadProgressInterval.value = window.setInterval(async () => {
      try {
        const progress = await diffusionApi.getDownloadProgress(modelId, source)
        downloadProgress.value = progress
        // If download is complete (100%), stop polling
        if (progress.percentage >= 100) {
          stopDownloadProgressPolling()
          // Refresh the models list — caller should handle this
        }
      } catch {
        // No active download or endpoint error — will be caught by error handling
      }
    }, 500)
  }

  /**
   * Stop polling download progress.
   */
  function stopDownloadProgressPolling() {
    if (downloadProgressInterval.value !== null) {
      clearInterval(downloadProgressInterval.value)
      downloadProgressInterval.value = null
    }
    // Only clear model IDs if progress is null (external request)
    if (downloadProgress.value === null) {
      downloadingModelId.value = null
      downloadingModelSource.value = 'huggingface'
    }
  }

  /**
   * Clear download progress state completely.
   */
  function clearDownloadProgress() {
    stopDownloadProgressPolling()
    downloadingModelId.value = null
    downloadingModelSource.value = 'huggingface'
    downloadProgress.value = null
  }

  return {
    status,
    generatedImages,
    isGenerating,
    isLoadingModel,
    error,
    downloadingModelId,
    downloadingModelSource,
    downloadProgress,
    fetchStatus,
    loadModel,
    generate,
    generateFromImage,
    generateSketchToInk,
    clearImages,
    clearError,
    getDownloadEvents,
    clearDownloadEvents,
    startDownloadProgressPolling,
    stopDownloadProgressPolling,
    clearDownloadProgress,
  }
})