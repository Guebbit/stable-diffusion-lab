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

  /** Fetch backend status and silently reset status if the request fails. */
  function fetchStatus() {
    const prevStatus = status.value?.status
    return diffusionApi.getStatus()
      .then((s) => {
        // Notify only on first connect or when coming back online
        if (prevStatus == null) {
          useNotificationStore().push('info', `Backend connected — device: ${s.device.toUpperCase()}`)
        }
        status.value = s
      })
      .catch(() => {
        if (status.value !== null) {
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
        notif.push(
          'success',
          `${label} done — ${response.images.length} image(s) in ${response.elapsed_seconds}s`,
        )
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

  return {
    status,
    generatedImages,
    isGenerating,
    isLoadingModel,
    error,
    fetchStatus,
    loadModel,
    generate,
    generateFromImage,
    generateSketchToInk,
    clearImages,
    clearError,
  }
})
