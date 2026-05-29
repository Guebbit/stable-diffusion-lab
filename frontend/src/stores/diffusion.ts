import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
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
  async function fetchStatus() {
    try {
      status.value = await diffusionApi.getStatus()
    } catch {
      status.value = null
    }
  }

  /**
   * Wrap generation calls with shared loading/error state handling.
   */
  async function runGeneration(
    request: () => Promise<GenerationResponse>,
    fallbackMessage: string,
  ) {
    isGenerating.value = true
    error.value = null

    try {
      const response = await request()
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : fallbackMessage
    } finally {
      isGenerating.value = false
    }
  }

  /** Load a model for the selected task and refresh backend status afterwards. */
  async function loadModel(modelId: string, source: ModelSource, task: GenerationTask = 'text2img') {
    isLoadingModel.value = true
    error.value = null

    try {
      await diffusionApi.loadModel({ model_id: modelId, model_source: source, task })
      await fetchStatus()
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Failed to load model'
    } finally {
      isLoadingModel.value = false
    }
  }

  /** Trigger standard text-to-image generation and prepend returned images. */
  function generate(request: GenerationRequest) {
    return runGeneration(() => diffusionApi.generate(request), 'Generation failed')
  }

  /** Trigger image-to-image generation and prepend returned images. */
  function generateFromImage(request: ImageGenerationRequest) {
    return runGeneration(() => diffusionApi.generateFromImage(request), 'Image generation failed')
  }

  /** Trigger sketch-to-ink generation and prepend returned images. */
  function generateSketchToInk(request: SketchToInkRequest) {
    return runGeneration(() => diffusionApi.generateSketchToInk(request), 'Sketch generation failed')
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
