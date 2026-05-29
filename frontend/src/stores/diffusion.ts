import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import type {
  BackendStatus,
  GenerationRequest,
  GeneratedImage,
  ImageGenerationRequest,
  ModelSource,
  GenerationTask,
  RecolorRequest,
  SketchToInkRequest,
  UpscaleRequest,
} from '../types'

export const useDiffusionStore = defineStore('diffusion', () => {
  /** Last backend status payload, used by the status banner. */
  const status = ref<BackendStatus | null>(null)
  /** Gallery entries rendered in reverse-chronological order. */
  const generatedImages = ref<GeneratedImage[]>([])
  /** Global generation loading state used to disable form actions. */
  const isGenerating = ref(false)
  /** Model-loading state is separate from generation to keep UX explicit. */
  const isLoadingModel = ref(false)
  /** User-facing error text shown when an API request fails. */
  const error = ref<string | null>(null)

  /** Fetch backend status safely; on failure we clear stale data. */
  async function fetchStatus() {
    try {
      status.value = await diffusionApi.getStatus()
    } catch {
      status.value = null
    }
  }

  /** Load model weights for the selected source/model/task combination. */
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

  /** Trigger standard text-to-image generation. */
  async function generate(request: GenerationRequest) {
    isGenerating.value = true
    error.value = null
    try {
      const response = await diffusionApi.generate(request)
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Generation failed'
    } finally {
      isGenerating.value = false
    }
  }

  /** Trigger generic image-to-image generation from an uploaded image. */
  async function generateFromImage(request: ImageGenerationRequest) {
    isGenerating.value = true
    error.value = null
    try {
      const response = await diffusionApi.generateFromImage(request)
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Image generation failed'
    } finally {
      isGenerating.value = false
    }
  }

  /** Trigger Phase 2 sketch-to-ink generation with ControlNet guidance. */
  async function generateSketchToInk(request: SketchToInkRequest) {
    isGenerating.value = true
    error.value = null
    try {
      const response = await diffusionApi.generateSketchToInk(request)
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Sketch generation failed'
    } finally {
      isGenerating.value = false
    }
  }

  /** Trigger Phase 3 recolor workflow. */
  async function generateRecolor(request: RecolorRequest) {
    isGenerating.value = true
    error.value = null
    try {
      const response = await diffusionApi.generateRecolor(request)
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Recolor generation failed'
    } finally {
      isGenerating.value = false
    }
  }

  /** Trigger Phase 3 upscale workflow. */
  async function generateUpscale(request: UpscaleRequest) {
    isGenerating.value = true
    error.value = null
    try {
      const response = await diffusionApi.generateUpscale(request)
      generatedImages.value = [...response.images, ...generatedImages.value]
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Upscale generation failed'
    } finally {
      isGenerating.value = false
    }
  }

  /** Remove all generated images from local state. */
  function clearImages() {
    generatedImages.value = []
  }

  /** Clear the latest error so the user can retry with a clean state. */
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
    generateRecolor,
    generateUpscale,
    clearImages,
    clearError,
  }
})
