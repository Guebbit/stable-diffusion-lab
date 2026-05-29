import { defineStore } from 'pinia'
import { ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import type {
  GeneratedImage,
  GenerationRequest,
  ImageGenerationRequest,
  ModelSource,
  BackendStatus,
} from '../types'

export const useDiffusionStore = defineStore('diffusion', () => {
  const status = ref<BackendStatus | null>(null)
  const generatedImages = ref<GeneratedImage[]>([])
  const isGenerating = ref(false)
  const isLoadingModel = ref(false)
  const error = ref<string | null>(null)

  async function fetchStatus() {
    try {
      status.value = await diffusionApi.getStatus()
    } catch {
      status.value = null
    }
  }

  async function loadModel(modelId: string, source: ModelSource) {
    isLoadingModel.value = true
    error.value = null
    try {
      await diffusionApi.loadModel({ model_id: modelId, model_source: source })
      await fetchStatus()
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : 'Failed to load model'
    } finally {
      isLoadingModel.value = false
    }
  }

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

  function clearImages() {
    generatedImages.value = []
  }

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
    clearImages,
    clearError,
  }
})
