import axios from 'axios'
import type {
  BackendStatus,
  DescribeImageRequest,
  DescribeImageResponse,
  GenerationRequest,
  GenerationResponse,
  ImageGenerationRequest,
  ModelLoadRequest,
  ModelLoadResponse,
  ModelRegistryAddRequest,
  ModelRegistryEntry,
  ModelSource,
  SketchToInkRequest,
} from '../types'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
})

interface ImageMultipartPayload {
  image: File
  prompt: string
  model_id: string
  model_source: ModelSource
  negative_prompt?: string
  num_inference_steps: number
  guidance_scale: number
  num_images: number
  width?: number
  height?: number
  seed?: number
}

/**
 * Append multipart fields only when they are explicitly set.
 */
function appendOptionalField(formData: FormData, key: string, value?: string | number): void {
  if (typeof value === 'undefined') return
  formData.append(key, String(value))
}

/**
 * Build shared multipart payload fields used by image-guided requests.
 */
function buildImageMultipartData(payload: ImageMultipartPayload): FormData {
  const formData = new FormData()
  formData.append('image', payload.image)
  formData.append('prompt', payload.prompt)
  formData.append('model_id', payload.model_id)
  formData.append('model_source', payload.model_source)
  appendOptionalField(formData, 'num_inference_steps', payload.num_inference_steps)
  appendOptionalField(formData, 'guidance_scale', payload.guidance_scale)
  appendOptionalField(formData, 'num_images', payload.num_images)
  appendOptionalField(formData, 'width', payload.width)
  appendOptionalField(formData, 'height', payload.height)
  appendOptionalField(formData, 'seed', payload.seed)

  if (payload.negative_prompt) {
    formData.append('negative_prompt', payload.negative_prompt)
  }

  return formData
}

export const diffusionApi = {
  /**
   * Read backend status, including active device and loaded model key.
   */
  getStatus(): Promise<BackendStatus> {
    return api.get<BackendStatus>('/status').then((r) => r.data)
  },

  /**
   * Ask the backend to load or switch the model pipeline for a task.
   */
  loadModel(payload: ModelLoadRequest): Promise<ModelLoadResponse> {
    return api.post<ModelLoadResponse>('/models/load', payload).then((r) => r.data)
  },

  /**
   * Run standard text-to-image generation.
   */
  generate(payload: GenerationRequest): Promise<GenerationResponse> {
    return api.post<GenerationResponse>('/generate', payload).then((r) => r.data)
  },

  /**
   * Run img2img generation by sending prompt settings + uploaded input image.
   */
  generateFromImage(payload: ImageGenerationRequest): Promise<GenerationResponse> {
    const formData = buildImageMultipartData(payload)
    appendOptionalField(formData, 'workflow_preset', payload.workflow_preset)
    appendOptionalField(formData, 'strength', payload.strength)

    return api.post<GenerationResponse>('/generate-from-image', formData).then((r) => r.data)
  },

  /**
   * Run sketch-to-ink generation using the backend ControlNet workflow.
   */
  generateSketchToInk(payload: SketchToInkRequest): Promise<GenerationResponse> {
    const formData = buildImageMultipartData(payload)
    appendOptionalField(formData, 'controlnet_conditioning_scale', payload.controlnet_conditioning_scale)

    return api.post<GenerationResponse>('/generate-sketch-to-ink', formData).then((r) => r.data)
  },

  /**
   * Send an image to the vision model for captioning/description.
   */
  describeImage(payload: DescribeImageRequest): Promise<DescribeImageResponse> {
    const formData = new FormData()
    formData.append('image', payload.image)
    formData.append('model_id', payload.model_id)

    return api.post<DescribeImageResponse>('/describe-image', formData).then((r) => r.data)
  },

  // ─── Model Registry endpoints ───────────────────────────────────────────

  /**
   * Get all registered models with their download status.
   */
  getModels(): Promise<ModelRegistryEntry[]> {
    return api.get<ModelRegistryEntry[]>('/models').then((r) => r.data)
  },

  /**
   * Get only models that are downloaded and ready to use.
   */
  getDownloadedModels(): Promise<ModelRegistryEntry[]> {
    return api.get<ModelRegistryEntry[]>('/models/downloaded').then((r) => r.data)
  },

  /**
   * Register a new model in the catalog.
   */
  addModel(payload: ModelRegistryAddRequest): Promise<ModelRegistryEntry> {
    return api.post<ModelRegistryEntry>('/models', payload).then((r) => r.data)
  },

  /**
   * Remove a model from the registry.
   */
  removeModel(modelId: string, source: ModelSource): Promise<void> {
    return api.delete(`/models/${encodeURIComponent(modelId)}`, { params: { source } }).then(() => undefined)
  },

  /**
   * Trigger a background download for a model.
   */
  downloadModel(modelId: string, source: ModelSource): Promise<{ detail: string; status: string }> {
    return api.post<{ detail: string; status: string }>(
      `/models/${encodeURIComponent(modelId)}/download`,
      null,
      { params: { source } },
    ).then((r) => r.data)
  },
}
