import axios from 'axios'
import type {
  BackendStatus,
  GenerationRequest,
  GenerationResponse,
  ImageGenerationRequest,
  ModelLoadRequest,
  ModelLoadResponse,
  RecolorRequest,
  SketchToInkRequest,
  UpscaleRequest,
} from '../types'

/** Shared Axios instance for every backend API call in the app. */
const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
})

/**
 * Convert shared image-to-image request fields into multipart form-data.
 * The backend expects multipart so it can read both the image binary and parameters.
 */
function createImageWorkflowFormData(payload: ImageGenerationRequest): FormData {
  const formData = new FormData()
  formData.append('image', payload.image)
  formData.append('prompt', payload.prompt)
  formData.append('model_id', payload.model_id)
  formData.append('model_source', payload.model_source)
  formData.append('strength', String(payload.strength))
  formData.append('num_inference_steps', String(payload.num_inference_steps))
  formData.append('guidance_scale', String(payload.guidance_scale))
  formData.append('num_images', String(payload.num_images))

  if (payload.negative_prompt) formData.append('negative_prompt', payload.negative_prompt)
  if (typeof payload.width === 'number') formData.append('width', String(payload.width))
  if (typeof payload.height === 'number') formData.append('height', String(payload.height))
  if (typeof payload.seed === 'number') formData.append('seed', String(payload.seed))

  return formData
}

export const diffusionApi = {
  /** Read backend health and currently loaded model/device information. */
  getStatus(): Promise<BackendStatus> {
    return api.get<BackendStatus>('/status').then((r) => r.data)
  },

  /** Ask the backend to load a model for a specific generation task. */
  loadModel(payload: ModelLoadRequest): Promise<ModelLoadResponse> {
    return api.post<ModelLoadResponse>('/models/load', payload).then((r) => r.data)
  },

  /** Standard text-to-image generation request. */
  generate(payload: GenerationRequest): Promise<GenerationResponse> {
    return api.post<GenerationResponse>('/generate', payload).then((r) => r.data)
  },

  /** Generic image-to-image generation endpoint introduced in Phase 1. */
  generateFromImage(payload: ImageGenerationRequest): Promise<GenerationResponse> {
    const formData = createImageWorkflowFormData(payload)
    return api.post<GenerationResponse>('/generate-from-image', formData).then((r) => r.data)
  },

  /** Sketch-to-ink endpoint backed by ControlNet conditioning. */
  generateSketchToInk(payload: SketchToInkRequest): Promise<GenerationResponse> {
    const formData = new FormData()
    formData.append('image', payload.image)
    formData.append('prompt', payload.prompt)
    formData.append('model_id', payload.model_id)
    formData.append('model_source', payload.model_source)
    formData.append('controlnet_conditioning_scale', String(payload.controlnet_conditioning_scale))
    formData.append('num_inference_steps', String(payload.num_inference_steps))
    formData.append('guidance_scale', String(payload.guidance_scale))
    formData.append('num_images', String(payload.num_images))

    if (payload.negative_prompt) formData.append('negative_prompt', payload.negative_prompt)
    if (typeof payload.width === 'number') formData.append('width', String(payload.width))
    if (typeof payload.height === 'number') formData.append('height', String(payload.height))
    if (typeof payload.seed === 'number') formData.append('seed', String(payload.seed))

    return api.post<GenerationResponse>('/generate-sketch-to-ink', formData).then((r) => r.data)
  },

  /** Phase 3 recolor endpoint for palette/style edits from an uploaded image. */
  generateRecolor(payload: RecolorRequest): Promise<GenerationResponse> {
    const formData = createImageWorkflowFormData(payload)
    return api.post<GenerationResponse>('/generate-recolor', formData).then((r) => r.data)
  },

  /** Phase 3 upscale endpoint for quality-focused enlargement workflows. */
  generateUpscale(payload: UpscaleRequest): Promise<GenerationResponse> {
    const formData = new FormData()
    formData.append('image', payload.image)
    formData.append('model_id', payload.model_id)
    formData.append('model_source', payload.model_source)
    formData.append('upscale_factor', String(payload.upscale_factor))
    formData.append('strength', String(payload.strength))
    formData.append('num_inference_steps', String(payload.num_inference_steps))
    formData.append('guidance_scale', String(payload.guidance_scale))
    formData.append('num_images', String(payload.num_images))

    if (payload.prompt) formData.append('prompt', payload.prompt)
    if (payload.negative_prompt) formData.append('negative_prompt', payload.negative_prompt)
    if (typeof payload.width === 'number') formData.append('width', String(payload.width))
    if (typeof payload.height === 'number') formData.append('height', String(payload.height))
    if (typeof payload.seed === 'number') formData.append('seed', String(payload.seed))

    return api.post<GenerationResponse>('/generate-upscale', formData).then((r) => r.data)
  },
}
