import axios from 'axios'
import type {
  BackendStatus,
  GenerationRequest,
  GenerationResponse,
  ImageGenerationRequest,
  ModelLoadRequest,
  ModelLoadResponse,
  SketchToInkRequest,
} from '../types'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
})

/**
 * Append optional numeric or string fields only when they are explicitly set.
 */
function appendOptionalField(formData: FormData, key: string, value?: string | number): void {
  if (typeof value === 'undefined') return
  formData.append(key, String(value))
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
    const formData = new FormData()
    formData.append('image', payload.image)
    formData.append('prompt', payload.prompt)
    formData.append('model_id', payload.model_id)
    formData.append('model_source', payload.model_source)
    appendOptionalField(formData, 'workflow_preset', payload.workflow_preset)
    formData.append('strength', String(payload.strength))
    formData.append('num_inference_steps', String(payload.num_inference_steps))
    formData.append('guidance_scale', String(payload.guidance_scale))
    formData.append('num_images', String(payload.num_images))

    if (payload.negative_prompt) {
      formData.append('negative_prompt', payload.negative_prompt)
    }
    appendOptionalField(formData, 'width', payload.width)
    appendOptionalField(formData, 'height', payload.height)
    appendOptionalField(formData, 'seed', payload.seed)

    return api.post<GenerationResponse>('/generate-from-image', formData).then((r) => r.data)
  },

  /**
   * Run sketch-to-ink generation using the backend ControlNet workflow.
   */
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

    if (payload.negative_prompt) {
      formData.append('negative_prompt', payload.negative_prompt)
    }
    appendOptionalField(formData, 'width', payload.width)
    appendOptionalField(formData, 'height', payload.height)
    appendOptionalField(formData, 'seed', payload.seed)

    return api.post<GenerationResponse>('/generate-sketch-to-ink', formData).then((r) => r.data)
  },
}
