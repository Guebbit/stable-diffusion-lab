import axios from 'axios'
import type {
  GenerationRequest,
  GenerationResponse,
  ImageGenerationRequest,
  ModelLoadRequest,
  ModelLoadResponse,
  BackendStatus,
} from '../types'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
})

export const diffusionApi = {
  getStatus(): Promise<BackendStatus> {
    return api.get<BackendStatus>('/status').then((r) => r.data)
  },

  loadModel(payload: ModelLoadRequest): Promise<ModelLoadResponse> {
    return api.post<ModelLoadResponse>('/models/load', payload).then((r) => r.data)
  },

  generate(payload: GenerationRequest): Promise<GenerationResponse> {
    return api.post<GenerationResponse>('/generate', payload).then((r) => r.data)
  },

  generateFromImage(payload: ImageGenerationRequest): Promise<GenerationResponse> {
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

    return api.post<GenerationResponse>('/generate-from-image', formData).then((r) => r.data)
  },
}
