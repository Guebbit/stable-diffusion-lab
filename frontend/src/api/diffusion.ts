import axios from 'axios'
import type {
  GenerationRequest,
  GenerationResponse,
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
}
