export type ModelSource = 'huggingface' | 'civitai'

export interface ModelOption {
  id: string
  name: string
  source: ModelSource
  description?: string
}

export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  model_id: string
  model_source: ModelSource
  width: number
  height: number
  num_inference_steps: number
  guidance_scale: number
  seed?: number
  num_images: number
}

export interface GeneratedImage {
  id: string
  url: string
  prompt: string
  negative_prompt?: string
  model_id: string
  width: number
  height: number
  seed: number
  created_at: string
}

export interface GenerationResponse {
  images: GeneratedImage[]
  model_id: string
  elapsed_seconds: number
}

export interface ModelLoadRequest {
  model_id: string
  model_source: ModelSource
}

export interface ModelLoadResponse {
  success: boolean
  model_id: string
  message: string
}

export interface BackendStatus {
  status: 'ok' | 'loading' | 'error'
  loaded_model?: string
  device: string
  message?: string
}
