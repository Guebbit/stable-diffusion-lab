import type { ModelSource } from '@/types/models'

/**
 * Backend status including active device and loaded model info.
 */
export interface BackendStatus {
  device: 'cpu' | 'cuda' | 'mps'
  vram_used_mb: number
  loaded_model_key: string
  task_type: 'text2img' | 'img2img' | 'inpainting' | 'sketch2ink'
}

/**
 * Standard text-to-image generation request.
 */
export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  num_inference_steps: number
  guidance_scale: number
  num_images: number
  width?: number
  height?: number
  seed?: number
  workflow_preset?: 'turbo' | 'hd' | 'realistic'
  img2img_strength?: number
  img2img_denoising?: number
}

/**
 * Response containing generated images and timing info.
 */
export interface GenerationResponse {
  images: GeneratedImage[]
  elapsed_seconds: number
  seed_used: number
  prompts: {
    prompt: string
    negative_prompt?: string
  }
}

/**
 * Single generated image with metadata.
 */
export interface GeneratedImage {
  uuid: string
  image_url: string
  thumbnail_url?: string
  prompt: string
  negative_prompt?: string
  seed: number
  width: number
  height: number
  steps: number
  guidance_scale: number
  workflow_preset?: string
  workflow_metadata?: any
  device: 'cpu' | 'cuda' | 'mps'
  vram_used_mb?: number
}

/**
 * Request to load a specific model for a generation task.
 */
export interface ModelLoadRequest {
  model_id: string
  model_source: ModelSource
  task?: GenerationTask
}

/**
 * Response confirming model load with timing and resource info.
 */
export interface ModelLoadResponse {
  model_id: string
  model_source: ModelSource
  task?: GenerationTask
  message: string
  elapsed_seconds: number
  device?: 'cpu' | 'cuda' | 'mps'
  vram_mb?: number
  loaded: boolean
}

/**
 * Supported generation task types.
 */
export type GenerationTask = 'text2img' | 'img2img' | 'inpainting' | 'sketch2ink'

/**
 * Image-based generation request (img2img, inpainting, sketch2ink).
 */
export interface ImageGenerationRequest {
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
  workflow_preset?: 'turbo' | 'hd' | 'realistic'
  strength?: number
  img2img_denoising?: number
  controlnet_conditioning_scale?: number
}

/**
 * Vision model request to describe an uploaded image.
 */
export interface DescribeImageRequest {
  image: File
  model_id: string
}

/**
 * Vision model response with generated caption.
 */
export interface DescribeImageResponse {
  image_url: string
  caption: string
  elapsed_seconds: number
}

/**
 * Model source enumeration.
 */
export type ModelSource = 'huggingface' | 'civitai' | 'local'

/**
 * Model registry entry showing download status.
 */
export interface ModelRegistryEntry {
  model_id: string
  task_type: GenerationTask
  model_url: string
  model_source: ModelSource
  is_downloaded: boolean
  loaded: boolean
  device?: 'cpu' | 'cuda' | 'mps'
  vram_mb?: number
  last_loaded_at?: string
  message?: string
  error?: string
}

/**
 * Request to add a model manually to the registry.
 */
export interface ModelRegistryAddRequest {
  model_id: string
  task_type: GenerationTask
  model_url: string
  model_source: ModelSource
}

/**
 * Download event for tracking background download operations.
 */
export interface DownloadEvent {
  model_id: string
  source: string
  timestamp: string
  status: 'started' | 'completed' | 'failed'
  message?: string
  error?: string
}

/**
 * Sketch-to-ink generation request.
 */
export interface SketchToInkRequest {
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
  workflow_preset?: 'turbo' | 'hd' | 'realistic'
  controlnet_conditioning_scale?: number
}

/**
 * Download history entry.
 */
export interface DownloadHistoryEntry {
  model_id: string
  model_name: string
  download_date: string
  source: ModelSource
  task_type?: GenerationTask
  size_gb: number
  status: 'completed' | 'failed' | 'partial'
  error?: string
}