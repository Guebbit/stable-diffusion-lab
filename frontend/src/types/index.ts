/**
 * Backend status response from /api/status.
 * The frontend polls this to show connection state and which model is loaded.
 */
export interface BackendStatus {
  status: 'ok' | 'loading' | 'error'
  loaded_model: string | null
  device: string
  message?: string
}

/**
 * Where the model weights are hosted.
 * Drives download logic: HuggingFace uses snapshot_download(), CivitAI fetches a single .safetensors.
 */
export type ModelSource = 'huggingface' | 'civitai'

/**
 * Stable Diffusion architecture family.
 * Matters for ControlNet compatibility and native resolution defaults.
 */
export type ModelFamily = 'sd15' | 'sdxl' | 'flux'

/**
 * Supported generation task types (matches backend GenerationTask).
 */
export type GenerationTask = 'text2img' | 'img2img' | 'sketch2ink'

/**
 * Frontend-only generation mode — maps to backend tasks + workflow presets.
 */
export type GenerationMode =
  | 'text-to-image'
  | 'image-to-image'
  | 'recolor-image'
  | 'style-transfer'
  | 'upscale-image'
  | 'sketch-to-ink'

/**
 * Preset names sent to the img2img endpoint to select generation defaults.
 * Each maps to a different (strength, steps, guidance) combination on the backend.
 */
export type ImageWorkflowPreset = 'general' | 'recolor' | 'style-transfer' | 'upscale'

/**
 * Standard text-to-image generation request (JSON body to /api/generate).
 */
export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  model_id: string
  model_source: ModelSource
  num_inference_steps: number
  guidance_scale: number
  num_images: number
  width?: number
  height?: number
  seed?: number
}

/**
 * Response from all generation endpoints (/api/generate, /api/generate-from-image, etc.).
 */
export interface GenerationResponse {
  images: GeneratedImage[]
  model_id: string
  elapsed_seconds: number
}

/**
 * Single generated image with metadata (returned by generation endpoints and /api/history).
 */
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
  num_inference_steps: number
  guidance_scale: number
  generation_time_seconds: number
  model_load_time_seconds?: number
  device: string
  vram_used_mb?: number
  scheduler: string
  pipeline_class: string
}

/**
 * Request to pre-load a model pipeline (so the first generation is faster).
 */
export interface ModelLoadRequest {
  model_id: string
  model_source: ModelSource
  task?: GenerationTask
}

/**
 * Response after a model load attempt.
 */
export interface ModelLoadResponse {
  success: boolean
  model_id: string
  message: string
}

/**
 * Multipart-form request for image-based generation (img2img, recolor, style-transfer, upscale).
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
  workflow_preset?: ImageWorkflowPreset
  strength?: number
}

/**
 * Multipart-form request for sketch-to-ink ControlNet generation.
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
 * Vision model response with the generated text description.
 */
export interface DescribeImageResponse {
  description: string
  model_id: string
  elapsed_seconds: number
}

/**
 * Model registry entry as returned by /api/models.
 * Mirrors the backend ModelRegistryEntry Pydantic schema.
 */
export interface ModelRegistryEntry {
  id: string
  name: string
  source: ModelSource
  family: ModelFamily
  description: string
  long_description: string
  tags: string[]
  source_url: string
  size: string
  downloaded: boolean
  status: string
}

/**
 * Payload to register a new model in the catalog (POST /api/models).
 */
export interface ModelRegistryAddRequest {
  id: string
  name: string
  source: ModelSource
  family: ModelFamily
  description: string
  long_description: string
  tags: string[]
  source_url: string
  size: string
}

/**
 * Download event for tracking background download operations.
 */
export interface DownloadEvent {
  model_id: string
  source: string
  timestamp: string
  status: 'started' | 'completed' | 'failed'
  detail?: string
}

/**
 * Frontend-only catalog entry (for the static model list in data/models.ts).
 * Uses camelCase for JS-friendly access; not sent to the backend.
 */
export interface ModelOption {
  id: string
  name: string
  source: ModelSource
  family: ModelFamily
  description: string
  tags: string[]
  longDescription: string
  sourceUrl: string
}