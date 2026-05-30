/**
 * Shared TypeScript types for the frontend.
 * These mirror the backend Pydantic schemas — they define the shape of
 * data flowing between frontend ↔ backend API calls.
 */

// Where a model lives (HuggingFace repo vs CivitAI community checkpoint)
export type ModelSource = 'huggingface' | 'civitai'

// Backend generation task — determines which pipeline class gets loaded
export type GenerationTask = 'text2img' | 'img2img' | 'sketch2ink'

// Preset config for img2img workflows (each one tweaks strength/steps/guidance)
export type ImageWorkflowPreset = 'general' | 'recolor' | 'style-transfer' | 'upscale'

// UI-level mode selector (maps to a GenerationTask + preset combo on submit)
export type GenerationMode =
  | 'text-to-image'
  | 'image-to-image'
  | 'recolor-image'
  | 'style-transfer'
  | 'upscale-image'
  | 'sketch-to-ink'

/** A selectable model entry shown in the dropdown. */
export interface ModelOption {
  id: string
  name: string
  source: ModelSource
  description?: string
}

/** Payload sent to /api/generate (text-to-image). */
export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  model_id: string
  model_source: ModelSource
  width: number
  height: number
  num_inference_steps: number   // Denoising iterations (more = better quality, slower)
  guidance_scale: number        // How strictly the AI follows your prompt
  seed?: number                 // For reproducible results
  num_images: number
}

/** Payload sent to /api/generate-from-image (img2img workflows). */
export interface ImageGenerationRequest {
  image: File                            // The uploaded reference image
  prompt: string
  negative_prompt?: string
  model_id: string
  model_source: ModelSource
  workflow_preset?: ImageWorkflowPreset  // Which preset defaults to use
  strength: number                       // How much to change the original (0=none, 1=total)
  num_inference_steps: number
  guidance_scale: number
  width?: number
  height?: number
  seed?: number
  num_images: number
}

/** Payload sent to /api/generate-sketch-to-ink (ControlNet scribble pipeline). */
export interface SketchToInkRequest {
  image: File                              // The sketch/scribble to clean up
  prompt: string
  negative_prompt?: string
  model_id: string
  model_source: 'huggingface'              // Only HF models supported for ControlNet
  controlnet_conditioning_scale: number    // How strongly the sketch constrains output
  num_inference_steps: number
  guidance_scale: number
  width?: number
  height?: number
  seed?: number
  num_images: number
}

/** A single generated image displayed in the gallery. */
export interface GeneratedImage {
  id: string
  url: string           // Base64 data URL (embedded PNG)
  prompt: string
  negative_prompt?: string
  model_id: string
  width: number
  height: number
  seed: number          // Seed used — for "recreate this exact image" feature
  created_at: string
}

/** Response returned by all generation endpoints. */
export interface GenerationResponse {
  images: GeneratedImage[]
  model_id: string
  elapsed_seconds: number   // How long the AI took to generate
}

/** Payload sent to /api/models/load (pre-load a model before generating). */
export interface ModelLoadRequest {
  model_id: string
  model_source: ModelSource
  task?: GenerationTask
}

/** Response from model load endpoint. */
export interface ModelLoadResponse {
  success: boolean
  model_id: string
  message: string
}

/** Backend health status (polled periodically by the UI). */
export interface BackendStatus {
  status: 'ok' | 'loading' | 'error'
  loaded_model?: string    // Currently cached model key (or undefined if cold start)
  device: string           // "cuda" or "cpu"
  message?: string
}
