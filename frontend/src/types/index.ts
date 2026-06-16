/**
 * System state snapshot from /api/v1/system/status.
 * Comprehensive health + resource info from the observability service.
 */
export interface SystemStatus {
  status: string
  version: string
  device: string
  runtime: { app_name: string; uptime_seconds: number; inference_backend: string }
  gpu: { busy: boolean; active_job_id: string | null; cuda_allocated_mb: number; cuda_reserved_mb: number }
  models: { cache_size: number; max_cached: number; active_model: string | null; loaded_models: string[] }
  jobs: { pending: number; running: number; failed: number; oldest_pending_age_seconds: number }
  health: { status: string; warnings: string[]; blockers: string[]; recommendations: string[] }
}

/**
 * Where the model weights are hosted (drives backend download strategy).
 */
export type ModelSource = 'huggingface' | 'civitai'

/**
 * Stable Diffusion architecture family.
 * Matters for ControlNet compatibility and native resolution defaults.
 */
export type ModelFamily = 'sd15' | 'sdxl' | 'flux' | 'custom'

/**
 * Supported generation task types (matches backend JobType).
 */
export type GenerationTask = 'text_to_image' | 'image_to_image' | 'video_generation'

/**
 * Frontend-only generation mode — maps to backend tasks + workflow presets.
 */
export type GenerationMode =
  | 'text-to-image'
  | 'image-to-image'
  | 'describe-image'
  | 'recolor-image'
  | 'sketch-to-ink'

/**
 * Text-to-image generation request (JSON body to /api/v1/generation/text-to-image).
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
  lora_model_id?: string
  lora_strength?: number
}

/**
 * Async job submission response — returned immediately with a job_id to track.
 */
export interface JobSubmissionResponse {
  job_id: string
  status: string
  message: string
}

/**
 * Full job status as returned by /api/v1/jobs/{job_id}.
 */
export interface JobStatusResponse {
  id: string
  job_type: string
  status: string
  progress_percent: number
  current_step: number
  total_steps: number
  message: string
  params: Record<string, unknown>
  result: Record<string, unknown>
  error: string
  created_at: string
  started_at: string | null
  completed_at: string | null
}

/**
 * Artifact entry from /api/v1/artifacts (replaces legacy GeneratedImage).
 */
export interface ArtifactEntry {
  id: string
  job_id: string | null
  file_path: string
  thumbnail_path: string
  media_type: string
  size_bytes: number
  width: number
  height: number
  prompt: string
  negative_prompt: string
  seed: number
  model_name: string
  model_id_ref: string
  generation_params: Record<string, unknown>
  is_favorite: boolean
  rating: number
  notes: string
  created_at: string
  updated_at: string | null
}

/**
 * Paginated response wrapper from backend list endpoints.
 */
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

/**
 * Model registry entry from /api/v1/models.
 * Mirrors the backend ModelRegistryResponse schema.
 */
export interface ModelRegistryEntry {
  id: string
  model_id: string
  name: string
  preferred_name: string | null
  source: string
  family: string
  variant: string
  model_type: string
  compatible_bases: string[]
  description: string
  short_description: string
  tags: string[]
  source_url: string
  capabilities: string[]
  allowed_capabilities: string[]
  config: Record<string, unknown>
  status: string
  total_size_bytes: number
  disk_size_bytes: number
  is_verified: boolean
  local_path: string | null
  file_count: number
  download_size_bytes: number | null
  recommended_vram_min_gb: number | null
  recommended_vram_max_gb: number | null
  license: string
  base_model: string
  precision: string
  requirements: Record<string, unknown>
  notes: string
  created_at: string
  updated_at: string | null
}

/**
 * Payload to register a new model in the catalog (POST /api/v1/models).
 */
export interface ModelRegistryAddRequest {
  model_id: string
  name: string
  source: ModelSource
  family: string
  variant?: string
  description: string
  tags: string[]
  source_url: string
  capabilities: string[]
  local_path?: string
  download_size_bytes?: number | null
  recommended_vram_min_gb?: number | null
  recommended_vram_max_gb?: number | null
  license?: string
  base_model?: string
  precision?: string
  requirements?: Record<string, unknown>
  notes?: string
}

/**
 * Typed observability event from /ws/observability stream.
 */
export interface ObservabilityEvent {
  event_id: string
  event_type: string
  timestamp: string
  correlation_id: string | null
  job_id: string | null
  level: string
  message: string
  payload: Record<string, unknown>
  type: string
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