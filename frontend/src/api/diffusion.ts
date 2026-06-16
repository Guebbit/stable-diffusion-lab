import axios from 'axios'
import type {
  ArtifactEntry,
  GenerationRequest,
  JobStatusResponse,
  JobSubmissionResponse,
  ModelRegistryAddRequest,
  ModelRegistryEntry,
  PaginatedResponse,
  SystemStatus,
} from '../types'

// Base client pointing to versioned API
export const api = axios.create({
  baseURL: '/api/v1',
  timeout: 600000,
})

export const diffusionApi = {
  // ─── System / Status ──

  /**
   * Fetch comprehensive system status from the observability service.
   */
  getStatus(): Promise<SystemStatus> {
    return api.get<SystemStatus>('/system/status').then((r) => r.data)
  },

  // ─── Generation (async job-based) ──

  /**
   * Submit a text-to-image generation job. Returns immediately with a job_id.
   */
  submitTextToImage(payload: GenerationRequest): Promise<JobSubmissionResponse> {
    return api.post<JobSubmissionResponse>('/generation/text-to-image', payload).then((r) => r.data)
  },

  /**
   * Submit an image-to-image generation job.
   */
  submitImageToImage(payload: GenerationRequest & { image?: string }): Promise<JobSubmissionResponse> {
    return api.post<JobSubmissionResponse>('/generation/image-to-image', payload).then((r) => r.data)
  },

  /**
   * Submit an upscale pipeline job (multipart/form-data with file upload).
   * Endpoint: POST /api/v1/upscale/run
   */
  submitUpscale(
    modelId: string,
    imageFile: File,
    options: {
      // Step 2: upscale (required params)
      scaleFactor?: number
      prompt?: string
      noiseLevel?: number
      numInferenceSteps?: number
      // Step 1: enhancement (optional — omit or pass undefined to skip)
      enhanceModelId?: string
      enhanceStrength?: number
      // Step 3: face restore (optional — omit or pass undefined to skip)
      faceRestoreModelId?: string
      faceRestoreFidelity?: number
    } = {},
  ): Promise<JobSubmissionResponse> {
    const formData = new FormData()
    formData.append('model_id', modelId)
    formData.append('image', imageFile)
    formData.append('scale_factor', String(options.scaleFactor ?? 2.0))
    formData.append('prompt', options.prompt ?? '')
    formData.append('noise_level', String(options.noiseLevel ?? 20))
    formData.append('num_inference_steps', String(options.numInferenceSteps ?? 20))
    if (options.enhanceModelId) {
      formData.append('enhance_model_id', options.enhanceModelId)
      formData.append('enhance_strength', String(options.enhanceStrength ?? 0.4))
    }
    if (options.faceRestoreModelId) {
      formData.append('face_restore_model_id', options.faceRestoreModelId)
      formData.append('face_restore_fidelity', String(options.faceRestoreFidelity ?? 0.5))
    }
    return api.post<JobSubmissionResponse>('/upscale/run', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  /**
   * Submit a describe-image (vision) job (multipart/form-data with file upload).
   */
  submitDescribe(modelId: string, imageFile: File): Promise<JobSubmissionResponse> {
    const formData = new FormData()
    formData.append('model_id', modelId)
    formData.append('image', imageFile)
    return api.post<JobSubmissionResponse>('/generation/describe', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  /**
   * Submit a recolor job (multipart/form-data with file upload).
   */
  submitRecolor(modelId: string, imageFile: File, prompt: string, strength: number = 0.75): Promise<JobSubmissionResponse> {
    const formData = new FormData()
    formData.append('model_id', modelId)
    formData.append('image', imageFile)
    formData.append('prompt', prompt)
    formData.append('strength', String(strength))
    return api.post<JobSubmissionResponse>('/generation/recolor', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  /**
   * Submit a sketch-to-ink job (multipart/form-data with file upload).
   */
  submitSketchToInk(
    modelId: string,
    imageFile: File,
    options: {
      prompt?: string
      negativePrompt?: string
      numInferenceSteps?: number
      guidanceScale?: number
      adapterConditioningScale?: number
      baseModelId?: string
      loraModelId?: string
      loraStrength?: number
    } = {},
  ): Promise<JobSubmissionResponse> {
    const formData = new FormData()
    formData.append('model_id', modelId)
    formData.append('image', imageFile)
    formData.append('prompt', options.prompt ?? '')
    formData.append('negative_prompt', options.negativePrompt ?? '')
    formData.append('num_inference_steps', String(options.numInferenceSteps ?? 28))
    formData.append('guidance_scale', String(options.guidanceScale ?? 8.0))
    formData.append('adapter_conditioning_scale', String(options.adapterConditioningScale ?? 0.9))
    if (options.baseModelId) formData.append('base_model_id', options.baseModelId)
    if (options.loraModelId) formData.append('lora_model_id', options.loraModelId)
    if (options.loraModelId) formData.append('lora_strength', String(options.loraStrength ?? 0.8))
    return api.post<JobSubmissionResponse>('/generation/sketch-to-ink', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  // ─── Jobs ──

  /**
   * List jobs with optional filtering and pagination (newest first).
   */
  getJobs(params?: { status?: string; job_type?: string; limit?: number; offset?: number }): Promise<PaginatedResponse<JobStatusResponse>> {
    return api.get<PaginatedResponse<JobStatusResponse>>('/jobs/', { params }).then((r) => r.data)
  },

  /**
   * Get detailed status for a specific job (poll for progress/completion).
   */
  getJobStatus(jobId: string): Promise<JobStatusResponse> {
    return api.get<JobStatusResponse>(`/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data)
  },

  /**
   * Cancel a running or pending job.
   */
  cancelJob(jobId: string): Promise<{ job_id: string; status: string; message: string }> {
    return api.post(`/jobs/${encodeURIComponent(jobId)}/cancel`).then((r) => r.data)
  },

  // ─── Artifacts (replaces legacy /history) ──

  /**
   * Fetch paginated artifact gallery (generated outputs).
   */
  getArtifacts(params?: { limit?: number; offset?: number; model_name?: string }): Promise<PaginatedResponse<ArtifactEntry>> {
    return api.get<PaginatedResponse<ArtifactEntry>>('/artifacts/', { params }).then((r) => r.data)
  },

  /**
   * Delete a single artifact by UUID.
   */
  deleteArtifact(artifactId: string): Promise<void> {
    return api.delete(`/artifacts/${encodeURIComponent(artifactId)}`).then(() => undefined)
  },

  deleteAllArtifacts(): Promise<void> {
    return api.delete('/artifacts/').then(() => undefined)
  },

  // ─── Model Registry ──

  /**
   * Get all registered models with their download/status info.
   */
  getModels(params?: { limit?: number; offset?: number }): Promise<ModelRegistryEntry[]> {
    return api.get<ModelRegistryEntry[]>('/models/', { params }).then((r) => r.data)
  },

  /**
   * Get a single model's details.
   */
  getModel(modelId: string): Promise<ModelRegistryEntry> {
    return api.get<ModelRegistryEntry>(`/models/${encodeURIComponent(modelId)}`).then((r) => r.data)
  },

  /**
   * Register a new model in the catalog.
   */
  addModel(payload: ModelRegistryAddRequest): Promise<ModelRegistryEntry> {
    return api.post<ModelRegistryEntry>('/models/', payload).then((r) => r.data)
  },

  /**
   * Remove a model from the registry (DB + disk). Permanent.
   */
  removeModel(modelId: string): Promise<void> {
    return api.delete(`/models/${encodeURIComponent(modelId)}`).then(() => undefined)
  },

  removeAllModels(): Promise<void> {
    return api.delete('/models/').then(() => undefined)
  },

  purgeModelFiles(modelId: string): Promise<void> {
    return api.post(`/models/${encodeURIComponent(modelId)}/purge-files`).then(() => undefined)
  },

  removeModelData(modelId: string): Promise<void> {
    return api.delete(`/models-data/${encodeURIComponent(modelId)}`).then(() => undefined)
  },

  purgeAllModelFiles(): Promise<void> {
    return api.delete('/models-data/').then(() => undefined)
  },

  deleteJob(jobId: string): Promise<void> {
    return api.delete(`/jobs/${encodeURIComponent(jobId)}`).then(() => undefined)
  },

  deleteAllFinishedJobs(): Promise<void> {
    return api.delete('/jobs/').then(() => undefined)
  },

  /**
   * Trigger a background download for a model (returns 202 with job_id).
   */
  downloadModel(modelId: string): Promise<JobSubmissionResponse> {
    return api.post<JobSubmissionResponse>(
      `/models/${encodeURIComponent(modelId)}/download`,
    ).then((r) => r.data)
  },
}