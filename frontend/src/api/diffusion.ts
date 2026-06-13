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

  // ─── Jobs ──

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
   * Remove a model from the registry.
   */
  removeModel(modelId: string): Promise<void> {
    return api.delete(`/models/${encodeURIComponent(modelId)}`).then(() => undefined)
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