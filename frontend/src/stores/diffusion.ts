import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useNotificationStore } from './notifications'
import { useObservabilityStream } from '../composables/useJobProgress'
import type { ObservabilityEvent } from '../types'
import type {
  SystemStatus,
  GenerationRequest,
  JobSubmissionResponse,
  JobStatusResponse,
  ArtifactEntry,
} from '../types'

export const useDiffusionStore = defineStore('diffusion', () => {
  /** System status from observability service. */
  const status = ref<SystemStatus | null>(null)
  /** Gallery items shown in the UI (newest first). */
  const generatedImages = ref<ArtifactEntry[]>([])
  /** Shared loading flag for any generation endpoint. */
  const isGenerating = ref(false)
  /** User-friendly error message displayed by the UI. */
  const error = ref<string | null>(null)

  // ─── WebSocket Observability Stream ──

  /** Real-time observability events via WebSocket (maps job_id → latest event). */
  const jobProgress = ref<Map<string, ObservabilityEvent>>(new Map())
  /** Whether the WebSocket is connected. */
  const wsConnected = ref(false)
  /** Observability stream composable instance (lazy-initialized). */
  let _ws: ReturnType<typeof useObservabilityStream> | null = null

  /**
   * Start the WebSocket connection for real-time job/system events.
   * Safe to call multiple times — only connects once.
   */
  function connectObservability() {
    if (!_ws) {
      _ws = useObservabilityStream('job')
      // Keep store refs in sync with composable's reactive state
      watch(() => _ws!.isConnected.value, (val) => { wsConnected.value = val })
      watch(() => _ws!.events.value, (val) => { jobProgress.value = val }, { deep: true })
    }
    _ws.connect()
  }

  /**
   * Disconnect observability stream.
   */
  function disconnectObservability() {
    _ws?.disconnect()
    wsConnected.value = false
  }

  /** Fetch system status and silently reset if the request fails. */
  function fetchStatus() {
    const wasOffline = status.value === null
    return diffusionApi.getStatus()
      .then((s) => {
        if (wasOffline) {
          useNotificationStore().push('info', `Backend connected — device: ${s.device.toUpperCase()}`)
        }
        status.value = s
      })
      .catch(() => {
        if (status.value !== null) {
          useNotificationStore().push('warning', 'Backend offline — cannot reach the API')
        }
        status.value = null
      })
  }

  /**
   * Submit a text-to-image job and poll until completion.
   * The new API is async: submit → get job_id → poll status.
   */
  function generate(request: GenerationRequest): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Text-to-image — submitting job…')

    return diffusionApi.submitTextToImage(request)
      .then((submission: JobSubmissionResponse) => {
        notif.push('info', `Job ${submission.job_id} queued`)
        // Poll until completion
        return _pollJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `Generation complete — job ${job.id}`)
          // Refresh artifacts gallery
          return refreshGallery()
        } else {
          const msg = job.error || `Job ended with status: ${job.status}`
          error.value = msg
          notif.push('error', msg)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Generation failed'
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
      })
  }

  /** Fetch artifacts from the gallery endpoint. */
  function refreshGallery(): Promise<void> {
    return diffusionApi.getArtifacts({ limit: 50 })
      .then((response) => {
        generatedImages.value = response.items
      })
      .catch(() => {
        // Silent failure — gallery will just be stale
      })
  }

  /** Clear gallery state. */
  function clearImages() {
    generatedImages.value = []
  }

  /** Clear current UI error. */
  function clearError() {
    error.value = null
  }

  /**
   * Poll a job until it reaches a terminal state (completed/failed/cancelled).
   * Returns the final job status response.
   */
  function _pollJobCompletion(jobId: string): Promise<JobStatusResponse> {
    const POLL_INTERVAL = 1000
    const MAX_ATTEMPTS = 600

    return new Promise((resolve, reject) => {
      let attempts = 0
      const interval = setInterval(() => {
        attempts++
        diffusionApi.getJobStatus(jobId)
          .then((job) => {
            if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
              clearInterval(interval)
              resolve(job)
            } else if (attempts >= MAX_ATTEMPTS) {
              clearInterval(interval)
              reject(new Error(`Job polling timed out after ${MAX_ATTEMPTS} attempts`))
            }
          })
          .catch((err) => {
            clearInterval(interval)
            reject(err)
          })
      }, POLL_INTERVAL)
    })
  }

  return {
    status,
    generatedImages,
    isGenerating,
    error,
    jobProgress,
    wsConnected,
    fetchStatus,
    generate,
    refreshGallery,
    clearImages,
    clearError,
    connectObservability,
    disconnectObservability,
  }
})