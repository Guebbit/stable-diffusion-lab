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

  // ─── SSE Observability Stream ──

  /** Real-time observability events via SSE (maps job_id → latest event). */
  const jobProgress = ref<Map<string, ObservabilityEvent>>(new Map())
  /** Whether the SSE stream is connected. */
  const sseConnected = ref(false)
  /** Observability stream composable instance (lazy-initialized). */
  let _stream: ReturnType<typeof useObservabilityStream> | null = null

  /**
   * Start the SSE connection for real-time job/system events.
   * Safe to call multiple times — only connects once.
   */
  function connectObservability() {
    if (!_stream) {
      _stream = useObservabilityStream('job')
      // Keep store refs in sync with composable's reactive state
      watch(() => _stream!.isConnected.value, (val) => { sseConnected.value = val })
      watch(() => _stream!.events.value, (val) => { jobProgress.value = val }, { deep: true })
    }
    _stream.connect()
  }

  /**
   * Disconnect observability stream.
   */
  function disconnectObservability() {
    _stream?.disconnect()
    sseConnected.value = false
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
   * Submit a text-to-image job and wait for completion.
   * Listens for SSE events (job.completed / job.failed / job.cancelled) to know
   * when the job reached a terminal state, then fetches the final status via API.
   */
  function generate(request: GenerationRequest): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Text-to-image — submitting job…')

    return diffusionApi.submitTextToImage(request)
      .then((submission: JobSubmissionResponse) => {
        notif.push('info', `Job ${submission.job_id} queued`)
        return _waitForJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `Generation complete — job ${job.id}`)
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
   * Wait for a job to reach a terminal state via SSE events.
   *
   * The SSE stream emits `job.progress`, `job.completed`, `job.failed`,
   * and `job.cancelled` events with progress percentage included.
   * We watch the `jobProgress` map for a terminal event, then fetch
   * the final status from the API to resolve the promise.
   */
  function _waitForJobCompletion(jobId: string): Promise<JobStatusResponse> {
    const terminalTypes = new Set([
      'job.completed',
      'job.failed',
      'job.cancelled',
    ])

    return new Promise((resolve, reject) => {
      let settled = false

      // Watch the jobProgress map for terminal events
      const unwatch = watch(() => jobProgress.value.get(jobId), (event) => {
        if (settled) return
        if (event && terminalTypes.has(event.event_type)) {
          settled = true
          unwatch()
          diffusionApi.getJobStatus(jobId)
            .then(resolve)
            .catch(reject)
        }
      })

      // Check immediately in case the event already arrived
      const event = jobProgress.value.get(jobId)
      if (event && terminalTypes.has(event.event_type)) {
        settled = true
        unwatch()
        diffusionApi.getJobStatus(jobId)
          .then(resolve)
          .catch(reject)
      }
    })
  }

  return {
    status,
    generatedImages,
    isGenerating,
    error,
    jobProgress,
    sseConnected,
    fetchStatus,
    generate,
    refreshGallery,
    clearImages,
    clearError,
    connectObservability,
    disconnectObservability,
  }
})