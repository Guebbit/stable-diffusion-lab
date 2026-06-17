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
  /** Job ID of the currently running generation (null when idle). */
  const currentJobId = ref<string | null>(null)
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
   */
  function generate(request: GenerationRequest): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Text-to-image — submitting job…')

    return diffusionApi.submitTextToImage(request)
      .then((submission: JobSubmissionResponse) => {
        currentJobId.value = submission.job_id
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
        currentJobId.value = null
      })
  }

  /**
   * Submit a describe-image (vision) job and wait for completion.
   */
  function describe(modelId: string, imageFile: File): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Describe image — submitting job…')

    return diffusionApi.submitDescribe(modelId, imageFile)
      .then((submission: JobSubmissionResponse) => {
        currentJobId.value = submission.job_id
        notif.push('info', `Job ${submission.job_id} queued`)
        return _waitForJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `Description complete — job ${job.id}`)
          return refreshGallery()
        } else {
          const msg = job.error || `Job ended with status: ${job.status}`
          error.value = msg
          notif.push('error', msg)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Description failed'
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
        currentJobId.value = null
      })
  }

  /**
   * Submit an image-to-image job and wait for completion.
   */
  function imageToImage(request: GenerationRequest & { image?: string }): Promise<void> {
    return _submitMode('Image-to-image', diffusionApi.submitImageToImage, request)
  }



  /**
   * Submit a recolor job and wait for completion.
   */
  function recolor(modelId: string, imageFile: File, prompt: string, strength: number = 0.75): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Recolor — submitting job…')

    return diffusionApi.submitRecolor(modelId, imageFile, prompt, strength)
      .then((submission: JobSubmissionResponse) => {
        currentJobId.value = submission.job_id
        notif.push('info', `Job ${submission.job_id} queued`)
        return _waitForJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `Recolor complete — job ${job.id}`)
          return refreshGallery()
        } else {
          const msg = job.error || `Job ended with status: ${job.status}`
          error.value = msg
          notif.push('error', msg)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Recolor failed'
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
        currentJobId.value = null
      })
  }

  /**
   * Submit a sketch-to-ink job and wait for completion.
   */
  function sketchToInk(
    modelId: string,
    imageFile: File,
    prompt: string = '',
    negativePrompt: string = '',
    numInferenceSteps: number = 28,
    guidanceScale: number = 8.0,
    adapterConditioningScale: number = 0.9,
    baseModelId: string = '',
    loraModelId: string = '',
    loraStrength: number = 0.8,
  ): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', 'Sketch-to-ink — submitting job…')

    return diffusionApi.submitSketchToInk(modelId, imageFile, {
      prompt,
      negativePrompt,
      numInferenceSteps,
      guidanceScale,
      adapterConditioningScale,
      baseModelId: baseModelId || undefined,
      loraModelId: loraModelId || undefined,
      loraStrength,
    })
      .then((submission: JobSubmissionResponse) => {
        currentJobId.value = submission.job_id
        notif.push('info', `Job ${submission.job_id} queued`)
        return _waitForJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `Sketch-to-ink complete — job ${job.id}`)
          return refreshGallery()
        } else {
          const msg = job.error || `Job ended with status: ${job.status}`
          error.value = msg
          notif.push('error', msg)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Sketch-to-ink failed'
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
        currentJobId.value = null
      })
  }

  /**
   * Generic submission helper for any mode.
   */
  function _submitMode(
    label: string,
    submitFn: (payload: any) => Promise<JobSubmissionResponse>,
    request: any,
  ): Promise<void> {
    const notif = useNotificationStore()
    isGenerating.value = true
    error.value = null
    notif.push('info', `${label} — submitting job…`)

    return submitFn(request)
      .then((submission: JobSubmissionResponse) => {
        currentJobId.value = submission.job_id
        notif.push('info', `Job ${submission.job_id} queued`)
        return _waitForJobCompletion(submission.job_id)
      })
      .then((job: JobStatusResponse) => {
        if (job.status === 'completed') {
          notif.push('success', `${label} complete — job ${job.id}`)
          return refreshGallery()
        } else {
          const msg = job.error || `Job ended with status: ${job.status}`
          error.value = msg
          notif.push('error', msg)
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : `${label} failed`
        error.value = msg
        notif.push('error', msg)
      })
      .finally(() => {
        isGenerating.value = false
        currentJobId.value = null
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

  /** Delete a single artifact from the backend and remove it from the gallery. */
  function deleteImage(artifactId: string): Promise<void> {
    const notif = useNotificationStore()
    return diffusionApi.deleteArtifact(artifactId)
      .then(() => {
        generatedImages.value = generatedImages.value.filter(img => img.id !== artifactId)
        notif.push('success', 'Image deleted')
      })
      .catch(() => {
        notif.push('error', 'Failed to delete image')
      })
  }

  /** Delete all artifacts from the backend and clear the gallery. */
  function clearImages(): Promise<void> {
    const notif = useNotificationStore()
    return diffusionApi.deleteAllArtifacts()
      .then(() => {
        generatedImages.value = []
        notif.push('success', 'All images deleted')
      })
      .catch(() => {
        notif.push('error', 'Failed to delete all images')
      })
  }

  /** Request cancellation of the currently running job. */
  function cancelCurrentJob(): Promise<void> {
    if (!currentJobId.value) return Promise.resolve()
    const jobId = currentJobId.value
    const notif = useNotificationStore()
    return diffusionApi.cancelJob(jobId)
      .then(() => {
        notif.push('info', 'Cancellation requested…')
      })
      .catch(() => {
        notif.push('warning', 'Failed to request cancellation')
      })
  }

  /** Clear current UI error. */
  function clearError() {
    error.value = null
  }

  /** Initialize store by fetching the gallery from the backend. */
  function init() {
    refreshGallery()
  }

  /**
   * Wait for a job to reach a terminal state via SSE events.
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
    currentJobId,
    error,
    jobProgress,
    sseConnected,
    fetchStatus,
    generate,
    describe,
    imageToImage,
    recolor,
    sketchToInk,
    cancelCurrentJob,
    refreshGallery,
    deleteImage,
    clearImages,
    clearError,
    connectObservability,
    disconnectObservability,
    init,
  }
})