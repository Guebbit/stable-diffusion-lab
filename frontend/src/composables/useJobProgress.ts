import { ref, onUnmounted } from 'vue'

/**
 * Job progress event from the backend WebSocket /ws/progress.
 * Mirrors the backend JobProgress dataclass broadcast format.
 */
export interface JobProgressEvent {
  job_id: string
  status: string
  progress_percent: number
  current_step: number
  total_steps: number
  message: string
  timestamp: string
}

/**
 * Composable that connects to the backend WebSocket for real-time job progress.
 * Provides reactive state for current progress events and connection status.
 *
 * Usage:
 *   const { events, isConnected, connect, disconnect } = useJobProgress()
 *   connect() // starts listening
 */
export function useJobProgress() {
  // Reactive state for the latest progress event per job
  const events = ref<Map<string, JobProgressEvent>>(new Map())
  // Connection status flag
  const isConnected = ref(false)
  // Internal WebSocket reference
  let ws: WebSocket | null = null
  // Reconnect timer handle
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  /**
   * Build the WebSocket URL from the current page origin.
   * Converts http(s) to ws(s) and appends the progress path.
   */
  function buildWsUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/ws/progress`
  }

  /**
   * Open the WebSocket connection and start listening for progress events.
   */
  function connect() {
    if (ws && ws.readyState <= WebSocket.OPEN) return

    ws = new WebSocket(buildWsUrl())

    ws.onopen = () => {
      isConnected.value = true
    }

    ws.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as JobProgressEvent
      // Store latest progress per job_id so the UI can track multiple jobs
      events.value = new Map(events.value).set(data.job_id, data)
    }

    ws.onclose = () => {
      isConnected.value = false
      // Auto-reconnect after 3 seconds
      scheduleReconnect()
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  /**
   * Schedule an automatic reconnection attempt.
   */
  function scheduleReconnect() {
    if (reconnectTimer) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, 3000)
  }

  /**
   * Cleanly close the WebSocket and stop reconnect attempts.
   */
  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.onclose = null // prevent auto-reconnect on intentional close
      ws.close()
      ws = null
    }
    isConnected.value = false
  }

  /**
   * Remove a completed/failed job from the tracked events map.
   */
  function clearJob(jobId: string) {
    const next = new Map(events.value)
    next.delete(jobId)
    events.value = next
  }

  // Cleanup on component unmount
  onUnmounted(() => {
    disconnect()
  })

  return {
    events,
    isConnected,
    connect,
    disconnect,
    clearJob,
  }
}
