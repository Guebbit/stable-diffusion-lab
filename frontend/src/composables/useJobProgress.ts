import { ref, onUnmounted } from 'vue'
import type { ObservabilityEvent } from '../types'

/**
 * Composable that connects to /ws/observability for real-time typed events.
 * Replaces the legacy /ws/progress endpoint with richer event streaming.
 *
 * Usage:
 *   const { events, isConnected, connect, disconnect } = useObservabilityStream()
 *   connect() // starts listening for all events
 *   connect('job,resource') // subscribe to specific event categories
 */
export function useObservabilityStream(defaultSubscribe?: string) {
  // Reactive state: latest event per job for progress tracking
  const events = ref<Map<string, ObservabilityEvent>>(new Map())
  // All recent events buffer (for activity log)
  const recentEvents = ref<ObservabilityEvent[]>([])
  // Connection status flag
  const isConnected = ref(false)
  // Internal WebSocket reference
  let ws: WebSocket | null = null
  // Reconnect timer handle
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  /**
   * Build the WebSocket URL for the observability stream.
   * Converts http(s) to ws(s) and appends optional subscription filters.
   */
  function buildWsUrl(subscribe?: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const base = `${protocol}//${window.location.host}/ws/observability`
    const filter = subscribe || defaultSubscribe
    return filter ? `${base}?subscribe=${encodeURIComponent(filter)}` : base
  }

  /**
   * Open the WebSocket and start listening for typed events.
   */
  function connect(subscribe?: string) {
    if (ws && ws.readyState === WebSocket.OPEN) return

    ws = new WebSocket(buildWsUrl(subscribe))

    ws.onopen = () => {
      isConnected.value = true
    }

    ws.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as ObservabilityEvent
      // Track latest event per job_id for progress display
      if (data.job_id) {
        events.value = new Map(events.value).set(data.job_id, data)
      }
      // Keep a rolling buffer of recent events (max 200)
      recentEvents.value = [data, ...recentEvents.value].slice(0, 200)
    }

    ws.onclose = () => {
      isConnected.value = false
      scheduleReconnect(subscribe)
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  /**
   * Schedule an automatic reconnection attempt.
   */
  function scheduleReconnect(subscribe?: string) {
    if (reconnectTimer) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect(subscribe)
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
      ws.onclose = null
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
    recentEvents,
    isConnected,
    connect,
    disconnect,
    clearJob,
  }
}
