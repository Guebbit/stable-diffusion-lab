import { ref, onUnmounted } from 'vue'
import type { ObservabilityEvent } from '../types'

/**
 * Composable that connects to /sse/observability for real-time typed events.
 * Uses the Server-Sent Events API (EventSource) — unidirectional, simpler than WS.
 *
 * Usage:
 *   const { events, isConnected, connect, disconnect } = useObservabilityStream()
 *   connect() // starts listening for all events
 *   connect('job,resource') // subscribe to specific event categories
 */
export function useObservabilityStream(defaultSubscribe?: string) {
  // Reactive state: latest event per job for progress tracking
  const events = ref<Map<string, ObservabilityEvent>>(new Map())
  // Connection status flag
  const isConnected = ref(false)
  // Internal EventSource reference
  let source: EventSource | null = null
  // Reconnect timer handle
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  /**
   * Build the SSE URL for the observability stream.
   * Resolves relative to the current origin so it works behind proxies.
   */
  function buildSseUrl(subscribe?: string): string {
    const base = `${window.location.origin}/sse/observability`
    const filter = subscribe || defaultSubscribe
    return filter ? `${base}?subscribe=${encodeURIComponent(filter)}` : base
  }

  /**
   * Open the EventSource and start listening for typed events.
   */
  function connect(subscribe?: string) {
    if (source && source.readyState === EventSource.OPEN) return

    source = new EventSource(buildSseUrl(subscribe))

    source.onopen = () => {
      isConnected.value = true
    }

    source.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as ObservabilityEvent
      // Track latest event per job_id for progress display
      if (data.job_id) {
        events.value = new Map(events.value).set(data.job_id, data)
      }
    }

    source.onerror = () => {
      isConnected.value = false
      source?.close()
      scheduleReconnect(subscribe)
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
    }, 5000)
  }

  /**
   * Cleanly close the EventSource and stop reconnect attempts.
   */
  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (source) {
      source.close()
      source = null
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