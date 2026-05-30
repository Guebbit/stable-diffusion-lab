/**
 * Notification / activity-log store.
 *
 * Two responsibilities:
 *  1. "current" toast — the single most recent message shown briefly in a snackbar.
 *  2. "logs" list — a running history of all notifications shown in the activity log panel.
 *
 * Usage: call push(level, message) from any store or component.
 * The toast auto-dismisses after TOAST_DURATION_MS; the log persists until clearLogs().
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'

export type NotificationLevel = 'info' | 'success' | 'warning' | 'error'

export interface Notification {
  id: number
  level: NotificationLevel
  message: string
  timestamp: Date
}

// Colors used by v-snackbar / v-chip to visually distinguish severity
export const LEVEL_COLOR: Record<NotificationLevel, string> = {
  info: 'info',
  success: 'success',
  warning: 'warning',
  error: 'error',
}

// Icons for the activity log list entries
export const LEVEL_ICON: Record<NotificationLevel, string> = {
  info: 'mdi-information-outline',
  success: 'mdi-check-circle-outline',
  warning: 'mdi-alert-outline',
  error: 'mdi-close-circle-outline',
}

const TOAST_DURATION_MS = 5000

let _nextId = 0

export const useNotificationStore = defineStore('notifications', () => {
  // All past notifications, newest first — drives the activity log panel
  const logs = ref<Notification[]>([])
  // The notification currently shown as a toast (null = hidden)
  const current = ref<Notification | null>(null)

  let _dismissTimer: ReturnType<typeof setTimeout> | null = null

  /** Add a new notification, surface it as a toast, and append it to the log. */
  function push(level: NotificationLevel, message: string) {
    const entry: Notification = { id: _nextId++, level, message, timestamp: new Date() }

    // Prepend to log (newest first)
    logs.value = [entry, ...logs.value]

    // Replace any in-flight toast and restart the auto-dismiss timer
    if (_dismissTimer) clearTimeout(_dismissTimer)
    current.value = entry
    _dismissTimer = setTimeout(dismiss, TOAST_DURATION_MS)
  }

  /** Manually close the current toast (e.g. user clicks × button). */
  function dismiss() {
    current.value = null
    if (_dismissTimer) {
      clearTimeout(_dismissTimer)
      _dismissTimer = null
    }
  }

  /** Wipe all log history (keeps current toast if one is showing). */
  function clearLogs() {
    logs.value = []
  }

  return { logs, current, push, dismiss, clearLogs }
})
