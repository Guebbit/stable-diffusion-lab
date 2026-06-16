/**
 * Notification store — drives the toast snackbar only.
 *
 * Usage: call push(level, message) from any store or component.
 * The toast auto-dismisses after TOAST_DURATION_MS.
 *
 * The activity log panel (in App.vue) now reads directly from the backend
 * job queue instead of this store.
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

export const LEVEL_COLOR: Record<NotificationLevel, string> = {
  info: 'info',
  success: 'success',
  warning: 'warning',
  error: 'error',
}

export const LEVEL_ICON: Record<NotificationLevel, string> = {
  info: 'mdi-information-outline',
  success: 'mdi-check-circle-outline',
  warning: 'mdi-alert-outline',
  error: 'mdi-close-circle-outline',
}

const TOAST_DURATION_MS = 5000
let _nextId = 0

export const useNotificationStore = defineStore('notifications', () => {
  const current = ref<Notification | null>(null)
  let _dismissTimer: ReturnType<typeof setTimeout> | null = null

  function push(level: NotificationLevel, message: string) {
    const entry: Notification = { id: _nextId++, level, message, timestamp: new Date() }
    if (_dismissTimer) clearTimeout(_dismissTimer)
    current.value = entry
    _dismissTimer = setTimeout(dismiss, TOAST_DURATION_MS)
  }

  function dismiss() {
    current.value = null
    if (_dismissTimer) {
      clearTimeout(_dismissTimer)
      _dismissTimer = null
    }
  }

  return { current, push, dismiss }
})
