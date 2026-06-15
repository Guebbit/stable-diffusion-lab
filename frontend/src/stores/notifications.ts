/**
 * Notification / activity-log store.
 *
 * Two responsibilities:
 *  1. "current" toast — the single most recent message shown briefly in a snackbar.
 *  2. "logs" list — a running history of all notifications shown in the activity log panel.
 *
 * Usage: call push(level, message) from any store or component.
 * The toast auto-dismisses after TOAST_DURATION_MS; the log persists until clearLogs().
 * On app startup, call loadFromJobs() to hydrate the log from persisted job history.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { JobStatusResponse } from '../types'

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
const CLEARED_AT_KEY = 'activity_log_cleared_at'

let _nextId = 0

function _jobToLevel(status: string): NotificationLevel {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'cancelled') return 'warning'
  return 'info'
}

function _jobToMessage(job: JobStatusResponse): string {
  const type = job.job_type.replace(/_/g, '-')
  if (job.status === 'completed') return `${type} completed`
  if (job.status === 'failed') return `${type} failed${job.error ? `: ${job.error}` : ''}`
  if (job.status === 'cancelled') return `${type} cancelled`
  return `${type} ${job.status}`
}

export const useNotificationStore = defineStore('notifications', () => {
  // All past notifications, newest first — drives the activity log panel
  const logs = ref<Notification[]>([])
  // The notification currently shown as a toast (null = hidden)
  const current = ref<Notification | null>(null)

  let _dismissTimer: ReturnType<typeof setTimeout> | null = null

  // Persisted cutoff: entries from before this date are not shown (set by clearLogs)
  const _clearedAt = localStorage.getItem(CLEARED_AT_KEY)
  const _clearedAtDate = _clearedAt ? new Date(_clearedAt) : null

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

  /**
   * Wipe all log history and persist the cutoff so that a page refresh
   * does not re-hydrate entries the user explicitly cleared.
   */
  function clearLogs() {
    logs.value = []
    localStorage.setItem(CLEARED_AT_KEY, new Date().toISOString())
  }

  /**
   * Hydrate the activity log from persisted job records fetched from the backend.
   * Only terminal-state jobs (completed/failed/cancelled) after the last clear are shown.
   * Appended below any in-session entries already in the log.
   */
  function loadFromJobs(jobs: JobStatusResponse[]) {
    const terminal = new Set(['completed', 'failed', 'cancelled'])
    const historical: Notification[] = jobs
      .filter((job) => terminal.has(job.status))
      .filter((job) => !_clearedAtDate || new Date(job.created_at) > _clearedAtDate)
      .map((job) => ({
        id: _nextId++,
        level: _jobToLevel(job.status),
        message: _jobToMessage(job),
        timestamp: new Date(job.completed_at ?? job.created_at),
      }))
    logs.value = [...logs.value, ...historical]
  }

  return { logs, current, push, dismiss, clearLogs, loadFromJobs }
})
