<script setup lang="ts">
/**
 * Root layout component.
 * - Top bar: backend connection status + navigation links
 * - Router view: renders the active page
 * - Error banner: API errors from the store
 * - Toast snackbar: brief auto-dismissing notification
 * - Activity log panel: live job queue / history from the backend
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useDiffusionStore } from './stores/diffusion'
import { useHistoryStore } from './stores/history'
import { useModelsStore } from './stores/models'
import { useNotificationStore, LEVEL_COLOR, LEVEL_ICON } from './stores/notifications'
import { diffusionApi } from './api/diffusion'
import type { JobStatusResponse } from './types'

const store = useDiffusionStore()
const historyStore = useHistoryStore()
const modelsStore = useModelsStore()
const notif = useNotificationStore()

// ── Job queue / activity log ───────────────────────────────────────────────

const jobs = ref<JobStatusResponse[]>([])

async function fetchJobs() {
  try {
    const res = await diffusionApi.getJobs({ limit: 100 })
    jobs.value = res.items
  } catch { /* backend offline — keep previous list */ }
}

async function clearFinishedJobs() {
  await diffusionApi.deleteAllFinishedJobs().catch(() => {})
  await fetchJobs()
}

async function cancelJob(jobId: string) {
  await diffusionApi.cancelJob(jobId).catch(() => {})
  await fetchJobs()
}

async function deleteJob(jobId: string) {
  await diffusionApi.deleteJob(jobId).catch(() => {})
  jobs.value = jobs.value.filter(j => j.id !== jobId)
}

// Re-fetch when a generation finishes or download completes
watch(() => store.isGenerating, (val) => { if (!val) fetchJobs() })
watch(() => modelsStore.isDownloading.size, (size) => { if (size === 0) fetchJobs() })

// ── Helpers ────────────────────────────────────────────────────────────────

const JOB_LABEL: Record<string, string> = {
  text_to_image:   'Text to Image',
  image_to_image:  'Image to Image',
  model_download:  'Model Download',
  upscale:         'Upscale',
  image_analysis:  'Image Analysis',
  describe:        'Describe',
  recolor:         'Recolor',
  sketch_to_ink:   'Sketch to Ink',
  video_generation:'Video Generation',
  llm_inference:   'LLM Inference',
}

const STATUS_COLOR: Record<string, string> = {
  pending:   'default',
  running:   'info',
  completed: 'success',
  failed:    'error',
  cancelled: 'warning',
}

const STATUS_ICON: Record<string, string> = {
  pending:   'mdi-clock-outline',
  running:   'mdi-loading mdi-spin',
  completed: 'mdi-check-circle-outline',
  failed:    'mdi-close-circle-outline',
  cancelled: 'mdi-cancel',
}

function jobLabel(job: JobStatusResponse): string {
  return JOB_LABEL[job.job_type] ?? job.job_type.replace(/_/g, ' ')
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return ''
  return new Date(iso).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

const hasFinished = computed(() =>
  jobs.value.some(j => ['completed', 'failed', 'cancelled'].includes(j.status))
)

const activeJobCount = computed(() =>
  jobs.value.filter(j => j.status === 'pending' || j.status === 'running').length
)

// ── Disk/storage formatting ────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

// ── App bootstrap ──────────────────────────────────────────────────────────

let _statusInterval: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  store.fetchStatus()
  store.connectObservability()
  store.init()
  historyStore.init()
  modelsStore.init()
  fetchJobs()
  _statusInterval = setInterval(() => store.fetchStatus(), 30_000)
})

onUnmounted(() => {
  if (_statusInterval) clearInterval(_statusInterval)
})

const showToast = computed({
  get: () => notif.current !== null,
  set: (val) => { if (!val) notif.dismiss() },
})

const navItems = [
  { title: 'Image',   icon: 'mdi-creation',              to: '/image' },
  { title: 'Upscale', icon: 'mdi-image-size-select-large', to: '/image-upscale' },
  { title: 'Models',  icon: 'mdi-brain',                  to: '/models' },
  { title: 'History', icon: 'mdi-history',                to: '/history' },
]
</script>

<template>
  <v-app>
    <v-app-bar color="surface-variant" elevation="1">
      <v-app-bar-title>
        <v-icon icon="mdi-atom-variant" class="mr-2" color="primary" />
        Stable Diffusion Lab
      </v-app-bar-title>

      <!-- Navigation buttons -->
      <v-btn
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        :prepend-icon="item.icon"
        variant="text"
        class="mx-1"
      >
        {{ item.title }}
      </v-btn>

      <v-spacer />

      <template #append>
        <template v-if="store.status">
          <span class="text-caption text-medium-emphasis mr-2">
            {{ formatBytes(store.status.models_disk_bytes) }} models
            · {{ formatBytes(store.status.disk_free_bytes) }} free
          </span>
          <v-chip
            :color="store.status.status === 'ok' ? 'success' : 'warning'"
            size="small"
            class="mr-3"
            prepend-icon="mdi-server"
          >
            {{ store.status.device.toUpperCase() }}
          </v-chip>
        </template>
        <v-chip
          v-else
          color="error"
          size="small"
          class="mr-3"
          prepend-icon="mdi-server-off"
        >
          Backend offline
        </v-chip>
      </template>
    </v-app-bar>

    <v-main>
      <v-container fluid class="py-6">
        <!-- Error banner -->
        <v-alert
          v-if="store.error"
          type="error"
          closable
          class="mb-4"
          @click:close="store.clearError()"
        >
          {{ store.error }}
        </v-alert>

        <!-- Active route renders here -->
        <router-view />

        <!-- Activity log (job queue) -->
        <v-row class="mt-4">
          <v-col cols="12">
            <v-expansion-panels variant="accordion">
              <v-expansion-panel @group:selected="fetchJobs()">
                <v-expansion-panel-title>
                  <v-icon icon="mdi-format-list-bulleted" class="mr-2" />
                  Activity Log
                  <v-chip
                    v-if="activeJobCount"
                    size="x-small"
                    class="ml-2"
                    color="info"
                  >
                    {{ activeJobCount }} active
                  </v-chip>
                  <v-chip
                    v-else-if="jobs.length"
                    size="x-small"
                    class="ml-2"
                    color="default"
                  >
                    {{ jobs.length }}
                  </v-chip>
                </v-expansion-panel-title>

                <v-expansion-panel-text>
                  <div v-if="!jobs.length" class="text-medium-emphasis text-body-2 py-2">
                    No jobs yet — activity appears here when you generate images or download models.
                  </div>

                  <v-list v-else density="compact" class="pa-0">
                    <v-list-item
                      v-for="job in jobs"
                      :key="job.id"
                      :prepend-icon="STATUS_ICON[job.status]"
                      :base-color="STATUS_COLOR[job.status]"
                    >
                      <v-list-item-title class="text-body-2 font-weight-medium">
                        {{ jobLabel(job) }}
                      </v-list-item-title>
                      <v-list-item-subtitle v-if="job.status === 'failed' && job.error" class="text-caption text-error">
                        {{ job.error }}
                      </v-list-item-subtitle>

                      <!-- Progress bar for running jobs -->
                      <v-progress-linear
                        v-if="job.status === 'running'"
                        :model-value="job.progress_percent"
                        :indeterminate="job.progress_percent === 0"
                        color="info"
                        height="2"
                        class="mt-1"
                        rounded
                      />

                      <template #append>
                        <div class="d-flex align-center gap-2">
                          <span class="text-caption text-medium-emphasis">
                            {{ formatTimestamp(job.completed_at ?? job.created_at) }}
                          </span>
                          <!-- Cancel active jobs -->
                          <v-tooltip v-if="job.status === 'pending' || job.status === 'running'" location="top" text="Cancel">
                            <template #activator="{ props: tp }">
                              <v-btn v-bind="tp" icon="mdi-close" size="x-small" variant="text" color="warning" @click.stop="cancelJob(job.id)" />
                            </template>
                          </v-tooltip>
                          <!-- Delete finished jobs -->
                          <v-tooltip v-else location="top" text="Remove from history">
                            <template #activator="{ props: tp }">
                              <v-btn v-bind="tp" icon="mdi-delete-outline" size="x-small" variant="text" color="error" @click.stop="deleteJob(job.id)" />
                            </template>
                          </v-tooltip>
                        </div>
                      </template>
                    </v-list-item>
                  </v-list>

                  <div class="d-flex gap-2 mt-2">
                    <v-btn
                      size="small"
                      variant="text"
                      prepend-icon="mdi-refresh"
                      @click="fetchJobs()"
                    >
                      Refresh
                    </v-btn>
                    <v-btn
                      v-if="hasFinished"
                      size="small"
                      variant="text"
                      color="error"
                      prepend-icon="mdi-delete-sweep"
                      @click="clearFinishedJobs()"
                    >
                      Clear finished
                    </v-btn>
                  </div>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
          </v-col>
        </v-row>
      </v-container>
    </v-main>

    <!-- Toast snackbar -->
    <v-snackbar
      v-model="showToast"
      :color="notif.current ? LEVEL_COLOR[notif.current.level] : undefined"
      :timeout="5000"
      location="bottom right"
      min-width="300"
    >
      <v-icon :icon="notif.current ? LEVEL_ICON[notif.current.level] : 'mdi-bell'" class="mr-2" />
      {{ notif.current?.message }}

      <template #actions>
        <v-btn variant="text" icon="mdi-close" @click="notif.dismiss()" />
      </template>
    </v-snackbar>
  </v-app>
</template>

<style>
html,
body {
  margin: 0;
  padding: 0;
}
</style>
