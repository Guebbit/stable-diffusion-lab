<script setup lang="ts">
/**
 * Root layout component.
 * - Top bar: shows backend connection status + navigation links
 * - Router view: renders the active page
 * - Error banner: displays API errors from the store
 * - Toast snackbar: brief auto-dismissing notification
 * - Activity log panel: collapsible list of all past notifications
 */
import { computed, onMounted } from 'vue'
import { useDiffusionStore } from './stores/diffusion'
import { useHistoryStore } from './stores/history'
import { useModelsStore } from './stores/models'
import { useNotificationStore, LEVEL_COLOR, LEVEL_ICON } from './stores/notifications'

const store = useDiffusionStore()
const historyStore = useHistoryStore()
const modelsStore = useModelsStore()
const notif = useNotificationStore()

// Initialize all stores on app mount so data survives page refresh
onMounted(() => {
  store.fetchStatus()
  store.connectObservability()
  store.init()
  historyStore.init()
  modelsStore.init()
})

// v-model for v-snackbar
const showToast = computed({
  get: () => notif.current !== null,
  set: (val) => { if (!val) notif.dismiss() },
})

// Navigation items for the app bar
const navItems = [
  { title: 'Image', icon: 'mdi-image-sparkle', to: '/image' },
  { title: 'Models', icon: 'mdi-brain', to: '/models' },
  { title: 'History', icon: 'mdi-history', to: '/history' },
]

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}
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
        <v-chip
          v-if="store.status"
          :color="store.status.status === 'ok' ? 'success' : 'warning'"
          size="small"
          class="mr-3"
          prepend-icon="mdi-server"
        >
          {{ store.status.device?.toUpperCase() ?? 'Backend' }}
          <template v-if="store.status.models?.active_model">
            · {{ store.status.models.active_model }}
          </template>
        </v-chip>
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

        <!-- Activity log -->
        <v-row class="mt-4">
          <v-col cols="12">
            <v-expansion-panels variant="accordion">
              <v-expansion-panel>
                <v-expansion-panel-title>
                  <v-icon icon="mdi-format-list-bulleted" class="mr-2" />
                  Activity Log
                  <v-chip
                    v-if="notif.logs.length"
                    size="x-small"
                    class="ml-2"
                    color="primary"
                  >
                    {{ notif.logs.length }}
                  </v-chip>
                </v-expansion-panel-title>

                <v-expansion-panel-text>
                  <div v-if="!notif.logs.length" class="text-medium-emphasis text-body-2 py-2">
                    No activity yet — log entries appear here as you interact with the app.
                  </div>

                  <v-list v-else density="compact" class="pa-0">
                    <v-list-item
                      v-for="entry in notif.logs"
                      :key="entry.id"
                      :prepend-icon="LEVEL_ICON[entry.level]"
                      :base-color="LEVEL_COLOR[entry.level]"
                    >
                      <v-list-item-title class="text-body-2">
                        {{ entry.message }}
                      </v-list-item-title>
                      <template #append>
                        <span class="text-caption text-medium-emphasis">
                          {{ formatTime(entry.timestamp) }}
                        </span>
                      </template>
                    </v-list-item>
                  </v-list>

                  <v-btn
                    v-if="notif.logs.length"
                    size="small"
                    variant="text"
                    color="error"
                    prepend-icon="mdi-delete-outline"
                    class="mt-2"
                    @click="notif.clearLogs()"
                  >
                    Clear log
                  </v-btn>
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
