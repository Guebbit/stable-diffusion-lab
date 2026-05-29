<script setup lang="ts">
/**
 * Root layout component.
 * - Top bar: shows backend connection status + loaded model name
 * - Left column: GenerationForm (all controls)
 * - Right column: ImageGallery (generated results)
 * - Error banner: displays API errors from the store
 */
import { onMounted } from 'vue'
import { useDiffusionStore } from './stores/diffusion'
import GenerationForm from './components/GenerationForm.vue'
import ImageGallery from './components/ImageGallery.vue'

const store = useDiffusionStore()

// Check backend health on first load (shows device + loaded model in the top bar)
onMounted(() => {
  store.fetchStatus()
})
</script>

<template>
  <v-app>
    <v-app-bar color="surface-variant" elevation="1">
      <v-app-bar-title>
        <v-icon icon="mdi-atom-variant" class="mr-2" color="primary" />
        Stable Diffusion Lab
      </v-app-bar-title>

      <template #append>
        <v-chip
          v-if="store.status"
          :color="store.status.status === 'ok' ? 'success' : 'warning'"
          size="small"
          class="mr-3"
          prepend-icon="mdi-server"
        >
          {{ store.status.device?.toUpperCase() ?? 'Backend' }}
          <template v-if="store.status.loaded_model">
            · {{ store.status.loaded_model }}
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

        <v-row>
          <!-- Left: Generation Form -->
          <v-col cols="12" md="4" lg="3">
            <GenerationForm />
          </v-col>

          <!-- Right: Image Gallery -->
          <v-col cols="12" md="8" lg="9">
            <ImageGallery />
          </v-col>
        </v-row>
      </v-container>
    </v-main>
  </v-app>
</template>

<style>
html,
body {
  margin: 0;
  padding: 0;
}
</style>
