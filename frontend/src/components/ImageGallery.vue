<script setup lang="ts">
/**
 * Image gallery component.
 * Displays generated images in a responsive grid with expandable metrics panel.
 * Shows: loading spinner during generation, empty state, or image cards.
 * Each card shows the image, its prompt, dimensions, seed, metrics, and a download button.
 */
import { ref } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import type { GeneratedImage } from '../types'

const store = useDiffusionStore()

// Track which image's detail dialog is open
const detailImage = ref<GeneratedImage | null>(null)
const showDetail = ref(false)

function openDetail(image: GeneratedImage) {
  detailImage.value = image
  showDetail.value = true
}
</script>

<template>
  <v-card class="pa-4" elevation="2">
    <v-card-title class="d-flex align-center justify-space-between">
      <span class="text-h6">
        <v-icon icon="mdi-image-multiple" class="mr-2" />
        Generated Images
        <v-chip
          v-if="store.generatedImages.length"
          size="small"
          color="primary"
          class="ml-2"
        >
          {{ store.generatedImages.length }}
        </v-chip>
      </span>
      <v-btn
        v-if="store.generatedImages.length"
        variant="text"
        color="error"
        size="small"
        prepend-icon="mdi-delete-sweep"
        @click="store.clearImages()"
      >
        Clear All
      </v-btn>
    </v-card-title>

    <v-card-text>
      <!-- Loading overlay -->
      <v-overlay
        v-if="store.isGenerating"
        contained
        class="align-center justify-center"
        :model-value="true"
      >
        <div class="text-center">
          <v-progress-circular indeterminate color="primary" size="64" />
          <div class="mt-3 text-body-1">Generating images...</div>
        </div>
      </v-overlay>

      <!-- Empty state -->
      <div
        v-if="!store.generatedImages.length && !store.isGenerating"
        class="d-flex flex-column align-center justify-center py-12 text-medium-emphasis"
      >
        <v-icon icon="mdi-image-off" size="64" class="mb-4 opacity-30" />
        <div class="text-body-1">No images yet. Fill in the form and click Generate.</div>
      </div>

      <!-- Image grid -->
      <v-row v-if="store.generatedImages.length">
        <v-col
          v-for="image in store.generatedImages"
          :key="image.id"
          cols="12"
          sm="6"
          md="4"
          lg="3"
        >
          <v-card elevation="1" class="image-card" @click="openDetail(image)">
            <v-img
              :src="image.url"
              :aspect-ratio="image.width / image.height"
              cover
              class="bg-grey-darken-3"
            >
              <template #placeholder>
                <div class="d-flex align-center justify-center fill-height">
                  <v-progress-circular indeterminate color="primary" />
                </div>
              </template>
            </v-img>

            <v-card-text class="pa-2">
              <div class="text-caption text-truncate mb-1" :title="image.prompt">
                {{ image.prompt }}
              </div>
              <div class="text-caption text-medium-emphasis">
                {{ image.width }}×{{ image.height }} · seed {{ image.seed }}
                · {{ image.generation_time_seconds }}s
              </div>
            </v-card-text>

            <v-card-actions class="pa-2 pt-0">
              <v-btn
                :href="image.url"
                target="_blank"
                download
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-download"
                @click.stop
              >
                Download
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>

  <!-- ─── Image Detail Dialog (metrics panel) ──────────────────────── -->
  <v-dialog v-model="showDetail" max-width="900">
    <v-card v-if="detailImage">
      <v-card-title class="d-flex align-center">
        <v-icon icon="mdi-chart-box" class="mr-2" />
        Generation Details
        <v-spacer />
        <v-btn icon="mdi-close" variant="text" @click="showDetail = false" />
      </v-card-title>

      <v-card-text>
        <v-row>
          <!-- Image preview -->
          <v-col cols="12" md="6">
            <v-img
              :src="detailImage.url"
              :aspect-ratio="detailImage.width / detailImage.height"
              class="rounded bg-grey-darken-3"
            />
            <div class="mt-2 text-center">
              <v-btn
                :href="detailImage.url"
                target="_blank"
                download
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-download"
              >
                Download Full Image
              </v-btn>
            </div>
          </v-col>

          <!-- Metrics panel -->
          <v-col cols="12" md="6">
            <!-- Prompt info -->
            <div class="mb-4">
              <div class="text-overline text-medium-emphasis">Prompt</div>
              <div class="text-body-2">{{ detailImage.prompt }}</div>
            </div>
            <div v-if="detailImage.negative_prompt" class="mb-4">
              <div class="text-overline text-medium-emphasis">Negative Prompt</div>
              <div class="text-body-2">{{ detailImage.negative_prompt }}</div>
            </div>

            <v-divider class="mb-4" />

            <!-- Generation metrics table -->
            <div class="text-overline text-medium-emphasis mb-2">Generation Metrics</div>
            <v-table density="compact">
              <tbody>
                <tr>
                  <td class="text-caption font-weight-bold">Model</td>
                  <td class="text-caption">{{ detailImage.model_id }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Pipeline</td>
                  <td class="text-caption">{{ detailImage.pipeline_class || '—' }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Scheduler</td>
                  <td class="text-caption">{{ detailImage.scheduler || '—' }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Resolution</td>
                  <td class="text-caption">{{ detailImage.width }} × {{ detailImage.height }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Inference Steps</td>
                  <td class="text-caption">{{ detailImage.num_inference_steps }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">CFG Scale</td>
                  <td class="text-caption">{{ detailImage.guidance_scale }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Seed</td>
                  <td class="text-caption">{{ detailImage.seed }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Generation Time</td>
                  <td class="text-caption">{{ detailImage.generation_time_seconds }}s</td>
                </tr>
                <tr v-if="detailImage.model_load_time_seconds">
                  <td class="text-caption font-weight-bold">Model Load Time</td>
                  <td class="text-caption">{{ detailImage.model_load_time_seconds }}s</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Device</td>
                  <td class="text-caption">
                    <v-chip size="x-small" :color="detailImage.device === 'cuda' ? 'success' : 'warning'">
                      {{ detailImage.device.toUpperCase() }}
                    </v-chip>
                  </td>
                </tr>
                <tr v-if="detailImage.vram_used_mb">
                  <td class="text-caption font-weight-bold">Peak VRAM</td>
                  <td class="text-caption">{{ detailImage.vram_used_mb }} MB</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Created</td>
                  <td class="text-caption">{{ detailImage.created_at }}</td>
                </tr>
              </tbody>
            </v-table>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.image-card {
  transition: transform 0.2s;
  cursor: pointer;
}
.image-card:hover {
  transform: translateY(-2px);
}
</style>
