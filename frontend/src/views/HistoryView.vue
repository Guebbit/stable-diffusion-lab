<script setup lang="ts">
/**
 * History view — shows all past generations that were persisted to disk.
 * Images survive container restarts; users can inspect details or delete entries.
 * Layout mirrors ImageGallery but pulls data from the history store instead.
 */
import { ref, onMounted } from 'vue'
import { useHistoryStore } from '../stores/history'
import type { GeneratedImage } from '../types'

const store = useHistoryStore()

// Detail dialog state — which image is being inspected
const detailImage = ref<GeneratedImage | null>(null)
const showDetail = ref(false)

// Confirmation dialog before wiping everything
const showClearConfirm = ref(false)

/** Open the detail dialog for the selected image. */
function openDetail(image: GeneratedImage) {
  detailImage.value = image
  showDetail.value = true
}

/** Delete the image shown in the detail dialog, closing the panel first. */
function confirmDelete(image: GeneratedImage) {
  // Close the detail panel first so the deleted card doesn't linger
  showDetail.value = false
  store.deleteEntry(image.id)
}

/** Dismiss the clear-all confirmation dialog and execute the wipe. */
function confirmClearAll() {
  showClearConfirm.value = false
  store.clearAll()
}

// Load history when the page mounts
onMounted(() => {
  store.fetchHistory()
})
</script>

<template>
  <v-card class="pa-4" elevation="2">
    <v-card-title class="d-flex align-center justify-space-between">
      <span class="text-h6">
        <v-icon icon="mdi-history" class="mr-2" />
        Generation History
        <v-chip
          v-if="store.images.length"
          size="small"
          color="primary"
          class="ml-2"
        >
          {{ store.images.length }}
        </v-chip>
      </span>
      <div class="d-flex gap-2">
        <!-- Refresh button — re-fetches from disk (useful if backend added images elsewhere) -->
        <v-btn
          variant="text"
          size="small"
          prepend-icon="mdi-refresh"
          :loading="store.isLoading"
          @click="store.fetchHistory()"
        >
          Refresh
        </v-btn>
        <!-- Wipe everything button -->
        <v-btn
          v-if="store.images.length"
          variant="text"
          color="error"
          size="small"
          prepend-icon="mdi-delete-sweep"
          @click="showClearConfirm = true"
        >
          Clear All
        </v-btn>
      </div>
    </v-card-title>

    <v-card-text>
      <!-- Loading spinner while fetching -->
      <div v-if="store.isLoading" class="d-flex justify-center py-12">
        <v-progress-circular indeterminate color="primary" size="64" />
      </div>

      <!-- Empty state -->
      <div
        v-else-if="!store.images.length"
        class="d-flex flex-column align-center justify-center py-12 text-medium-emphasis"
      >
        <v-icon icon="mdi-image-off-outline" size="64" class="mb-4 opacity-30" />
        <div class="text-body-1">No history yet — generate some images first.</div>
        <div class="text-caption mt-1">Generated images are saved automatically and survive restarts.</div>
      </div>

      <!-- Image grid -->
      <v-row v-else>
        <v-col
          v-for="image in store.images"
          :key="image.id"
          cols="12"
          sm="6"
          md="4"
          lg="3"
        >
          <v-card elevation="1" class="history-card" @click="openDetail(image)">
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
              </div>
              <div class="text-caption text-medium-emphasis">
                {{ image.created_at }}
              </div>
            </v-card-text>

            <v-card-actions class="pa-2 pt-0">
              <!-- Download the image directly (no extra round-trip) -->
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
              <v-spacer />
              <!-- Quick delete without opening the detail dialog -->
              <v-btn
                size="small"
                variant="text"
                color="error"
                icon="mdi-delete-outline"
                @click.stop="store.deleteEntry(image.id)"
              />
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>

  <!-- ─── Image Detail Dialog ─────────────────────────────────────────── -->
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
            <div class="mt-2 text-center d-flex justify-center gap-2">
              <v-btn
                :href="detailImage.url"
                target="_blank"
                download
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-download"
              >
                Download
              </v-btn>
              <!-- Permanent delete from detail view -->
              <v-btn
                size="small"
                variant="tonal"
                color="error"
                prepend-icon="mdi-delete-outline"
                @click="confirmDelete(detailImage)"
              >
                Delete
              </v-btn>
            </div>
          </v-col>

          <!-- Metrics panel -->
          <v-col cols="12" md="6">
            <div class="mb-4">
              <div class="text-overline text-medium-emphasis">Prompt</div>
              <div class="text-body-2">{{ detailImage.prompt }}</div>
            </div>
            <div v-if="detailImage.negative_prompt" class="mb-4">
              <div class="text-overline text-medium-emphasis">Negative Prompt</div>
              <div class="text-body-2">{{ detailImage.negative_prompt }}</div>
            </div>

            <v-divider class="mb-4" />

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

  <!-- ─── Confirm clear-all dialog ────────────────────────────────────── -->
  <v-dialog v-model="showClearConfirm" max-width="400">
    <v-card>
      <v-card-title>
        <v-icon icon="mdi-alert-outline" color="error" class="mr-2" />
        Clear all history?
      </v-card-title>
      <v-card-text>
        This will permanently delete <strong>all {{ store.images.length }} saved images</strong>
        from disk. This cannot be undone.
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="showClearConfirm = false">Cancel</v-btn>
        <v-btn color="error" variant="tonal" prepend-icon="mdi-delete-sweep" @click="confirmClearAll">
          Delete all
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.history-card {
  transition: transform 0.2s;
  cursor: pointer;
}
.history-card:hover {
  transform: translateY(-2px);
}
</style>
