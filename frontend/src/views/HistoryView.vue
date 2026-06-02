<script setup lang="ts">
/**
 * History view — shows all past generations that were persisted to disk.
 * Images survive container restarts; users can inspect details or delete entries.
 * Layout mirrors ImageGallery but pulls data from the history store instead.
 */
import { ref, onMounted } from 'vue'
import { useHistoryStore } from '../stores/history'
import type { ArtifactEntry } from '../types'

const store = useHistoryStore()

// Detail dialog state — which image is being inspected
const detailImage = ref<ArtifactEntry | null>(null)
const showDetail = ref(false)

/** Open the detail dialog for the selected image. */
function openDetail(image: ArtifactEntry) {
  detailImage.value = image
  showDetail.value = true
}

/** Delete the image shown in the detail dialog, closing the panel first. */
function confirmDelete(image: ArtifactEntry) {
  // Close the detail panel first so the deleted card doesn't linger
  showDetail.value = false
  store.deleteEntry(image.id)
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
              :src="image.file_path"
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
                :href="image.file_path"
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
              :src="detailImage.file_path"
              :aspect-ratio="detailImage.width / detailImage.height"
              class="rounded bg-grey-darken-3"
            />
            <div class="mt-2 text-center d-flex justify-center gap-2">
              <v-btn
                :href="detailImage.file_path"
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
                  <td class="text-caption">{{ detailImage.model_name || detailImage.model_id_ref || '—' }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Resolution</td>
                  <td class="text-caption">{{ detailImage.width }} × {{ detailImage.height }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Seed</td>
                  <td class="text-caption">{{ detailImage.seed }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">Media Type</td>
                  <td class="text-caption">{{ detailImage.media_type }}</td>
                </tr>
                <tr>
                  <td class="text-caption font-weight-bold">File Size</td>
                  <td class="text-caption">{{ Math.round(detailImage.size_bytes / 1024) }} KB</td>
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
.history-card {
  transition: transform 0.2s;
  cursor: pointer;
}
.history-card:hover {
  transform: translateY(-2px);
}
</style>
