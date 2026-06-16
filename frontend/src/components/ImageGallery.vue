<script setup lang="ts">
/**
 * Image gallery component.
 * Displays generated images in a responsive grid with expandable metrics panel.
 * Shows: loading spinner during generation, empty state, or image cards.
 * Each card shows the image, its prompt, dimensions, seed, metrics, and a download button.
 */
import { ref, onMounted } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import type { ArtifactEntry } from '../types'

const store = useDiffusionStore()

// Load gallery from backend on mount so images survive page navigation
onMounted(() => {
  store.refreshGallery()
})

// Currently displayed image in the metrics detail dialog
const detailImage = ref<ArtifactEntry | null>(null)
const showDetail = ref(false)

// Confirm dialog for "Clear All"
const showClearConfirm = ref(false)

function openDetail(image: ArtifactEntry) {
  detailImage.value = image
  showDetail.value = true
}

async function deleteImage(image: ArtifactEntry) {
  await store.deleteImage(image.id)
  if (detailImage.value?.id === image.id) {
    showDetail.value = false
    detailImage.value = null
  }
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
        @click="showClearConfirm = true"
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
              :src="image.file_path"
              :aspect-ratio="image.width / image.height || 1"
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
            </v-card-text>

            <v-card-actions class="pa-2 pt-0">
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
              <v-btn
                icon="mdi-delete"
                size="small"
                variant="text"
                color="error"
                @click.stop="deleteImage(image)"
              />
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>

  <!-- ─── Clear All Confirmation ───────────────────────────────────── -->
  <v-dialog v-model="showClearConfirm" max-width="420" persistent>
    <v-card>
      <v-card-title class="d-flex align-center gap-2 pt-4 pb-1">
        <v-icon icon="mdi-delete-sweep" color="error" />
        Clear all images?
      </v-card-title>
      <v-card-text class="pb-2">
        <v-alert type="error" variant="tonal" density="compact" class="text-body-2">
          This permanently deletes all {{ store.generatedImages.length }} generated images from disk and database. This cannot be undone.
        </v-alert>
      </v-card-text>
      <v-card-actions class="pa-4 pt-2">
        <v-spacer />
        <v-btn variant="text" @click="showClearConfirm = false">Cancel</v-btn>
        <v-btn color="error" variant="elevated" @click="showClearConfirm = false; store.clearImages()">
          Delete all
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

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
              :src="detailImage.file_path"
              :aspect-ratio="detailImage.width / detailImage.height || 1"
              class="rounded bg-grey-darken-3"
            />
            <div class="mt-2 d-flex justify-center gap-2">
              <v-btn
                :href="detailImage.file_path"
                target="_blank"
                download
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-download"
              >
                Download Full Image
              </v-btn>
              <v-btn
                size="small"
                variant="tonal"
                color="error"
                prepend-icon="mdi-delete"
                @click="deleteImage(detailImage)"
              >
                Delete
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
.image-card {
  transition: transform 0.2s;
  cursor: pointer;
}
.image-card:hover {
  transform: translateY(-2px);
}
</style>
