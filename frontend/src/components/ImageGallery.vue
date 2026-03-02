<script setup lang="ts">
import { useDiffusionStore } from '../stores/diffusion'

const store = useDiffusionStore()
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
          <v-card elevation="1" class="image-card">
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
              >
                Download
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>
</template>

<style scoped>
.image-card {
  transition: transform 0.2s;
}
.image-card:hover {
  transform: translateY(-2px);
}
</style>
