<script setup lang="ts">
/**
 * Vision: Describe — upload an image and get a text description from an AI model.
 * Broad and experimental: users can pick different vision models.
 */
import { ref, computed, onBeforeUnmount } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import { useNotificationStore } from '../stores/notifications'
import { diffusionApi } from '../api/diffusion'

const store = useDiffusionStore()
const notif = useNotificationStore()

// Available vision models for captioning
const visionModels = [
  { id: 'Salesforce/blip-image-captioning-large', name: 'BLIP Large', description: 'General image captioning' },
  { id: 'Salesforce/blip2-opt-2.7b', name: 'BLIP-2 (2.7B)', description: 'More detailed descriptions' },
  { id: 'nlpconnect/vit-gpt2-image-captioning', name: 'ViT-GPT2', description: 'Lightweight captioning' },
]

const selectedModel = ref(visionModels[0]!.id)
const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)
const description = ref<string | null>(null)
const isProcessing = ref(false)

const canSubmit = computed(() => !!imageFile.value && !isProcessing.value)

/** Handle file selection and generate a local preview URL. */
function handleImageSelection(value: File | File[] | null) {
  const selected = Array.isArray(value) ? value[0] ?? null : value
  imageFile.value = selected

  if (imagePreviewUrl.value) {
    URL.revokeObjectURL(imagePreviewUrl.value)
    imagePreviewUrl.value = null
  }

  if (selected) {
    imagePreviewUrl.value = URL.createObjectURL(selected)
  }

  // Reset previous result when new image is selected
  description.value = null
}

/** Send the image to the backend vision endpoint. */
function handleDescribe() {
  if (!imageFile.value) return

  isProcessing.value = true
  description.value = null
  notif.push('info', 'Analyzing image…')

  diffusionApi.describeImage({ image: imageFile.value, model_id: selectedModel.value })
    .then((res) => {
      description.value = res.description
      notif.push('success', 'Image described successfully')
    })
    .catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : 'Failed to describe image'
      store.error = msg
      notif.push('error', msg)
    })
    .finally(() => {
      isProcessing.value = false
    })
}

onBeforeUnmount(() => {
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
})
</script>

<template>
  <v-row>
    <v-col cols="12" md="5" lg="4">
      <v-card class="pa-4" elevation="2">
        <v-card-title class="text-h5 mb-2">
          <v-icon icon="mdi-eye" class="mr-2" />
          Describe Image
        </v-card-title>

        <v-card-subtitle class="mb-4">
          Upload an image and let the AI tell you what it sees. Experimental — try different models.
        </v-card-subtitle>

        <v-card-text>
          <v-select
            v-model="selectedModel"
            :items="visionModels"
            item-title="name"
            item-value="id"
            label="Vision Model"
            variant="outlined"
            prepend-inner-icon="mdi-brain"
            class="mb-4"
          >
            <template #item="{ item, props: itemProps }">
              <v-list-item v-bind="itemProps" :subtitle="item.raw.description" />
            </template>
          </v-select>

          <v-file-input
            accept="image/*"
            label="Upload Image"
            variant="outlined"
            prepend-icon="mdi-image-plus"
            show-size
            @update:model-value="handleImageSelection"
          />

          <v-img
            v-if="imagePreviewUrl"
            :src="imagePreviewUrl"
            max-height="300"
            contain
            class="mt-2 mb-4 rounded"
          />
        </v-card-text>

        <v-card-actions class="pa-4 pt-0">
          <v-btn
            color="primary"
            size="large"
            :loading="isProcessing"
            :disabled="!canSubmit"
            prepend-icon="mdi-text-search"
            block
            @click="handleDescribe"
          >
            {{ isProcessing ? 'Analyzing...' : 'Describe' }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="7" lg="8">
      <v-card v-if="description" class="pa-4" elevation="2">
        <v-card-title class="text-h6">
          <v-icon icon="mdi-text-box-outline" class="mr-2" />
          Result
        </v-card-title>
        <v-card-text class="text-body-1">
          {{ description }}
        </v-card-text>
      </v-card>

      <v-card v-else class="pa-6 text-center" elevation="1" color="surface-variant">
        <v-icon icon="mdi-image-search-outline" size="64" class="mb-4 text-medium-emphasis" />
        <div class="text-body-1 text-medium-emphasis">
          Upload an image and click "Describe" to get an AI-generated description.
        </div>
      </v-card>
    </v-col>
  </v-row>
</template>
