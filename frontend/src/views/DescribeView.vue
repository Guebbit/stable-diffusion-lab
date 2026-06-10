<script setup lang="ts">
/**
 * Vision: Describe — upload an image and get a text description from an AI model.
 * Dynamically loads captioning models from the backend model store.
 */
import { ref, computed, onBeforeUnmount, onMounted } from 'vue'
import { useNotificationStore } from '../stores/notifications'
import { useModelsStore } from '../stores/models'
import api from '../api/diffusion'

const notif = useNotificationStore()
const modelsStore = useModelsStore()

const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)
const description = ref<string | null>(null)
const isProcessing = ref(false)

// Load captioning models from backend on mount
onMounted(async () => {
  await modelsStore.fetchRegistry()
  // Auto-select first captioning model if available
  if (modelsStore.captioningModels.length > 0 && !selectedModel.value) {
    selectedModel.value = modelsStore.captioningModels[0]!.model_id
  }
})

const selectedModel = ref<string>('')

const canSubmit = computed(() => !!imageFile.value && !!selectedModel.value && !isProcessing.value)

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
async function handleDescribe() {
  if (!imageFile.value || !selectedModel.value) return

  isProcessing.value = true
  description.value = null
  notif.push('info', 'Analyzing image…')

  try {
    const formData = new FormData()
    formData.append('image', imageFile.value)
    formData.append('model_id', selectedModel.value)

    const response = await api.post('/generation/describe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    const jobId = response.data.job_id
    notif.push('success', 'Description job submitted, waiting for result…')

    // Poll for job completion
    const result = await pollJob(jobId)
    if (result) {
      description.value = result
      notif.push('success', 'Image described successfully!')
    }
  } catch (error: unknown) {
    console.error('Describe failed:', error)
    const errorMessage = error instanceof Error && 'response' in error
      ? (error as any).response?.data?.detail || 'Failed to describe image'
      : 'Failed to describe image'
    notif.push('error', `Describe failed: ${errorMessage}`)
  } finally {
    isProcessing.value = false
  }
}

/** Poll a job until it completes or fails. */
async function pollJob(jobId: string, maxAttempts = 60): Promise<string | null> {
  for (let i = 0; i < maxAttempts; i++) {
    const response = await api.get(`/generation/jobs/${jobId}`)
    const job = response.data

    if (job.status === 'completed') {
      // Extract caption from job params or result
      return job.params?.caption || job.result?.caption || JSON.stringify(job)
    }

    if (job.status === 'failed' || job.status === 'error') {
      throw new Error(job.error_message || 'Job failed')
    }

    await new Promise((resolve) => setTimeout(resolve, 2000))
  }
  throw new Error('Job timed out')
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
            :items="modelsStore.captioningModels"
            item-title="name"
            item-value="model_id"
            label="Vision Model"
            variant="outlined"
            prepend-inner-icon="mdi-brain"
            class="mb-4"
            :disabled="modelsStore.captioningModels.length === 0"
          >
            <template #item="{ raw, props: itemProps }">
              <v-list-item
                v-bind="itemProps"
                :subtitle="`${raw.source} - ${raw.status}`"
              />
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