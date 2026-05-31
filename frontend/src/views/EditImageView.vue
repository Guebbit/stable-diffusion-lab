<script setup lang="ts">
/**
 * Vision: Edit — upload an image + a prompt to transform it with img2img.
 * Broad and experimental: users control all parameters freely.
 */
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import type { ModelSource } from '../types'

const store = useDiffusionStore()

// Elapsed-time counter — updates every second while a generation is running
const elapsedSeconds = ref(0)
let elapsedTimer: ReturnType<typeof setInterval> | null = null

watch(
  () => store.isGenerating,
  (generating) => {
    if (generating) {
      elapsedSeconds.value = 0
      elapsedTimer = setInterval(() => { elapsedSeconds.value++ }, 1000)
    } else {
      if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
    }
  },
)

// Model options (same as generation page)
const models = [
  { id: 'runwayml/stable-diffusion-v1-5', name: 'Stable Diffusion v1.5', source: 'huggingface' as ModelSource },
  { id: 'stabilityai/sdxl-turbo', name: 'SDXL Turbo', source: 'huggingface' as ModelSource },
  { id: 'stabilityai/stable-diffusion-xl-base-1.0', name: 'SDXL 1.0', source: 'huggingface' as ModelSource },
]

const selectedModelId = ref(models[0]!.id)
const prompt = ref('')
const negativePrompt = ref('')
const strength = ref(0.6)
const steps = ref(20)
const guidanceScale = ref(7.5)
const seed = ref<number | null>(null)
const numImages = ref(1)

const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)

const canSubmit = computed(() => !!imageFile.value && prompt.value.trim().length > 0 && !store.isGenerating)

/** Handle file selection. */
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
}

/** Submit the image + prompt to img2img endpoint. */
function handleEdit() {
  if (!imageFile.value || !prompt.value.trim()) return

  const model = models.find((m) => m.id === selectedModelId.value) ?? models[0]!

  store.generateFromImage({
    image: imageFile.value,
    prompt: prompt.value.trim(),
    negative_prompt: negativePrompt.value.trim() || undefined,
    model_id: model.id,
    model_source: model.source,
    workflow_preset: 'general',
    strength: strength.value,
    num_inference_steps: steps.value,
    guidance_scale: guidanceScale.value,
    seed: seed.value ?? undefined,
    num_images: numImages.value,
  })
}

onBeforeUnmount(() => {
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
  if (elapsedTimer) clearInterval(elapsedTimer)
})
</script>

<template>
  <v-row>
    <v-col cols="12" md="4" lg="3">
      <v-card class="pa-4" elevation="2">
        <v-card-title class="text-h5 mb-2">
          <v-icon icon="mdi-image-edit" class="mr-2" />
          Edit Image
        </v-card-title>

        <v-card-subtitle class="mb-4">
          Upload an image and describe how you want it changed. Experimental — play with settings.
        </v-card-subtitle>

        <v-card-text>
          <v-select
            v-model="selectedModelId"
            :items="models"
            item-title="name"
            item-value="id"
            label="Model"
            variant="outlined"
            prepend-inner-icon="mdi-brain"
            class="mb-4"
          />

          <v-file-input
            accept="image/*"
            label="Upload Image"
            variant="outlined"
            prepend-icon="mdi-image-plus"
            show-size
            class="mb-3"
            @update:model-value="handleImageSelection"
          />

          <v-img
            v-if="imagePreviewUrl"
            :src="imagePreviewUrl"
            max-height="200"
            contain
            class="mb-4 rounded"
          />

          <v-textarea
            v-model="prompt"
            label="Edit Prompt"
            placeholder="Describe the changes you want..."
            rows="3"
            auto-grow
            variant="outlined"
            prepend-inner-icon="mdi-pencil"
            class="mb-3"
          />

          <v-textarea
            v-model="negativePrompt"
            label="Negative Prompt"
            rows="2"
            auto-grow
            variant="outlined"
            prepend-inner-icon="mdi-pencil-off"
            class="mb-4"
          />

          <v-divider class="mb-4" />

          <!-- Parameters -->
          <div class="text-caption text-medium-emphasis mb-1">
            Strength: {{ strength }} — how much to change the original
          </div>
          <v-slider v-model="strength" :min="0.1" :max="1" :step="0.05" thumb-label color="primary" class="mb-2" />

          <div class="text-caption text-medium-emphasis mb-1">Steps: {{ steps }}</div>
          <v-slider v-model="steps" :min="1" :max="100" :step="1" thumb-label color="primary" class="mb-2" />

          <div class="text-caption text-medium-emphasis mb-1">CFG Scale: {{ guidanceScale }}</div>
          <v-slider v-model="guidanceScale" :min="1" :max="20" :step="0.5" thumb-label color="primary" class="mb-2" />

          <v-row>
            <v-col cols="6">
              <v-select v-model="numImages" :items="[1, 2, 3, 4]" label="Images" variant="outlined" />
            </v-col>
            <v-col cols="6">
              <v-text-field v-model.number="seed" label="Seed" type="number" variant="outlined" placeholder="Random" clearable />
            </v-col>
          </v-row>
        </v-card-text>

        <v-card-actions class="pa-4 pt-0">
          <v-btn
            color="primary"
            size="large"
            :loading="store.isGenerating"
            :disabled="!canSubmit"
            prepend-icon="mdi-auto-fix"
            block
            @click="handleEdit"
          >
            {{ store.isGenerating ? 'Editing...' : 'Edit Image' }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="8" lg="9">
      <!-- Indeterminate progress bar visible while generating -->
      <v-progress-linear
        v-if="store.isGenerating"
        indeterminate
        color="primary"
        class="mb-3"
      />

      <!-- Error alert -->
      <v-alert
        v-if="store.error"
        type="error"
        closable
        class="mb-3"
        @click:close="store.clearError()"
      >
        {{ store.error }}
      </v-alert>

      <v-card class="pa-4" elevation="2" style="position: relative; min-height: 200px;">
        <!-- Loading overlay with spinner + elapsed time -->
        <v-overlay
          v-if="store.isGenerating"
          contained
          :model-value="true"
          class="align-center justify-center"
        >
          <div class="text-center">
            <v-progress-circular indeterminate color="primary" size="64" />
            <div class="mt-3 text-body-1">Editing image…</div>
            <div class="mt-1 text-caption text-medium-emphasis">
              {{ elapsedSeconds }}s elapsed — this can take a while
            </div>
          </div>
        </v-overlay>

        <template v-if="store.generatedImages.length">
          <v-card-title class="text-h6 mb-2">Results</v-card-title>
          <v-row>
            <v-col v-for="img in store.generatedImages" :key="img.id" cols="12" sm="6" md="4">
              <v-card elevation="1" class="image-card">
                <v-img :src="img.url" :aspect-ratio="img.width / img.height" cover class="bg-grey-darken-3">
                  <template #placeholder>
                    <div class="d-flex align-center justify-center fill-height">
                      <v-progress-circular indeterminate color="primary" />
                    </div>
                  </template>
                </v-img>

                <v-card-text class="pa-2">
                  <div class="text-caption text-truncate mb-1" :title="img.prompt">
                    {{ img.prompt }}
                  </div>
                  <div class="text-caption text-medium-emphasis">
                    {{ img.width }}×{{ img.height }} · seed {{ img.seed }}
                  </div>
                </v-card-text>

                <v-card-actions class="pa-2 pt-0">
                  <v-btn
                    :href="img.url"
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
        </template>

        <div
          v-else-if="!store.isGenerating"
          class="d-flex flex-column align-center justify-center py-12 text-medium-emphasis"
        >
          <v-icon icon="mdi-image-edit-outline" size="64" class="mb-4 opacity-30" />
          <div class="text-body-1">
            Upload an image, describe your edits, and hit "Edit Image" to see results.
          </div>
        </div>
      </v-card>
    </v-col>
  </v-row>
</template>

<style scoped>
.image-card {
  transition: transform 0.2s;
}
.image-card:hover {
  transform: translateY(-2px);
}
</style>
