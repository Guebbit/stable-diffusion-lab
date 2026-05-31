<script setup lang="ts">
/**
 * Vision: Ink Sketch — strict workflow for inking hand-drawn sketches.
 * Uses ControlNet scribble pipeline with locked-down parameters.
 * Only HuggingFace SD 1.5/SDXL models supported.
 */
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'

const store = useDiffusionStore()

// Elapsed-time counter — updates every second while a generation is running
const elapsedSeconds = ref(0)
// Holds the setInterval ID so we can stop the timer on completion or unmount
let elapsedTimer: ReturnType<typeof setInterval> | null = null

// Start/stop the timer based on generation state
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

// Only HuggingFace models work with ControlNet
const models = [
  { id: 'runwayml/stable-diffusion-v1-5', name: 'Stable Diffusion v1.5' },
  { id: 'stabilityai/stable-diffusion-xl-base-1.0', name: 'SDXL 1.0' },
]

// Strict defaults — these are locked for consistent inking results
const PROMPT = 'clean black ink line art, crisp comic inking, bold outlines, high contrast, white background'
const NEGATIVE_PROMPT = 'color, shading, painterly, blurry, messy sketch lines, grayscale wash, textured paper'
const STEPS = 28
const GUIDANCE_SCALE = 8.0
const CONTROLNET_SCALE = 1.1

const selectedModelId = ref(models[0]!.id)
const controlnetScale = ref(CONTROLNET_SCALE)
const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)

const canSubmit = computed(() => !!imageFile.value && !store.isGenerating)

/** Handle sketch file selection. */
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

/** Submit the sketch to the ControlNet inking pipeline. */
function handleInk() {
  if (!imageFile.value) return

  store.generateSketchToInk({
    image: imageFile.value,
    prompt: PROMPT,
    negative_prompt: NEGATIVE_PROMPT,
    model_id: selectedModelId.value,
    model_source: 'huggingface',
    controlnet_conditioning_scale: controlnetScale.value,
    num_inference_steps: STEPS,
    guidance_scale: GUIDANCE_SCALE,
    num_images: 1,
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
          <v-icon icon="mdi-draw-pen" class="mr-2" />
          Ink Sketch
        </v-card-title>

        <v-card-subtitle class="mb-4">
          Upload a hand-drawn sketch and get clean ink line art. Strict pipeline — optimized for inking.
        </v-card-subtitle>

        <v-card-text>
          <v-alert type="info" variant="tonal" density="compact" class="mb-4">
            This uses a ControlNet scribble pipeline with locked parameters for consistent inking results.
            Only the model and sketch guidance can be adjusted.
          </v-alert>

          <v-select
            v-model="selectedModelId"
            :items="models"
            item-title="name"
            item-value="id"
            label="Base Model"
            variant="outlined"
            prepend-inner-icon="mdi-brain"
            class="mb-4"
          />

          <v-file-input
            accept="image/*"
            label="Upload Sketch"
            variant="outlined"
            prepend-icon="mdi-draw"
            show-size
            class="mb-3"
            @update:model-value="handleImageSelection"
          />

          <v-img
            v-if="imagePreviewUrl"
            :src="imagePreviewUrl"
            max-height="240"
            contain
            class="mb-4 rounded"
          />

          <v-divider class="mb-4" />

          <div class="text-caption text-medium-emphasis mb-1">
            Sketch Guidance: {{ controlnetScale }} — how strictly to follow sketch lines
          </div>
          <v-slider
            v-model="controlnetScale"
            :min="0.5"
            :max="2.0"
            :step="0.05"
            thumb-label
            color="primary"
            class="mb-2"
          />

          <!-- Show locked parameters as read-only info -->
          <v-list density="compact" class="bg-surface-variant rounded pa-2">
            <v-list-item prepend-icon="mdi-lock">
              <v-list-item-title class="text-caption">Steps: {{ STEPS }}</v-list-item-title>
            </v-list-item>
            <v-list-item prepend-icon="mdi-lock">
              <v-list-item-title class="text-caption">CFG: {{ GUIDANCE_SCALE }}</v-list-item-title>
            </v-list-item>
            <v-list-item prepend-icon="mdi-lock">
              <v-list-item-title class="text-caption text-truncate">Prompt: {{ PROMPT }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>

        <v-card-actions class="pa-4 pt-0">
          <v-btn
            color="primary"
            size="large"
            :loading="store.isGenerating"
            :disabled="!canSubmit"
            prepend-icon="mdi-draw-pen"
            block
            @click="handleInk"
          >
            {{ store.isGenerating ? 'Inking...' : 'Ink Sketch' }}
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
            <div class="mt-3 text-body-1">Inking sketch…</div>
            <div class="mt-1 text-caption text-medium-emphasis">
              {{ elapsedSeconds }}s elapsed — this can take a while
            </div>
          </div>
        </v-overlay>

        <template v-if="store.generatedImages.length">
          <v-card-title class="text-h6 mb-2">Inked Result</v-card-title>
          <v-row>
            <v-col v-for="img in store.generatedImages" :key="img.id" cols="12" sm="6">
              <v-card elevation="1" class="image-card">
                <v-img :src="img.url" :aspect-ratio="img.width / img.height" contain class="bg-grey-darken-3">
                  <template #placeholder>
                    <div class="d-flex align-center justify-center fill-height">
                      <v-progress-circular indeterminate color="primary" />
                    </div>
                  </template>
                </v-img>

                <v-card-text class="pa-2">
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
          <v-icon icon="mdi-draw" size="64" class="mb-4 opacity-30" />
          <div class="text-body-1">
            Upload a sketch and click "Ink Sketch" to get clean line art output.
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
