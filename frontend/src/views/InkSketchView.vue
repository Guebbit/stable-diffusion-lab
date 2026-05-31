<script setup lang="ts">
/**
 * Vision: Ink Sketch — strict workflow for inking hand-drawn sketches.
 * Uses ControlNet scribble pipeline with locked-down parameters.
 * Only HuggingFace SD 1.5/SDXL models supported.
 */
import { ref, computed, onBeforeUnmount } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'

const store = useDiffusionStore()

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
      <v-card v-if="store.generatedImages.length" class="pa-4" elevation="2">
        <v-card-title class="text-h6 mb-2">Inked Result</v-card-title>
        <v-row>
          <v-col v-for="img in store.generatedImages" :key="img.id" cols="12" sm="6">
            <v-img :src="img.url" aspect-ratio="1" contain class="rounded" />
          </v-col>
        </v-row>
      </v-card>

      <v-card v-else class="pa-6 text-center" elevation="1" color="surface-variant">
        <v-icon icon="mdi-draw" size="64" class="mb-4 text-medium-emphasis" />
        <div class="text-body-1 text-medium-emphasis">
          Upload a sketch and click "Ink Sketch" to get clean line art output.
        </div>
      </v-card>
    </v-col>
  </v-row>
</template>
