<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import { useModelsStore } from '../stores/models'
import type { GenerationMode, ModelSource } from '../types'

const store = useDiffusionStore()
const modelsStore = useModelsStore()

onMounted(() => {
  modelsStore.fetchRegistry()
})

const modelSourceSelection = ref<ModelSource>('huggingface')
const modelSourceItems = [
  { title: 'HuggingFace', value: 'huggingface' as ModelSource },
  { title: 'CivitAI', value: 'civitai' as ModelSource },
]

const generationMode = ref<GenerationMode>('text-to-image')
const generationModeItems = [
  { title: 'Text to Image', value: 'text-to-image' as GenerationMode, icon: 'mdi-text-box-plus' },
  { title: 'Image to Image', value: 'image-to-image' as GenerationMode, icon: 'mdi-image-filter-center-focus' },
  { title: 'Upscale', value: 'upscale-image' as GenerationMode, icon: 'mdi-arrow-expand-all' },
  { title: 'Describe', value: 'describe-image' as GenerationMode, icon: 'mdi-text-recognition' },
  { title: 'Recolor', value: 'recolor-image' as GenerationMode, icon: 'mdi-palette' },
  { title: 'Sketch to Ink', value: 'sketch-to-ink' as GenerationMode, icon: 'mdi-pencil' },
]

const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)
const useCustomModel = ref(false)

interface IForm {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  numInferenceSteps: number
  guidanceScale: number
  strength: number
  controlnetConditioningScale: number
  seed: number | null
  numImages: number
}

const defaultForm: IForm = {
  prompt: '',
  negativePrompt: '',
  width: 512,
  height: 512,
  numInferenceSteps: 20,
  guidanceScale: 7.5,
  strength: 0.6,
  controlnetConditioningScale: 1.1,
  seed: null,
  numImages: 1,
}

const form = ref<IForm>({ ...defaultForm })
const dimensionOptions = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

// Mode metadata: which fields each mode needs
interface ModeMeta {
  needsPrompt: boolean
  needsImage: boolean
  needsNegativePrompt: boolean
  needsStrength: boolean
  needsControlnetScale: boolean
  defaultSteps: number
  defaultGuidance: number
  defaultStrength: number
  helperText: string
}

const modeMeta: Record<GenerationMode, ModeMeta> = {
  'text-to-image': {
    needsPrompt: true,
    needsImage: false,
    needsNegativePrompt: true,
    needsStrength: false,
    needsControlnetScale: false,
    defaultSteps: 20,
    defaultGuidance: 7.5,
    defaultStrength: 0.6,
    helperText: 'Generate an image from a text description.',
  },
  'image-to-image': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: true,
    needsControlnetScale: false,
    defaultSteps: 20,
    defaultGuidance: 7.5,
    defaultStrength: 0.6,
    helperText: 'Transform an existing image using a text prompt.',
  },
  'upscale-image': {
    needsPrompt: false,
    needsImage: true,
    needsNegativePrompt: false,
    needsStrength: true,
    needsControlnetScale: false,
    defaultSteps: 18,
    defaultGuidance: 6.5,
    defaultStrength: 0.3,
    helperText: 'Increase resolution and add detail to an image.',
  },
  'describe-image': {
    needsPrompt: false,
    needsImage: true,
    needsNegativePrompt: false,
    needsStrength: false,
    needsControlnetScale: false,
    defaultSteps: 1,
    defaultGuidance: 1,
    defaultStrength: 0,
    helperText: 'Generate a text description of an uploaded image.',
  },
  'recolor-image': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: true,
    needsControlnetScale: false,
    defaultSteps: 24,
    defaultGuidance: 7,
    defaultStrength: 0.45,
    helperText: 'Change the colors of an image while preserving shapes.',
  },
  'sketch-to-ink': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: false,
    needsControlnetScale: true,
    defaultSteps: 28,
    defaultGuidance: 8,
    defaultStrength: 0.6,
    helperText: 'Convert a sketch or scribble into clean ink line art.',
  },
}

const currentModeMeta = computed(() => modeMeta[generationMode.value])
const availableModels = computed(() =>
  modelSourceSelection.value === 'huggingface'
    ? modelsStore.huggingfaceModels
    : modelsStore.civitaiModels,
)
const isSketchToInkMode = computed(() => generationMode.value === 'sketch-to-ink')
const needsImage = computed(() => currentModeMeta.value.needsImage)
const needsPrompt = computed(() => currentModeMeta.value.needsPrompt)

const availableModelSourceItems = computed(() =>
  isSketchToInkMode.value ? [modelSourceItems[0]] : modelSourceItems,
)

const modelIdSelected = ref('')
const customModelId = ref('')
const activeModelId = computed(() =>
  useCustomModel.value ? customModelId.value : modelIdSelected.value,
)

const formValid = computed(
  () =>
    (needsPrompt.value ? form.value.prompt.trim().length > 0 : true) &&
    activeModelId.value.trim().length > 0 &&
    (!needsImage.value || !!imageFile.value),
)

function handleGenerationModeChange(mode: GenerationMode) {
  const meta = modeMeta[mode]
  form.value.numInferenceSteps = meta.defaultSteps
  form.value.guidanceScale = meta.defaultGuidance
  form.value.strength = meta.defaultStrength

  if (mode === 'sketch-to-ink') {
    modelSourceSelection.value = 'huggingface'
    useCustomModel.value = false
    modelIdSelected.value = modelsStore.huggingfaceModels[0]?.id ?? ''
    if (!form.value.prompt.trim()) {
      form.value.prompt = 'clean black ink line art, crisp comic inking, bold outlines, high contrast, white background'
    }
    if (!form.value.negativePrompt.trim()) {
      form.value.negativePrompt = 'color, shading, painterly, blurry, messy sketch lines, grayscale wash, textured paper'
    }
    return
  }

  if (mode === 'upscale-image' && !form.value.prompt.trim()) {
    form.value.prompt = 'highly detailed upscale, sharp edges, clean textures'
  }
  if (mode === 'image-to-image' && !form.value.prompt.trim()) {
    form.value.prompt = 'refine details, preserve composition, clean textures'
  }
  if (mode === 'recolor-image' && !form.value.prompt.trim()) {
    form.value.prompt = 'recolor with cohesive palette, preserve shapes and edges'
  }
}

function buildCommonPayload() {
  return {
    prompt: (needsPrompt.value ? form.value.prompt : '').trim(),
    negative_prompt: (needsPrompt.value && form.value.negativePrompt.trim()) ? form.value.negativePrompt.trim() : undefined,
    model_id: activeModelId.value,
  }
}

function handleGenerate() {
  if (!formValid.value) return

  const mode = generationMode.value
  const modelId = activeModelId.value

  // Modes that require an image file
  if (mode === 'describe-image') {
    if (!imageFile.value) return
    return store.describe(modelId, imageFile.value)
  }
  if (mode === 'upscale-image') {
    if (!imageFile.value) return
    // Use strength as scale factor (map 0.1-1.0 to 1.5-4.0)
    const scaleFactor = form.value.strength * 4 + 0.5
    return store.upscale(modelId, imageFile.value!, scaleFactor)
  }
  if (mode === 'recolor-image') {
    if (!imageFile.value) return
    return store.recolor(modelId, imageFile.value, form.value.prompt, form.value.strength)
  }
  if (mode === 'sketch-to-ink') {
    if (!imageFile.value) return
    return store.sketchToInk(modelId, imageFile.value)
  }
  if (mode === 'image-to-image') {
    const payload = {
      ...buildCommonPayload(),
      model_source: modelSourceSelection.value,
      width: form.value.width,
      height: form.value.height,
      num_inference_steps: form.value.numInferenceSteps,
      guidance_scale: form.value.guidanceScale,
      seed: form.value.seed ?? undefined,
      num_images: form.value.numImages,
    }
    return store.imageToImage(payload)
  }

  // Default: text-to-image
  const payload = {
    ...buildCommonPayload(),
    model_source: modelSourceSelection.value,
    width: form.value.width,
    height: form.value.height,
    num_inference_steps: form.value.numInferenceSteps,
    guidance_scale: form.value.guidanceScale,
    seed: form.value.seed ?? undefined,
    num_images: form.value.numImages,
  }
  return store.generate(payload)
}

const isCurrentModelLoaded = computed(
  () =>
    !useCustomModel.value &&
    activeModelId.value.trim().length > 0 &&
    store.status?.models?.active_model === activeModelId.value,
)

function loadSelectedModel() {}

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

onBeforeUnmount(() => {
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
})
</script>

<template>
  <v-card class="pa-4" elevation="2">
    <v-card-title class="text-h5 mb-2">
      <v-icon icon="mdi-image-sparkle" class="mr-2"/>
      Image Generation
    </v-card-title>

    <v-card-text>
      <!-- Mode Selector -->
      <v-select
          v-model="generationMode"
          :items="generationModeItems"
          label="Generation Mode"
          variant="outlined"
          prepend-inner-icon="mdi-tune-variant"
          class="mb-4"
          item-title="title"
          item-value="value"
          @update:model-value="handleGenerationModeChange"
      />

      <!-- Helper text -->
      <v-alert
          :type="isSketchToInkMode ? 'info' : 'success'"
          variant="tonal"
          density="compact"
          class="mb-4"
      >
        {{ currentModeMeta.helperText }}
      </v-alert>

      <!-- Prompt (modes that need it) -->
      <v-textarea
          v-if="needsPrompt"
          v-model="form.prompt"
          label="Prompt"
          rows="3"
          auto-grow
          variant="outlined"
          prepend-inner-icon="mdi-pencil"
          class="mb-3"
      />

      <!-- Negative Prompt (modes that need it) -->
      <v-textarea
          v-if="currentModeMeta.needsNegativePrompt"
          v-model="form.negativePrompt"
          label="Negative Prompt"
          rows="2"
          auto-grow
          variant="outlined"
          prepend-inner-icon="mdi-pencil-off"
          class="mb-4"
      />

      <!-- Image Upload (modes that need it) -->
      <div v-if="needsImage" class="mb-4">
        <v-file-input
            accept="image/*"
            :label="isSketchToInkMode ? 'Sketch Upload' : 'Input Image'"
            variant="outlined"
            prepend-icon="mdi-image-plus"
            show-size
            @update:model-value="handleImageSelection"
        />
        <v-img
            v-if="imagePreviewUrl"
            :src="imagePreviewUrl"
            max-height="240"
            cover
            class="mt-2 rounded"
        />
      </div>

      <v-divider class="mb-4"/>

      <!-- Model Source + Model -->
      <v-row>
        <v-col cols="12">
          <v-select
              v-model="modelSourceSelection"
              :items="availableModelSourceItems"
              label="Model Source"
              variant="outlined"
              prepend-inner-icon="mdi-database"
              @update:model-value="() => modelIdSelected = availableModels[0]?.id ?? ''"
          />
        </v-col>

        <v-col cols="12">
          <v-select
              v-if="!useCustomModel"
              v-model="modelIdSelected"
              :items="availableModels"
              item-title="name"
              item-value="id"
              label="Model"
              variant="outlined"
              prepend-inner-icon="mdi-brain"
          >
            <template #item="{ item, props: itemProps }">
              <v-list-item v-bind="itemProps" :subtitle="item.raw.description"/>
            </template>
          </v-select>
          <v-text-field
              v-else
              v-model="customModelId"
              :label="modelSourceSelection === 'huggingface' ? 'HuggingFace repo ID' : 'CivitAI model ID'"
              :placeholder="
              modelSourceSelection === 'huggingface'
                ? 'e.g. runwayml/stable-diffusion-v1-5'
                : 'e.g. 4201'
          "
              variant="outlined"
              prepend-inner-icon="mdi-identifier"
          />
        </v-col>
      </v-row>

      <v-checkbox
          v-model="useCustomModel"
          label="Enter custom model ID"
          density="compact"
          class="mb-2"
      />

      <v-divider class="mb-4"/>

      <!-- Generation Parameters -->
      <v-row>
        <v-col cols="6">
          <v-select
              v-model="form.width"
              :items="dimensionOptions"
              label="Width"
              variant="outlined"
              suffix="px"
          />
        </v-col>
        <v-col cols="6">
          <v-select
              v-model="form.height"
              :items="dimensionOptions"
              label="Height"
              variant="outlined"
              suffix="px"
          />
        </v-col>
        <v-col cols="6">
          <v-select
              v-model="form.numImages"
              :items="[1, 2, 3, 4, 6, 8]"
              label="Images"
              variant="outlined"
          />
        </v-col>
        <v-col cols="6">
          <v-text-field
              v-model.number="form.seed"
              label="Seed"
              type="number"
              variant="outlined"
              placeholder="Random"
              clearable
          />
        </v-col>
      </v-row>

      <v-row>
        <v-col cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            Steps: {{ form.numInferenceSteps }}
            <p><small>inference steps for drawing conclusions from evidence.</small></p>
          </div>
          <v-slider
              v-model="form.numInferenceSteps"
              :min="1"
              :max="150"
              :step="1"
              thumb-label
              color="primary"
          />
        </v-col>
        <v-col cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            CFG Scale: {{ form.guidanceScale }}
            <p><small>higher = stricter prompt adherence, lower = more creative.</small></p>
          </div>
          <v-slider
              v-model="form.guidanceScale"
              :min="1"
              :max="30"
              :step="0.5"
              thumb-label
              color="primary"
          />
        </v-col>
        <v-col v-if="currentModeMeta.needsStrength" cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            Strength: {{ form.strength }}
          </div>
          <v-slider
              v-model="form.strength"
              :min="0.1"
              :max="1"
              :step="0.05"
              thumb-label
              color="primary"
          />
        </v-col>
        <v-col v-if="currentModeMeta.needsControlnetScale" cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            Sketch Guidance: {{ form.controlnetConditioningScale }}
          </div>
          <v-slider
              v-model="form.controlnetConditioningScale"
              :min="0.1"
              :max="5"
              :step="0.05"
              thumb-label
              color="primary"
          />
        </v-col>
      </v-row>
    </v-card-text>

    <!-- Model Loading Status -->
    <div class="mb-4" v-if="!useCustomModel && activeModelId">
      <v-alert
          :type="isCurrentModelLoaded ? 'success' : 'warning'"
          :text="isCurrentModelLoaded ? 'Model loaded' : 'Model not loaded — click to load'"
          :icon="isCurrentModelLoaded ? 'mdi-check-circle' : 'mdi-memory'"
          variant="tonal"
          density="compact"
          class="mb-2"
      >
        <span class="text-body-2">{{ activeModelId }}</span>
      </v-alert>
      <v-btn
          v-if="!isCurrentModelLoaded"
          color="primary"
          variant="tonal"
          size="small"
          prepend-icon="mdi-download"
          @click="loadSelectedModel"
      >
        Load Model
      </v-btn>
    </div>

    <v-card-actions class="pa-4 pt-0">
      <v-btn
          color="primary"
          size="large"
          :loading="store.isGenerating || false"
          :disabled="!formValid || false"
          prepend-icon="mdi-creation"
          block
          @click="handleGenerate"
      >
        {{ store.isGenerating ? 'Generating...' : 'Generate' }}
      </v-btn>
    </v-card-actions>
  </v-card>
</template>