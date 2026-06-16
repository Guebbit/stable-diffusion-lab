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

// ── LoRA / base model override ─────────────────────────────────────────────

const loraEnabled = ref(false)
const loraModelId = ref<string>('')
const loraStrength = ref(0.8)

const baseModelOverrideEnabled = ref(false)
const baseModelOverrideId = ref<string>('')

// LoRA models available for the current mode (all registered, not-downloaded shown as disabled)
const loraModels = computed(() =>
  modelsStore.registry.filter((m) => m.model_type === 'lora'),
)

// Base diffusion models compatible with the currently selected sketch adapter (all registered)
const compatibleBaseModels = computed(() => {
  const adapter = modelsStore.registry.find((m) => m.model_id === activeModelId.value)
  const compatBases: string[] = adapter?.compatible_bases ?? []
  return modelsStore.registry.filter(
    (m) =>
      m.model_type === 'base_diffusion' &&
      (compatBases.length === 0 || compatBases.some((b) => m.family === b || m.base_model === b)),
  )
})

// ── Mode metadata: which fields each mode needs and its suggested defaults ──

interface ModeMeta {
  needsPrompt: boolean
  needsImage: boolean
  needsNegativePrompt: boolean
  needsStrength: boolean
  needsControlnetScale: boolean
  needsDimensions: boolean       // width/height — only meaningful when generating from scratch
  needsNumImages: boolean        // batch count — only for text-to-image
  needsGenerationParams: boolean // steps/CFG/seed — hidden for pure vision tasks
  needsSourceSelection: boolean  // show HuggingFace/CivitAI selector
  needsLora: boolean             // show optional LoRA select
  needsBaseModelOverride: boolean // show optional base model override (sketch-to-ink)
  defaultSteps: number
  defaultGuidance: number
  defaultStrength: number
  defaultConditioningScale: number
  // Capability string that models must have to appear in the model select.
  // Matches the capability values in the backend model registry.
  requiredCapability: string
  helperText: string
}

const modeMeta: Record<GenerationMode, ModeMeta> = {
  'text-to-image': {
    needsPrompt: true,
    needsImage: false,
    needsNegativePrompt: true,
    needsStrength: false,
    needsControlnetScale: false,
    needsDimensions: true,
    needsNumImages: true,
    needsGenerationParams: true,
    needsSourceSelection: true,
    needsLora: true,
    needsBaseModelOverride: false,
    defaultSteps: 20,
    defaultGuidance: 7.5,
    defaultStrength: 0,
    defaultConditioningScale: 1.0,
    requiredCapability: 'text_to_image',
    helperText: 'Generate an image from a text description.',
  },
  'image-to-image': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: true,
    needsControlnetScale: false,
    needsDimensions: false,
    needsNumImages: false,
    needsGenerationParams: true,
    needsSourceSelection: true,
    needsLora: true,
    needsBaseModelOverride: false,
    defaultSteps: 30,
    defaultGuidance: 7.5,
    defaultStrength: 0.6,
    defaultConditioningScale: 1.0,
    requiredCapability: 'image_to_image',
    helperText: 'Transform an existing image using a text prompt. Output dimensions follow the source.',
  },
  'describe-image': {
    needsPrompt: false,
    needsImage: true,
    needsNegativePrompt: false,
    needsStrength: false,
    needsControlnetScale: false,
    needsDimensions: false,
    needsNumImages: false,
    needsGenerationParams: false,
    needsSourceSelection: false,
    needsLora: false,
    needsBaseModelOverride: false,
    defaultSteps: 1,
    defaultGuidance: 1,
    defaultStrength: 0,
    defaultConditioningScale: 1.0,
    requiredCapability: 'image_description',
    helperText: 'Generate a text description of an uploaded image.',
  },
  'recolor-image': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: true,
    needsControlnetScale: false,
    needsDimensions: false,
    needsNumImages: false,
    needsGenerationParams: true,
    needsSourceSelection: false,
    needsLora: false,
    needsBaseModelOverride: false,
    defaultSteps: 24,
    defaultGuidance: 7.0,
    defaultStrength: 0.45,
    defaultConditioningScale: 1.0,
    requiredCapability: 'recolor_image',
    helperText: 'Change the colors of an image while preserving shapes. Output dimensions follow the source.',
  },
  'sketch-to-ink': {
    needsPrompt: true,
    needsImage: true,
    needsNegativePrompt: true,
    needsStrength: false,
    needsControlnetScale: true,
    needsDimensions: false,
    needsNumImages: false,
    needsGenerationParams: true,
    needsSourceSelection: false,
    needsLora: true,
    needsBaseModelOverride: true,
    defaultSteps: 28,
    defaultGuidance: 8.0,
    defaultStrength: 0,
    // 0.9 is the safe middle ground for both SD1.5 ControlNet lineart and SDXL T2I-Adapter sketch.
    // Too high (>1.2) makes the output rigid; too low (<0.5) ignores the sketch.
    defaultConditioningScale: 0.9,
    requiredCapability: 'sketch_to_ink',
    helperText: 'Convert a sketch into clean ink line art using a ControlNet or T2I-Adapter model.',
  },
}

// Capabilities that identify specialized/conditioned models — excluded from the
// general image-to-image select so only base diffusion models appear there.
const SPECIALIZED_CAPS = ['sketch_to_ink', 'recolor_image', 'image_description', 'upscale_image', 'face_restore']

const currentModeMeta = computed(() => modeMeta[generationMode.value])

// Filter models by the current mode's requiredCapability. All registered models are shown;
// not-downloaded ones appear as disabled options so the user knows what's available.
const availableModels = computed(() => {
  const { requiredCapability, needsSourceSelection } = currentModeMeta.value
  return modelsStore.registry.filter((m) => {
    const caps = m.capabilities ?? []
    if (!caps.includes(requiredCapability)) return false
    if (requiredCapability === 'image_to_image' && caps.some((c) => SPECIALIZED_CAPS.includes(c))) return false
    if (needsSourceSelection && m.source !== modelSourceSelection.value) return false
    return true
  })
})

const isSketchToInkMode = computed(() => generationMode.value === 'sketch-to-ink')
const needsImage = computed(() => currentModeMeta.value.needsImage)
const needsPrompt = computed(() => currentModeMeta.value.needsPrompt)

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
  form.value.controlnetConditioningScale = meta.defaultConditioningScale

  // Reset model selection and optional helpers — available options change per mode
  useCustomModel.value = false
  modelIdSelected.value = ''
  loraEnabled.value = false
  loraModelId.value = ''
  loraStrength.value = 0.8
  baseModelOverrideEnabled.value = false
  baseModelOverrideId.value = ''

  if (mode === 'sketch-to-ink') {
    if (!form.value.prompt.trim()) {
      form.value.prompt = 'clean black ink line art, crisp comic inking, bold outlines, high contrast, white background'
    }
    if (!form.value.negativePrompt.trim()) {
      form.value.negativePrompt = 'color, shading, painterly, blurry, messy sketch lines, grayscale wash, textured paper'
    }
    return
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
  if (mode === 'recolor-image') {
    if (!imageFile.value) return
    return store.recolor(modelId, imageFile.value, form.value.prompt, form.value.strength)
  }
  if (mode === 'sketch-to-ink') {
    if (!imageFile.value) return
    return store.sketchToInk(
      modelId,
      imageFile.value,
      form.value.prompt,
      form.value.negativePrompt,
      form.value.numInferenceSteps,
      form.value.guidanceScale,
      form.value.controlnetConditioningScale,
      baseModelOverrideEnabled.value ? baseModelOverrideId.value : '',
      loraEnabled.value ? loraModelId.value : '',
      loraStrength.value,
    )
  }
  if (mode === 'image-to-image') {
    const payload = {
      ...buildCommonPayload(),
      model_source: modelSourceSelection.value,
      num_inference_steps: form.value.numInferenceSteps,
      guidance_scale: form.value.guidanceScale,
      seed: form.value.seed ?? undefined,
      num_images: 1,
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
    lora_model_id: loraEnabled.value ? loraModelId.value : undefined,
    lora_strength: loraEnabled.value ? loraStrength.value : undefined,
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
        <v-col v-if="currentModeMeta.needsSourceSelection" cols="12">
          <v-select
              v-model="modelSourceSelection"
              :items="modelSourceItems"
              label="Model Source"
              variant="outlined"
              prepend-inner-icon="mdi-database"
              @update:model-value="() => modelIdSelected = ''"
          />
        </v-col>

        <v-col cols="12">
          <v-select
              v-if="!useCustomModel"
              v-model="modelIdSelected"
              :items="availableModels"
              item-title="name"
              item-value="model_id"
              label="Model"
              variant="outlined"
              prepend-inner-icon="mdi-brain"
              :item-props="(item: any) => ({ disabled: item.status !== 'downloaded' })"
          >
            <template #item="{ item, props: itemProps }">
              <v-list-item v-bind="itemProps" :subtitle="item.raw.short_description">
                <template #append>
                  <v-chip v-if="item.raw.status !== 'downloaded'" size="x-small" color="warning" variant="tonal">
                    Not downloaded
                  </v-chip>
                </template>
              </v-list-item>
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

      <!-- ── Optional: Base model override (sketch-to-ink only) ── -->
      <template v-if="currentModeMeta.needsBaseModelOverride">
        <v-divider class="mb-3"/>
        <v-checkbox
            v-model="baseModelOverrideEnabled"
            label="Override base model (auto-resolved by default)"
            density="compact"
            class="mb-1"
        />
        <v-select
            v-if="baseModelOverrideEnabled"
            v-model="baseModelOverrideId"
            :items="compatibleBaseModels"
            item-title="preferred_name"
            item-value="model_id"
            label="Base model"
            density="compact"
            variant="outlined"
            prepend-inner-icon="mdi-cube-outline"
            no-data-text="No compatible base models in registry"
            class="mb-2"
            :item-props="(item: any) => ({ disabled: item.status !== 'downloaded' })"
        >
          <template #item="{ item, props: itemProps }">
            <v-list-item v-bind="itemProps" :subtitle="item.raw.short_description">
              <template #append>
                <v-chip v-if="item.raw.status !== 'downloaded'" size="x-small" color="warning" variant="tonal">
                  Not downloaded
                </v-chip>
              </template>
            </v-list-item>
          </template>
        </v-select>
      </template>

      <!-- ── Optional: LoRA ── -->
      <template v-if="currentModeMeta.needsLora">
        <v-divider class="mb-3"/>
        <v-checkbox
            v-model="loraEnabled"
            label="Apply LoRA"
            density="compact"
            class="mb-1"
        />
        <template v-if="loraEnabled">
          <v-select
              v-model="loraModelId"
              :items="loraModels"
              item-title="preferred_name"
              item-value="model_id"
              label="LoRA model"
              density="compact"
              variant="outlined"
              prepend-inner-icon="mdi-layers-plus"
              no-data-text="No LoRA models in registry"
              class="mb-2"
              :item-props="(item: any) => ({ disabled: item.status !== 'downloaded' })"
          >
            <template #item="{ item, props: itemProps }">
              <v-list-item v-bind="itemProps" :subtitle="item.raw.short_description">
                <template #append>
                  <v-chip v-if="item.raw.status !== 'downloaded'" size="x-small" color="warning" variant="tonal">
                    Not downloaded
                  </v-chip>
                </template>
              </v-list-item>
            </template>
          </v-select>
          <div class="d-flex justify-space-between mb-1">
            <span class="text-caption text-medium-emphasis">LoRA strength</span>
            <span class="text-caption font-weight-bold">{{ loraStrength }}</span>
          </div>
          <v-slider
              v-model="loraStrength"
              :min="0"
              :max="2.0"
              :step="0.05"
              density="compact"
              color="secondary"
              thumb-label
              hint="0 = no effect • 1 = full strength • >1 = intensified"
              persistent-hint
              class="mb-2"
          />
        </template>
      </template>

      <v-divider class="mb-4"/>

      <!-- Dimensions (text-to-image only) -->
      <v-row v-if="currentModeMeta.needsDimensions">
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
      </v-row>

      <!-- Generation parameters (hidden for pure vision tasks like describe) -->
      <template v-if="currentModeMeta.needsGenerationParams">
        <v-row>
          <v-col v-if="currentModeMeta.needsNumImages" cols="6">
            <v-select
                v-model="form.numImages"
                :items="[1, 2, 3, 4, 6, 8]"
                label="Images"
                variant="outlined"
            />
          </v-col>
          <v-col :cols="currentModeMeta.needsNumImages ? 6 : 12">
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
              <span class="ml-1 opacity-60">higher = more detail, slower</span>
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
              <span class="ml-1 opacity-60">higher = stricter prompt</span>
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
              <span class="ml-1 opacity-60">how much to change the source</span>
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
              <span class="ml-1 opacity-60">how closely to follow the sketch</span>
            </div>
            <v-slider
                v-model="form.controlnetConditioningScale"
                :min="0.1"
                :max="2.0"
                :step="0.05"
                thumb-label
                color="primary"
            />
          </v-col>
        </v-row>
      </template>
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