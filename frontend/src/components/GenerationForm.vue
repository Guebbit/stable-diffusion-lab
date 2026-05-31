<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import type { GenerationMode, GenerationTask, ImageWorkflowPreset, ModelSource } from '../types'
// Shared model catalog — single source of truth for the dropdown and the Models catalog page
import { huggingfaceModels, civitaiModels } from '../data/models'

const store = useDiffusionStore()

/**
 * Current model source and available source options.
 */
const modelSourceSelection = ref<ModelSource>('huggingface')
const modelSourceItems = [
  { title: 'HuggingFace', value: 'huggingface' as ModelSource },
  { title: 'CivitAI', value: 'civitai' as ModelSource },
]

/**
 * Mode selector includes practical image workflow options.
 */
const generationMode = ref<GenerationMode>('text-to-image')
const generationModeItems = [
  { title: 'Text to Image', value: 'text-to-image' as GenerationMode },
  { title: 'Image to Image', value: 'image-to-image' as GenerationMode },
  { title: 'Recolor Image', value: 'recolor-image' as GenerationMode },
  { title: 'Style Transfer', value: 'style-transfer' as GenerationMode },
  { title: 'Upscale Image', value: 'upscale-image' as GenerationMode },
  { title: 'Sketch to Ink', value: 'sketch-to-ink' as GenerationMode },
]
const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)

/**
 * Toggle for custom model ID input.
 */
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

interface ImageWorkflowPresetConfig {
  workflowPreset: ImageWorkflowPreset
  prompt: string
  negativePrompt: string
  strength: number
  numInferenceSteps: number
  guidanceScale: number
  width?: number
  height?: number
  helperText: string
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

const sketchPromptPreset = 'clean black ink line art, crisp comic inking, bold outlines, high contrast, white background'
const sketchNegativePromptPreset = 'color, shading, painterly, blurry, messy sketch lines, grayscale wash, textured paper'
const sketchDefaultSteps = 28
const sketchDefaultGuidanceScale = 8

/**
 * Prompt + parameter presets for image-guided workflows.
 */
const imageWorkflowPresets: Record<
  Exclude<GenerationMode, 'text-to-image' | 'sketch-to-ink'>,
  ImageWorkflowPresetConfig
> = {
  'image-to-image': {
    workflowPreset: 'general',
    prompt: 'refine details, preserve composition, clean textures',
    negativePrompt: 'low quality, blur, distorted anatomy, artifacts',
    strength: 0.6,
    numInferenceSteps: 20,
    guidanceScale: 7.5,
    helperText: 'Balanced transformation while preserving original composition.',
  },
  'recolor-image': {
    workflowPreset: 'recolor',
    prompt: 'recolor image with cohesive palette, preserve shapes and edges',
    negativePrompt: 'muddy colors, grayscale, oversaturated, color bleeding',
    strength: 0.45,
    numInferenceSteps: 24,
    guidanceScale: 7,
    helperText: 'Best when you want new colors but stable line work.',
  },
  'style-transfer': {
    workflowPreset: 'style-transfer',
    prompt: 'apply stylized painterly look with detailed brushwork',
    negativePrompt: 'blurry, deformed, low detail, washed out',
    strength: 0.72,
    numInferenceSteps: 30,
    guidanceScale: 8,
    helperText: 'Higher creativity mode for major style changes.',
  },
  'upscale-image': {
    workflowPreset: 'upscale',
    prompt: 'highly detailed upscale, sharp edges, clean textures',
    negativePrompt: 'pixelated, blurry, noise, compression artifacts',
    strength: 0.3,
    numInferenceSteps: 18,
    guidanceScale: 6.5,
    width: 1024,
    height: 1024,
    helperText: 'Gentle denoise setting intended for resolution/detail upgrades.',
  },
}

/**
 * Form state is shared across all modes so users can switch without losing context.
 */
const form = ref<IForm>({ ...defaultForm })

/**
 * Allowed image sizes expected by the backend.
 */
const dimensionOptions = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

/**
 * Select model list based on current source.
 */
const availableModels = computed(() =>
  modelSourceSelection.value === 'huggingface' ? huggingfaceModels : civitaiModels,
)
const isSketchToInkMode = computed(() => generationMode.value === 'sketch-to-ink')
const isImageGuidedMode = computed(() => generationMode.value !== 'text-to-image')
const availableModelSourceItems = computed(() =>
  isSketchToInkMode.value ? [modelSourceItems[0]] : modelSourceItems,
)
const selectedPhaseThreePreset = computed(() =>
  isImageWorkflowPresetMode(generationMode.value)
    ? imageWorkflowPresets[generationMode.value]
    : null,
)

/**
 * Model selection controls.
 */
const modelIdSelected = ref(huggingfaceModels[0]?.id ?? '')
const customModelId = ref('')
const activeModelId = computed(() =>
  useCustomModel.value ? customModelId.value : modelIdSelected.value,
)

/**
 * Keep generate button disabled until required fields are valid.
 */
const formValid = computed(
  () =>
    form.value.prompt.trim().length > 0 &&
    activeModelId.value.trim().length > 0 &&
    (!isImageGuidedMode.value || !!imageFile.value),
)
const initializedWorkflowModes = ref<Set<Exclude<GenerationMode, 'text-to-image' | 'sketch-to-ink'>>>(
  new Set(),
)

/**
 * Map UI generation mode to backend model-loading task.
 */
function resolveGenerationTask(mode: GenerationMode): GenerationTask {
  if (isImageWorkflowPresetMode(mode)) return 'img2img'
  if (mode === 'sketch-to-ink') return 'sketch2ink'
  return 'text2img'
}

/**
 * Type guard for modes that use image-workflow preset configs.
 */
function isImageWorkflowPresetMode(
  mode: GenerationMode,
): mode is Exclude<GenerationMode, 'text-to-image' | 'sketch-to-ink'> {
  return mode in imageWorkflowPresets
}

/**
 * Apply default prompt/inference values when the user switches generation mode.
 */
function handleGenerationModeChange(mode: GenerationMode) {
  if (mode === 'sketch-to-ink') {
    modelSourceSelection.value = 'huggingface'
    useCustomModel.value = false
    modelIdSelected.value = huggingfaceModels[0]?.id ?? ''

    if (!form.value.prompt.trim()) {
      form.value.prompt = sketchPromptPreset
    }

    if (!form.value.negativePrompt.trim()) {
      form.value.negativePrompt = sketchNegativePromptPreset
    }

    form.value.numInferenceSteps = sketchDefaultSteps
    form.value.guidanceScale = sketchDefaultGuidanceScale
    form.value.controlnetConditioningScale = 1.1
    return
  }

  if (isImageWorkflowPresetMode(mode)) {
    if (initializedWorkflowModes.value.has(mode)) return
    const preset = imageWorkflowPresets[mode]
    form.value.prompt = preset.prompt
    form.value.negativePrompt = preset.negativePrompt
    form.value.strength = preset.strength
    form.value.numInferenceSteps = preset.numInferenceSteps
    form.value.guidanceScale = preset.guidanceScale
    if (typeof preset.width === 'number') form.value.width = preset.width
    if (typeof preset.height === 'number') form.value.height = preset.height
    initializedWorkflowModes.value.add(mode)
  }
}

/**
 * Load currently selected model with the task type implied by the active mode.
 */
function handleLoadModel() {
  if (!activeModelId.value) return
  return store.loadModel(
    activeModelId.value,
    modelSourceSelection.value,
    resolveGenerationTask(generationMode.value),
  )
}

/**
 * Build the common prompt/model payload fields shared by all generation calls.
 */
function buildCommonGenerationPayload() {
  return {
    prompt: form.value.prompt.trim(),
    negative_prompt: form.value.negativePrompt.trim() || undefined,
    model_id: activeModelId.value,
  }
}

/**
 * Trigger one of the image-to-image preset workflows.
 */
function generatePresetWorkflow() {
  if (!imageFile.value) return
  const workflowPreset = selectedPhaseThreePreset.value?.workflowPreset ?? 'general'
  const payload = buildCommonGenerationPayload()
  return store.generateFromImage({
    ...payload,
    image: imageFile.value,
    model_source: modelSourceSelection.value,
    workflow_preset: workflowPreset,
    width: form.value.width,
    height: form.value.height,
    strength: form.value.strength,
    num_inference_steps: form.value.numInferenceSteps,
    guidance_scale: form.value.guidanceScale,
    seed: form.value.seed ?? undefined,
    num_images: form.value.numImages,
  })
}

/**
 * Trigger sketch-to-ink generation with ControlNet conditioning settings.
 */
function generateSketchWorkflow() {
  if (!imageFile.value) return
  const payload = buildCommonGenerationPayload()
  return store.generateSketchToInk({
    ...payload,
    image: imageFile.value,
    model_source: 'huggingface',
    width: form.value.width,
    height: form.value.height,
    controlnet_conditioning_scale: form.value.controlnetConditioningScale,
    num_inference_steps: form.value.numInferenceSteps,
    guidance_scale: form.value.guidanceScale,
    seed: form.value.seed ?? undefined,
    num_images: form.value.numImages,
  })
}

/**
 * Trigger standard text-to-image generation.
 */
function generateTextWorkflow() {
  const payload = buildCommonGenerationPayload()
  return store.generate({
    ...payload,
    model_source: modelSourceSelection.value,
    width: form.value.width,
    height: form.value.height,
    num_inference_steps: form.value.numInferenceSteps,
    guidance_scale: form.value.guidanceScale,
    seed: form.value.seed ?? undefined,
    num_images: form.value.numImages,
  })
}

/**
 * Route the request to the correct backend endpoint based on generation mode.
 */
function handleGenerate() {
  if (!formValid.value) return
  if (isImageGuidedMode.value && !isSketchToInkMode.value) return generatePresetWorkflow()
  if (generationMode.value === 'sketch-to-ink') return generateSketchWorkflow()
  return generateTextWorkflow()
}

/**
 * Keep one uploaded file and manage its local preview URL lifecycle safely.
 */
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

/**
 * Cleanup object URL to avoid browser memory leaks.
 */
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
      <v-select
          v-model="generationMode"
          :items="generationModeItems"
          label="Generation Mode"
          variant="outlined"
          prepend-inner-icon="mdi-tune-variant"
          class="mb-4"
          @update:model-value="handleGenerationModeChange"
      />
      
      <v-textarea
          v-model="form.prompt"
          label="Prompt"
          rows="3"
          auto-grow
          variant="outlined"
          prepend-inner-icon="mdi-pencil"
          class="mb-3"
      />

      <v-textarea
          v-model="form.negativePrompt"
          label="Negative Prompt"
          rows="2"
          auto-grow
          variant="outlined"
          prepend-inner-icon="mdi-pencil-off"
          class="mb-4"
      />

      <v-alert
          v-if="isSketchToInkMode"
          type="info"
          variant="tonal"
          density="compact"
          class="mb-4"
      >
        Sketch to Ink uses a built-in ControlNet scribble pipeline and currently supports HuggingFace SD 1.5 / SDXL base models.
      </v-alert>
      <v-alert
          v-else-if="selectedPhaseThreePreset"
          type="info"
          variant="tonal"
          density="compact"
          class="mb-4"
      >
        {{ selectedPhaseThreePreset.helperText }}
      </v-alert>

      <div v-if="isImageGuidedMode" class="mb-4">
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

      <!-- Model Source -->
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

      <v-btn
          color="secondary"
          variant="tonal"
          :loading="store.isLoadingModel"
          :disabled="!activeModelId || store.isGenerating"
          prepend-icon="mdi-download"
          class="mb-4"
          @click="handleLoadModel"
      >
        Load Model
      </v-btn>

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
            <p>
              <small>
                inference steps refer to the process of drawing conclusions based on evidence and reasoning. This typically involves gathering relevant information, identifying patterns, and synthesizing these details to reach a logical conclusion.
              </small>
            </p>
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
            <p>
              <small>
                controls how closely the generated image follows the text prompt. A higher CFG scale value means the
                image will adhere more strictly to the prompt, while a lower value allows for more creative freedom and
                variation in the output.
              </small>
            </p>
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
        <v-col v-if="generationMode === 'image-to-image'" cols="12" sm="6">
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
        <v-col v-if="isSketchToInkMode" cols="12" sm="6">
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

    <v-card-actions class="pa-4 pt-0">
      <v-btn
          color="primary"
          size="large"
          :loading="store.isGenerating"
          :disabled="!formValid || store.isLoadingModel"
          prepend-icon="mdi-creation"
          block
          @click="handleGenerate"
      >
        {{ store.isGenerating ? 'Generating...' : 'Generate' }}
      </v-btn>
    </v-card-actions>
  </v-card>
</template>
