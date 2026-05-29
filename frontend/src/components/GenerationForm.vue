<script setup lang="ts">
import {computed, onBeforeUnmount, ref} from 'vue'
import {useDiffusionStore} from '../stores/diffusion'
import type {GenerationMode, GenerationTask, ModelOption, ModelSource} from '../types'

/** Pinia diffusion store containing model-load and generation actions. */
const store = useDiffusionStore()

/** Curated HuggingFace base models shown in the quick-select dropdown. */
const huggingfaceModels: ModelOption[] = [
  {
    id: 'runwayml/stable-diffusion-v1-5',
    name: 'Stable Diffusion v1.5',
    source: 'huggingface',
    description: 'Bad and fast (2022)',
  },
  {
    id: 'stabilityai/sdxl-turbo',
    name: 'Stable Diffusion SDXL',
    source: 'huggingface',
    description: 'SDXL Turbo',
  },
  {
    id: 'stabilityai/stable-diffusion-xl-base-1.0',
    name: 'Stable Diffusion XL 1.0',
    source: 'huggingface',
    description: 'SDXL Quality',
  },
]

/** Curated CivitAI model IDs. IDs map to CivitAI model version IDs. */
const civitaiModels: ModelOption[] = [
  {
    id: '128713',
    name: 'DreamShaper',
    source: 'civitai',
    description: 'Versatile art/photo model from CivitAI',
  },
]

/** Active model source selection (HuggingFace or CivitAI). */
const modelSourceSelection = ref<ModelSource>('huggingface')
/** Dropdown options for model source selection. */
const modelSourceItems = [
  {title: 'HuggingFace', value: 'huggingface' as ModelSource},
  {title: 'CivitAI', value: 'civitai' as ModelSource},
]

/** UI mode controls which backend endpoint/workflow is called. */
const generationMode = ref<GenerationMode>('text-to-image')
/** Human-readable labels for every generation mode. */
const generationModeItems = [
  {title: 'Text to Image', value: 'text-to-image' as GenerationMode},
  {title: 'Image to Image', value: 'image-to-image' as GenerationMode},
  {title: 'Sketch to Ink', value: 'sketch-to-ink' as GenerationMode},
  {title: 'Recolor Image', value: 'recolor-image' as GenerationMode},
  {title: 'Upscale Image', value: 'upscale-image' as GenerationMode},
]
/** Uploaded source image used by every image-guided workflow. */
const imageFile = ref<File | null>(null)
/** Browser object URL used to preview the currently selected image file. */
const imagePreviewUrl = ref<string | null>(null)

/** Toggle for advanced users who want to type a model ID manually. */
const useCustomModel = ref(false)

/** Reactive form model for shared generation controls. */
interface IForm {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  numInferenceSteps: number
  guidanceScale: number
  strength: number
  controlnetConditioningScale: number
  upscaleFactor: number
  seed: number | null
  numImages: number
}

/** Default values used when the form is first rendered or reset by mode presets. */
const defaultForm: IForm = {
  prompt: '',
  negativePrompt: '',
  width: 512,
  height: 512,
  numInferenceSteps: 20,
  guidanceScale: 7.5,
  strength: 0.6,
  controlnetConditioningScale: 1.1,
  upscaleFactor: 2,
  seed: null,
  numImages: 1,
}

/** Preset defaults for sketch cleanup + inking. */
const sketchPromptPreset = 'clean black ink line art, crisp comic inking, bold outlines, high contrast, white background'
const sketchNegativePromptPreset = 'color, shading, painterly, blurry, messy sketch lines, grayscale wash, textured paper'
const sketchDefaultSteps = 28
const sketchDefaultGuidanceScale = 8
const sketchDefaultStrength = 0.6

/** Preset defaults for recolor transformations while preserving composition. */
const recolorPromptPreset = 'recolor this image with clean coherent color palette, preserve shapes and composition'
const recolorNegativePromptPreset = 'black and white, grayscale, washed out colors, muddy palette, distorted geometry'

/** Preset defaults for detail-preserving upscale passes. */
const upscalePromptPreset = 'highly detailed, sharp textures, clean edges, natural lighting, high quality'
const upscaleNegativePromptPreset = 'blurry, noisy, artifacts, oversharpened halos, low resolution, compression artifacts'

/** Main reactive form state. */
const form = ref<IForm>({...defaultForm})

/** Supported dimensions exposed in width/height dropdown controls. */
const dimensionOptions = [256, 512, 768, 1024]

/** Models available for the currently selected source. */
const availableModels = computed(() =>
    modelSourceSelection.value === 'huggingface' ? huggingfaceModels : civitaiModels,
)
/** Convenience flag for sketch mode-specific controls and alerts. */
const isSketchToInkMode = computed(() => generationMode.value === 'sketch-to-ink')
/** Convenience flag for recolor mode-specific presets/labels. */
const isRecolorMode = computed(() => generationMode.value === 'recolor-image')
/** Convenience flag for upscale mode-specific controls. */
const isUpscaleMode = computed(() => generationMode.value === 'upscale-image')
/** Strength slider is relevant only for img2img-derived (non-ControlNet) image workflows. */
const showsStrengthControl = computed(
    () =>
      generationMode.value === 'image-to-image' ||
      isRecolorMode.value ||
      isUpscaleMode.value,
)
/** Every non-text mode needs an uploaded source image. */
const isImageGuidedMode = computed(() => generationMode.value !== 'text-to-image')
/** Sketch mode currently supports HuggingFace only, so we hide CivitAI there. */
const availableModelSourceItems = computed(() =>
    isSketchToInkMode.value ? [modelSourceItems[0]] : modelSourceItems,
)

/** Model ID selected from dropdown list. */
const modelIdSelected = ref(huggingfaceModels[0]?.id ?? '')
/** Manually typed model ID when custom-model mode is enabled. */
const customModelId = ref('')
/** Final active model ID sent to the backend. */
const activeModelId = computed(() =>
    useCustomModel.value ? customModelId.value : modelIdSelected.value,
)

/** Upscale mode can use backend defaults, so prompt is optional there. */
const requiresPrompt = computed(() => !isUpscaleMode.value)

/** Main form validity guard used to enable/disable the generate button. */
const formValid = computed(
    () =>
      (!requiresPrompt.value || form.value.prompt.trim().length > 0) &&
      activeModelId.value.trim().length > 0 &&
      (!isImageGuidedMode.value || !!imageFile.value),
)

/**
 * Map UI mode values to backend model-loading tasks.
 * This keeps model caching correct per task family.
 */
function resolveGenerationTask(mode: GenerationMode): GenerationTask {
  if (mode === 'image-to-image' || mode === 'recolor-image' || mode === 'upscale-image') return 'img2img'
  if (mode === 'sketch-to-ink') return 'sketch2ink'
  return 'text2img'
}

/**
 * Apply mode-specific presets so each workflow starts with useful defaults.
 * Presets are intentionally simple to keep the flow ADHD-friendly and predictable.
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
    form.value.strength = sketchDefaultStrength
    return
  }

  if (mode === 'recolor-image') {
    if (!form.value.prompt.trim()) {
      form.value.prompt = recolorPromptPreset
    }
    if (!form.value.negativePrompt.trim()) {
      form.value.negativePrompt = recolorNegativePromptPreset
    }
    form.value.numInferenceSteps = 24
    form.value.guidanceScale = 7
    form.value.strength = 0.55
    return
  }

  if (mode === 'upscale-image') {
    if (!form.value.prompt.trim()) {
      form.value.prompt = upscalePromptPreset
    }
    if (!form.value.negativePrompt.trim()) {
      form.value.negativePrompt = upscaleNegativePromptPreset
    }
    form.value.numInferenceSteps = 28
    form.value.guidanceScale = 6
    form.value.strength = 0.35
    form.value.upscaleFactor = 2
  }
}

/**
 * Ask backend to preload the selected model for the current workflow type.
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
 * Submit the generation request by branching to the matching workflow action.
 * We keep branches explicit so each workflow payload is easy to reason about.
 */
function handleGenerate() {
  if (!formValid.value) return
  if (generationMode.value === 'image-to-image' && imageFile.value) {
    return store.generateFromImage({
      image: imageFile.value,
      prompt: form.value.prompt.trim(),
      negative_prompt: form.value.negativePrompt.trim() || undefined,
      model_id: activeModelId.value,
      model_source: modelSourceSelection.value,
      width: form.value.width,
      height: form.value.height,
      strength: form.value.strength,
      num_inference_steps: form.value.numInferenceSteps,
      guidance_scale: form.value.guidanceScale,
      seed: form.value.seed ?? undefined,
      num_images: form.value.numImages,
    })
  }

  if (generationMode.value === 'sketch-to-ink' && imageFile.value) {
    return store.generateSketchToInk({
      image: imageFile.value,
      prompt: form.value.prompt.trim(),
      negative_prompt: form.value.negativePrompt.trim() || undefined,
      model_id: activeModelId.value,
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

  if (generationMode.value === 'recolor-image' && imageFile.value) {
    return store.generateRecolor({
      image: imageFile.value,
      prompt: form.value.prompt.trim(),
      negative_prompt: form.value.negativePrompt.trim() || undefined,
      model_id: activeModelId.value,
      model_source: modelSourceSelection.value,
      width: form.value.width,
      height: form.value.height,
      strength: form.value.strength,
      num_inference_steps: form.value.numInferenceSteps,
      guidance_scale: form.value.guidanceScale,
      seed: form.value.seed ?? undefined,
      num_images: form.value.numImages,
    })
  }

  if (generationMode.value === 'upscale-image' && imageFile.value) {
    return store.generateUpscale({
      image: imageFile.value,
      prompt: form.value.prompt.trim() || undefined,
      negative_prompt: form.value.negativePrompt.trim() || undefined,
      model_id: activeModelId.value,
      model_source: modelSourceSelection.value,
      upscale_factor: form.value.upscaleFactor,
      strength: form.value.strength,
      num_inference_steps: form.value.numInferenceSteps,
      guidance_scale: form.value.guidanceScale,
      width: form.value.width,
      height: form.value.height,
      seed: form.value.seed ?? undefined,
      num_images: form.value.numImages,
    })
  }

  return store.generate({
    prompt: form.value.prompt.trim(),
    negative_prompt: form.value.negativePrompt.trim() || undefined,
    model_id: activeModelId.value,
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
 * Handle file-input payload differences and update local preview state.
 * Vuetify can emit either a single file or an array depending on configuration.
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

/** Release object URL memory when component unmounts. */
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
          :label="isUpscaleMode ? 'Prompt (optional for upscale)' : 'Prompt'"
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
          v-if="isRecolorMode"
          type="info"
          variant="tonal"
          density="compact"
          class="mb-4"
      >
        Recolor keeps structure from the uploaded image while changing palette/style from your prompt.
      </v-alert>
      <v-alert
          v-if="isUpscaleMode"
          type="info"
          variant="tonal"
          density="compact"
          class="mb-4"
      >
        Upscale enlarges the uploaded image and runs a gentle image-to-image pass to restore detail.
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
              :items="[1, 2, 4]"
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
              :min="10"
              :max="100"
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
              :max="20"
              :step="0.5"
              thumb-label
              color="primary"
          />
        </v-col>
        <v-col v-if="showsStrengthControl" cols="12" sm="6">
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
        <v-col v-if="isUpscaleMode" cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            Upscale Factor: {{ form.upscaleFactor }}x
          </div>
          <v-slider
              v-model="form.upscaleFactor"
              :min="1"
              :max="4"
              :step="1"
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
              :max="2"
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
