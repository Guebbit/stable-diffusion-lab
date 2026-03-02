<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDiffusionStore } from '../stores/diffusion'
import type { ModelOption, ModelSource } from '../types'

const store = useDiffusionStore()

const huggingfaceModels: ModelOption[] = [
  {
    id: 'runwayml/stable-diffusion-v1-5',
    name: 'Stable Diffusion v1.5',
    source: 'huggingface',
    description: 'Classic SD 1.5 by RunwayML',
  },
  {
    id: 'stabilityai/stable-diffusion-2-1',
    name: 'Stable Diffusion v2.1',
    source: 'huggingface',
    description: 'SD 2.1 by Stability AI',
  },
  {
    id: 'stabilityai/stable-diffusion-xl-base-1.0',
    name: 'Stable Diffusion XL 1.0',
    source: 'huggingface',
    description: 'SDXL base model by Stability AI',
  },
]

const civitaiModels: ModelOption[] = [
  {
    id: '4201',
    name: 'Realistic Vision V6.0',
    source: 'civitai',
    description: 'Photorealistic model from CivitAI',
  },
  {
    id: '7240',
    name: 'DreamShaper',
    source: 'civitai',
    description: 'Versatile art/photo model from CivitAI',
  },
]

const modelSourceOptions = [
  { title: 'HuggingFace', value: 'huggingface' as ModelSource },
  { title: 'CivitAI', value: 'civitai' as ModelSource },
]

const modelSource = ref<ModelSource>('huggingface')
const selectedModelId = ref(huggingfaceModels[0]?.id ?? '')
const customModelId = ref('')
const useCustomModel = ref(false)

const prompt = ref('')
const negativePrompt = ref('blurry, bad quality, nsfw, watermark')
const width = ref(512)
const height = ref(512)
const numInferenceSteps = ref(20)
const guidanceScale = ref(7.5)
const seed = ref<number | null>(null)
const numImages = ref(1)

const dimensionOptions = [256, 512, 768, 1024]

const availableModels = computed(() =>
  modelSource.value === 'huggingface' ? huggingfaceModels : civitaiModels,
)

const activeModelId = computed(() =>
  useCustomModel.value ? customModelId.value : selectedModelId.value,
)

const formValid = computed(
  () => prompt.value.trim().length > 0 && activeModelId.value.trim().length > 0,
)

function onModelSourceChange() {
  selectedModelId.value = availableModels.value[0]?.id ?? ''
}

async function handleLoadModel() {
  if (!activeModelId.value) return
  await store.loadModel(activeModelId.value, modelSource.value)
}

async function handleGenerate() {
  if (!formValid.value) return
  await store.generate({
    prompt: prompt.value.trim(),
    negative_prompt: negativePrompt.value.trim() || undefined,
    model_id: activeModelId.value,
    model_source: modelSource.value,
    width: width.value,
    height: height.value,
    num_inference_steps: numInferenceSteps.value,
    guidance_scale: guidanceScale.value,
    seed: seed.value ?? undefined,
    num_images: numImages.value,
  })
}
</script>

<template>
  <v-card class="pa-4" elevation="2">
    <v-card-title class="text-h5 mb-2">
      <v-icon icon="mdi-image-sparkle" class="mr-2" />
      Image Generation
    </v-card-title>

    <v-card-text>
      <!-- Prompt -->
      <v-textarea
        v-model="prompt"
        label="Prompt"
        placeholder="a beautiful landscape painting, oil on canvas, highly detailed..."
        rows="3"
        auto-grow
        variant="outlined"
        prepend-inner-icon="mdi-pencil"
        class="mb-3"
      />

      <!-- Negative Prompt -->
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

      <!-- Model Source -->
      <v-row>
        <v-col cols="12" sm="4">
          <v-select
            v-model="modelSource"
            :items="modelSourceOptions"
            label="Model Source"
            variant="outlined"
            prepend-inner-icon="mdi-database"
            @update:model-value="onModelSourceChange"
          />
        </v-col>

        <v-col cols="12" sm="8">
          <v-select
            v-if="!useCustomModel"
            v-model="selectedModelId"
            :items="availableModels"
            item-title="name"
            item-value="id"
            label="Model"
            variant="outlined"
            prepend-inner-icon="mdi-brain"
          >
            <template #item="{ item, props: itemProps }">
              <v-list-item v-bind="itemProps" :subtitle="item.raw.description" />
            </template>
          </v-select>
          <v-text-field
            v-else
            v-model="customModelId"
            :label="modelSource === 'huggingface' ? 'HuggingFace repo ID' : 'CivitAI model ID'"
            :placeholder="
              modelSource === 'huggingface'
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

      <v-divider class="mb-4" />

      <!-- Generation Parameters -->
      <v-row>
        <v-col cols="6" sm="3">
          <v-select
            v-model="width"
            :items="dimensionOptions"
            label="Width"
            variant="outlined"
            suffix="px"
          />
        </v-col>
        <v-col cols="6" sm="3">
          <v-select
            v-model="height"
            :items="dimensionOptions"
            label="Height"
            variant="outlined"
            suffix="px"
          />
        </v-col>
        <v-col cols="6" sm="3">
          <v-select
            v-model="numImages"
            :items="[1, 2, 4]"
            label="Images"
            variant="outlined"
          />
        </v-col>
        <v-col cols="6" sm="3">
          <v-text-field
            v-model.number="seed"
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
            Steps: {{ numInferenceSteps }}
          </div>
          <v-slider
            v-model="numInferenceSteps"
            :min="10"
            :max="100"
            :step="1"
            thumb-label
            color="primary"
          />
        </v-col>
        <v-col cols="12" sm="6">
          <div class="text-caption text-medium-emphasis mb-1">
            CFG Scale: {{ guidanceScale }}
          </div>
          <v-slider
            v-model="guidanceScale"
            :min="1"
            :max="20"
            :step="0.5"
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
