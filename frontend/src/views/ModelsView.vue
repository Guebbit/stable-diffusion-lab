<script setup lang="ts">
/**
 * Models catalog page.
 * Lists every available model with detailed descriptions, architecture info,
 * tags, and a link to the original source (HuggingFace / CivitAI).
 *
 * Data comes from src/data/models.ts — the same source used by GenerationForm.
 */
import { computed, ref } from 'vue'
import { huggingfaceModels, civitaiModels } from '../data/models'
import type { ModelOption } from '../types'

// Combine HuggingFace and CivitAI model arrays into a single flat list
// to enable unified filtering across all sources
const allModels: ModelOption[] = [...huggingfaceModels, ...civitaiModels]

// Active filter chips — user can narrow by source or tags
const activeSource = ref<'all' | 'huggingface' | 'civitai'>('all')
const activeFamily = ref<'all' | 'sd15' | 'sdxl'>('all')

// Extract all unique tags from the combined model list for filter UI — sorted alphabetically
const allTags = computed(() => {
  const tagSet = new Set<string>()
  allModels.forEach(m => m.tags?.forEach(t => tagSet.add(t)))
  return [...tagSet].sort()
})

const activeTag = ref<string | null>(null)

// Apply active filters in sequence: source filter → family filter → tag filter.
// Models must pass all active filters to be included in the result.
const filteredModels = computed(() =>
  allModels.filter(m => {
    if (activeSource.value !== 'all' && m.source !== activeSource.value) return false
    if (activeFamily.value !== 'all' && m.family !== activeFamily.value) return false
    if (activeTag.value && !m.tags?.includes(activeTag.value)) return false
    return true
  }),
)

// Badge colour per architecture family
function familyColor(family?: string) {
  if (family === 'sdxl') return 'purple'
  if (family === 'sd15') return 'blue'
  return 'grey'
}

// Label shown in the family chip
function familyLabel(family?: string) {
  if (family === 'sdxl') return 'SDXL'
  if (family === 'sd15') return 'SD 1.x'
  return 'Unknown'
}
</script>

<template>
  <div>
    <div class="text-h5 mb-1">
      <v-icon icon="mdi-brain" class="mr-2" color="primary" />
      Model Catalog
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6">
      All models available in the generation form — with architecture info, descriptions and source links.
      Use the custom model field in the form to load any other HuggingFace repo or CivitAI version ID.
    </p>

    <!-- ─── Filter bar ──────────────────────────────────────────────── -->
    <v-row class="mb-4" align="center">
      <!-- Source filter -->
      <v-col cols="12" sm="auto">
        <v-btn-toggle
          v-model="activeSource"
          mandatory
          variant="outlined"
          density="compact"
          divided
        >
          <v-btn value="all">All</v-btn>
          <v-btn value="huggingface">
            <v-icon icon="mdi-robot" class="mr-1" size="16" />
            HuggingFace
          </v-btn>
          <v-btn value="civitai">
            <v-icon icon="mdi-star-circle" class="mr-1" size="16" />
            CivitAI
          </v-btn>
        </v-btn-toggle>
      </v-col>

      <!-- Family filter -->
      <v-col cols="12" sm="auto">
        <v-btn-toggle
          v-model="activeFamily"
          mandatory
          variant="outlined"
          density="compact"
          divided
        >
          <v-btn value="all">All families</v-btn>
          <v-btn value="sd15">SD 1.x</v-btn>
          <v-btn value="sdxl">SDXL</v-btn>
        </v-btn-toggle>
      </v-col>

      <!-- Tag chips -->
      <v-col cols="12">
        <v-chip
          v-for="tag in allTags"
          :key="tag"
          :color="activeTag === tag ? 'primary' : undefined"
          :variant="activeTag === tag ? 'elevated' : 'outlined'"
          size="small"
          class="mr-1 mb-1"
          style="cursor: pointer"
          @click="activeTag = activeTag === tag ? null : tag"
        >
          {{ tag }}
        </v-chip>
        <v-chip
          v-if="activeTag"
          color="error"
          variant="text"
          size="small"
          prepend-icon="mdi-close"
          class="mb-1"
          style="cursor: pointer"
          @click="activeTag = null"
        >
          Clear
        </v-chip>
      </v-col>
    </v-row>

    <!-- ─── Result count ────────────────────────────────────────────── -->
    <p class="text-caption text-medium-emphasis mb-4">
      Showing {{ filteredModels.length }} of {{ allModels.length }} models
    </p>

    <!-- ─── Model cards grid ────────────────────────────────────────── -->
    <v-row>
      <v-col
        v-for="model in filteredModels"
        :key="`${model.source}-${model.id}`"
        cols="12"
        md="6"
        xl="4"
      >
        <v-card height="100%" variant="outlined">
          <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
            {{ model.name }}
          </v-card-title>

          <v-card-subtitle class="pb-2">
            <!-- Architecture family badge -->
            <v-chip
              :color="familyColor(model.family)"
              size="x-small"
              class="mr-1"
              label
            >
              {{ familyLabel(model.family) }}
            </v-chip>
            <!-- Source badge -->
            <v-chip
              :color="model.source === 'huggingface' ? 'teal' : 'orange'"
              size="x-small"
              class="mr-1"
              label
            >
              {{ model.source === 'huggingface' ? 'HuggingFace' : 'CivitAI' }}
            </v-chip>
          </v-card-subtitle>

          <v-card-text>
            <!-- Model ID pill -->
            <code class="text-caption d-block mb-3 pa-1 rounded bg-surface-variant">
              {{ model.id }}
            </code>

            <!-- Long description (falls back to short description) -->
            <p class="text-body-2 text-medium-emphasis mb-3">
              {{ model.longDescription ?? model.description ?? 'No description available.' }}
            </p>

            <!-- Tag chips -->
            <div v-if="model.tags?.length" class="mb-1">
              <v-chip
                v-for="tag in model.tags"
                :key="tag"
                size="x-small"
                variant="tonal"
                class="mr-1 mb-1"
              >
                {{ tag }}
              </v-chip>
            </div>
          </v-card-text>

          <v-card-actions class="px-4 pb-4">
            <v-btn
              v-if="model.sourceUrl"
              :href="model.sourceUrl"
              target="_blank"
              rel="noopener noreferrer"
              variant="tonal"
              color="primary"
              size="small"
              prepend-icon="mdi-open-in-new"
            >
              View source
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <!-- Empty state -->
    <v-alert
      v-if="filteredModels.length === 0"
      type="info"
      variant="tonal"
      class="mt-4"
    >
      No models match the current filters.
    </v-alert>
  </div>
</template>
