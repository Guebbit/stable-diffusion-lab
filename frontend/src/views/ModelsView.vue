<script setup lang="ts">
/**
 * Models management page.
 * Shows all registered models with download status, allows adding new models
 * and triggering downloads. The backend is the source of truth for the catalog.
 */
import { computed, onMounted, ref } from 'vue'
import { useModelsStore } from '../stores/models'
import type { ModelFamily, ModelRegistryAddRequest, ModelSource } from '../types'

const modelsStore = useModelsStore()

// Fetch registry on mount
onMounted(() => {
  modelsStore.fetchRegistry()
})

// ─── Filters ──────────────────────────────────────────────────────────────

const activeSource = ref<'all' | 'huggingface' | 'civitai'>('all')
const activeFamily = ref<'all' | 'sd15' | 'sdxl'>('all')
const activeDownloaded = ref<'all' | 'downloaded' | 'not-downloaded'>('all')

const filteredModels = computed(() =>
  modelsStore.registry.filter(m => {
    if (activeSource.value !== 'all' && m.source !== activeSource.value) return false
    if (activeFamily.value !== 'all' && m.family !== activeFamily.value) return false
    if (activeDownloaded.value === 'downloaded' && m.status !== 'downloaded') return false
    if (activeDownloaded.value === 'not-downloaded' && m.status === 'downloaded') return false
    return true
  }),
)

// ─── Add Model dialog ─────────────────────────────────────────────────────

const showAddDialog = ref(false)
const addForm = ref<ModelRegistryAddRequest>({
  id: '',
  name: '',
  source: 'huggingface',
  family: 'sd15',
  description: '',
  long_description: '',
  tags: [],
  source_url: '',
  size: '',
})
const tagInput = ref('')

function resetAddForm() {
  addForm.value = {
    id: '',
    name: '',
    source: 'huggingface',
    family: 'sd15',
    description: '',
    long_description: '',
    tags: [],
    source_url: '',
    size: '',
  }
  tagInput.value = ''
}

function handleAddModel() {
  // Parse comma-separated tags
  if (tagInput.value.trim()) {
    addForm.value.tags = tagInput.value.split(',').map(t => t.trim()).filter(Boolean)
  }
  modelsStore.addModel(addForm.value)
    .then(() => {
      showAddDialog.value = false
      resetAddForm()
    })
    .catch(() => { /* error handled by store */ })
}

// ─── Actions ──────────────────────────────────────────────────────────────

function handleDownload(modelId: string, source: ModelSource) {
  modelsStore.downloadModel(modelId, source)
}

function handleRemove(modelId: string, source: ModelSource) {
  modelsStore.removeModel(modelId, source)
}

// ─── UI helpers ───────────────────────────────────────────────────────────

function familyColor(family?: string) {
  if (family === 'sdxl') return 'purple'
  if (family === 'sd15') return 'blue'
  return 'grey'
}

function familyLabel(family?: string) {
  if (family === 'sdxl') return 'SDXL'
  if (family === 'sd15') return 'SD 1.x'
  return 'Unknown'
}

// ─── Download Progress ───────────────────────────────────────────────────

function downloadPercentageFor(model: typeof modelsStore.registry[0]): number {
  const key = `${model.source}:${model.id}`
  const p = modelsStore.downloadProgress.get(key)
  if (!p || p.total_bytes === 0) return 0
  return Math.round((p.percentage) || 0)
}

const progressColor = 'blue'
</script>

<template>
  <div>
    <div class="d-flex align-center mb-1">
      <div class="text-h5">
        <v-icon icon="mdi-brain" class="mr-2" color="primary" />
        Model Manager
      </div>
      <v-spacer />
      <v-btn
        color="primary"
        prepend-icon="mdi-plus"
        @click="showAddDialog = true"
      >
        Add Model
      </v-btn>
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Manage your model catalog — download models to make them available for generation.
      Only downloaded models appear in the generation form.
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

      <!-- Download status filter -->
      <v-col cols="12" sm="auto">
        <v-btn-toggle
          v-model="activeDownloaded"
          mandatory
          variant="outlined"
          density="compact"
          divided
        >
          <v-btn value="all">All</v-btn>
          <v-btn value="downloaded">
            <v-icon icon="mdi-check-circle" class="mr-1" size="16" color="success" />
            Downloaded
          </v-btn>
          <v-btn value="not-downloaded">
            <v-icon icon="mdi-cloud-download" class="mr-1" size="16" />
            Not Downloaded
          </v-btn>
        </v-btn-toggle>
      </v-col>
    </v-row>

    <!-- ─── Result count ────────────────────────────────────────────── -->
    <p class="text-caption text-medium-emphasis mb-4">
      Showing {{ filteredModels.length }} of {{ modelsStore.registry.length }} models
    </p>

    <!-- ─── Loading state ───────────────────────────────────────────── -->
    <div v-if="modelsStore.isLoading" class="d-flex justify-center py-8">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <!-- ─── Model cards grid ────────────────────────────────────────── -->
    <v-row v-else>
      <v-col
        v-for="model in filteredModels"
        :key="`${model.source}-${model.id}`"
        cols="12"
        md="6"
        xl="4"
      >
        <v-card height="100%" variant="outlined">
          <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1 d-flex align-center">
            {{ model.name }}
            <v-spacer />
            <!-- Download status badge -->
            <v-chip
              :color="model.status === 'downloaded' ? 'success' : 'grey'"
              size="x-small"
              :prepend-icon="model.status === 'downloaded' ? 'mdi-check-circle' : 'mdi-cloud-download'"
            >
              {{ model.status === 'downloaded' ? 'Ready' : 'Not downloaded' }}
            </v-chip>
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
              v-if="modelsStore.isModelDownloading(model.id, model.source)"
              :color="progressColor"
              size="x-small"
              class="mr-1"
              label
            >
              <v-progress-circular
                :model-value="downloadPercentageFor(model)"
                size="14"
                width="2"
                :color="progressColor"
                class="mr-1"
              />
              {{ downloadPercentageFor(model) }}%
            </v-chip>
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
            <!-- Model ID -->
            <code class="text-caption d-block mb-3 pa-1 rounded bg-surface-variant">
              {{ model.id }}
            </code>

            <!-- Description -->
            <p class="text-body-2 text-medium-emphasis mb-2">
              {{ model.description || 'No description available.' }}
            </p>

            <!-- Long description -->
            <p v-if="model.long_description" class="text-body-2 mb-3">
              {{ model.long_description }}
            </p>

            <!-- Size + source link row -->
            <div class="d-flex align-center flex-wrap gap-2 mb-3">
              <v-chip
                v-if="model.size"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-database"
              >
                {{ model.size }}
              </v-chip>
              <v-btn
                v-if="model.source_url"
                :href="model.source_url"
                target="_blank"
                rel="noopener"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-open-in-new"
              >
                View source
              </v-btn>
            </div>

            <!-- Tags -->
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
            <!-- Download button (only if not yet downloaded) -->
            <div v-if="model.status !== 'downloaded'" class="d-flex align-center gap-2">
              <v-progress-linear
                v-if="modelsStore.isModelDownloading(model.id, model.source)"
                :model-value="downloadPercentageFor(model)"
                height="8"
                width="200"
                color="blue"
                rounded
              >
                <template #default>
                  <strong class="text-caption">{{ downloadPercentageFor(model) }}%</strong>
                </template>
              </v-progress-linear>

              <v-btn
                v-if="!modelsStore.isModelDownloading(model.id, model.source)"
                color="primary"
                variant="tonal"
                size="small"
                prepend-icon="mdi-download"
                @click="handleDownload(model.id, model.source)"
              >
                Download
              </v-btn>

              <v-btn
                v-if="modelsStore.isModelDownloading(model.id, model.source)"
                color="primary"
                variant="tonal"
                size="small"
                prepend-icon="mdi-download"
                :loading="true"
                :disabled="true"
              >
                Downloading...
              </v-btn>
            </div>

            <v-spacer />

            <!-- Remove from registry -->
            <v-btn
              color="error"
              variant="text"
              size="small"
              icon="mdi-delete-outline"
              @click="handleRemove(model.id, model.source)"
            />
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <!-- Empty state -->
    <v-alert
      v-if="!modelsStore.isLoading && filteredModels.length === 0"
      type="info"
      variant="tonal"
      class="mt-4"
    >
      No models match the current filters.
    </v-alert>

    <!-- ─── Add Model Dialog ────────────────────────────────────────── -->
    <v-dialog v-model="showAddDialog" max-width="600" persistent>
      <v-card>
        <v-card-title class="text-h6">
          <v-icon icon="mdi-plus-circle" class="mr-2" />
          Add Model to Registry
        </v-card-title>

        <v-card-text>
          <v-text-field
            v-model="addForm.id"
            label="Model ID"
            hint="HuggingFace: org/repo (e.g. runwayml/stable-diffusion-v1-5) · CivitAI: version number (e.g. 128713)"
            persistent-hint
            variant="outlined"
            class="mb-3"
          />

          <v-text-field
            v-model="addForm.name"
            label="Display Name"
            variant="outlined"
            class="mb-3"
          />

          <v-row>
            <v-col cols="6">
              <v-select
                v-model="addForm.source"
                :items="[
                  { title: 'HuggingFace', value: 'huggingface' as ModelSource },
                  { title: 'CivitAI', value: 'civitai' as ModelSource },
                ]"
                label="Source"
                variant="outlined"
              />
            </v-col>
            <v-col cols="6">
              <v-select
                v-model="addForm.family"
                :items="[
                  { title: 'SD 1.x / 2.x', value: 'sd15' as ModelFamily },
                  { title: 'SDXL', value: 'sdxl' as ModelFamily },
                ]"
                label="Architecture"
                variant="outlined"
              />
            </v-col>
          </v-row>

          <v-text-field
            v-model="addForm.description"
            label="Description (optional)"
            hint="Brief one-liner shown in the card header"
            persistent-hint
            variant="outlined"
            class="mb-3"
          />

          <v-textarea
            v-model="addForm.long_description"
            label="Long description (optional)"
            hint="Detailed multi-sentence description of the model"
            persistent-hint
            variant="outlined"
            rows="3"
            class="mb-3"
          />

          <v-row>
            <v-col cols="12" sm="6">
              <v-text-field
                v-model="addForm.source_url"
                label="Source URL (optional)"
                placeholder="https://huggingface.co/..."
                variant="outlined"
              />
            </v-col>
            <v-col cols="12" sm="6">
              <v-text-field
                v-model="addForm.size"
                label="Approximate size (optional)"
                placeholder="e.g. ~2.1 GB"
                variant="outlined"
              />
            </v-col>
          </v-row>

          <v-text-field
            v-model="tagInput"
            label="Tags (comma-separated)"
            placeholder="e.g. photorealistic, portraits, fast"
            variant="outlined"
          />
        </v-card-text>

        <v-card-actions class="pa-4">
          <v-spacer />
          <v-btn variant="text" @click="showAddDialog = false; resetAddForm()">
            Cancel
          </v-btn>
          <v-btn
            color="primary"
            variant="elevated"
            :disabled="!addForm.id.trim() || !addForm.name.trim()"
            @click="handleAddModel"
          >
            Add Model
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
