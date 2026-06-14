<script setup lang="ts">
/**
 * Models management page.
 * Shows all registered models with download status, allows adding new models
 * and triggering downloads. The backend is the source of truth for the catalog.
 */
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useModelsStore } from '../stores/models'
import type { ModelFamily, ModelRegistryAddRequest, ModelRegistryEntry, ModelSource } from '../types'

const modelsStore = useModelsStore()

onMounted(() => {
  modelsStore.fetchRegistry()
  modelsStore.connectSSE()
})

onUnmounted(() => {
  modelsStore.disconnectSSE()
})

// ─── Filters ──────────────────────────────────────────────────────────────

const activeSource = ref<'all' | 'huggingface' | 'civitai'>('all')
const activeFamily = ref<'all' | 'sd15' | 'sdxl' | 'flux' | 'custom'>('all')
const activeDownloaded = ref<'all' | 'downloaded' | 'not-downloaded'>('all')
const activeCapability = ref<'all' | 'analysis' | 'txt2img' | 'img2img'>('all')

const filteredModels = computed(() =>
  modelsStore.registry.filter(m => {
    if (activeSource.value !== 'all' && m.source !== activeSource.value) return false
    if (activeFamily.value !== 'all' && m.family !== activeFamily.value) return false
    if (activeDownloaded.value === 'downloaded' && m.status !== 'downloaded') return false
    if (activeDownloaded.value === 'not-downloaded' && m.status === 'downloaded') return false
    if (activeCapability.value !== 'all' && !(m.capabilities?.includes(activeCapability.value))) return false
    return true
  }),
)

// ─── Add Model dialog ─────────────────────────────────────────────────────

const showAddDialog = ref(false)
const addForm = ref<ModelRegistryAddRequest>({
  model_id: '',
  name: '',
  source: 'huggingface',
  family: 'sd15',
  description: '',
  tags: [],
  source_url: '',
  capabilities: [],
})
const tagInput = ref('')

function resetAddForm() {
  addForm.value = {
    model_id: '',
    name: '',
    source: 'huggingface',
    family: 'sd15',
    description: '',
    tags: [],
    source_url: '',
    capabilities: [],
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

function handleDownload(modelId: string) {
  modelsStore.downloadModel(modelId)
}

function handleRemove(modelId: string) {
  modelsStore.removeModel(modelId)
}

// ─── UI helpers ───────────────────────────────────────────────────────────

function familyColor(family: string): string {
  const colors: Record<string, string> = {
    sd15: 'blue',
    sdxl: 'purple',
    flux: 'orange',
    custom: 'grey',
  }
  return colors[family] || 'grey'
}

function downloadPercentageFor(model: ModelRegistryEntry): number {
  return model.download_progress || 0
}

function statusChipColor(model: ModelRegistryEntry): string {
  if (model.status === 'downloaded') return 'success'
  if (model.status === 'downloading' || model.status === 'download_paused') return 'warning'
  if (model.status === 'error') return 'error'
  return 'grey'
}

function statusChipIcon(model: ModelRegistryEntry): string {
  if (model.status === 'downloaded') return 'mdi-check-circle'
  if (model.status === 'downloading' || model.status === 'download_paused') return 'mdi-cloud-download'
  if (model.status === 'error') return 'mdi-alert-circle'
  return 'mdi-cloud-off'
}

function statusChipLabel(model: ModelRegistryEntry): string {
  if (model.status === 'downloaded') return 'Ready'
  if (model.status === 'downloading') return 'Downloading...'
  if (model.status === 'download_paused') return 'Paused'
  if (model.status === 'error') return 'Error'
  return 'Not downloaded'
}
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
           <v-btn value="flux">Flux</v-btn>
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

      <!-- Capability filter -->
      <v-col cols="12" sm="auto">
        <v-btn-toggle
          v-model="activeCapability"
          mandatory
          variant="outlined"
          density="compact"
          divided
        >
          <v-btn value="all">All caps</v-btn>
          <v-btn value="analysis">
            <v-icon icon="mdi-eye" class="mr-1" size="16" />
            Analysis
          </v-btn>
          <v-btn value="txt2img">
            <v-icon icon="mdi-text-box" class="mr-1" size="16" />
            Txt2Img
          </v-btn>
          <v-btn value="img2img">
            <v-icon icon="mdi-image-edit" class="mr-1" size="16" />
            Img2Img
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
            {{ model.preferred_name || model.name }}
            <v-spacer />
             <!-- Download status badge -->
             <v-chip
               :color="statusChipColor(model)"
               size="x-small"
               :prepend-icon="statusChipIcon(model)"
             >
               {{ statusChipLabel(model) }}
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
              {{ model.id }}
            </v-chip>

            <!-- VRAM requirements -->
            <div v-if="model.recommended_vram_min_gb || model.recommended_vram_max_gb" class="d-flex align-center flex-wrap gap-2 mb-3">
              <v-chip
                v-if="model.recommended_vram_min_gb"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-memory"
              >
                Min VRAM: {{ model.recommended_vram_min_gb }} GB
              </v-chip>
              <v-chip
                v-if="model.recommended_vram_max_gb"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-memory"
              >
                Max VRAM: {{ model.recommended_vram_max_gb }} GB
              </v-chip>
            </div>

            <!-- Description -->
            <p class="text-body-2 text-medium-emphasis mb-2">
              {{ model.description || 'No description available.' }}
            </p>

            <!-- Size + source link row -->
            <div class="d-flex align-center flex-wrap gap-2 mb-3">
              <v-chip
                v-if="model.total_size_bytes"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-database"
              >
                {{ Math.round(model.total_size_bytes / 1024 / 1024) }} MB
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

            <!-- Capabilities -->
            <div v-if="model.capabilities?.length" class="mb-1">
              <v-chip
                v-for="cap in model.capabilities"
                :key="cap"
                size="x-small"
                :color="cap === 'analysis' ? 'cyan' : cap === 'txt2img' ? 'purple' : 'teal'"
                class="mr-1 mb-1"
              >
                {{ cap }}
              </v-chip>
            </div>
          </v-card-subtitle>

          <v-card-actions class="px-4 pb-4">
            <!-- Download button (only if not yet downloaded) -->
            <div v-if="model.status !== 'downloaded'" class="d-flex align-center gap-2">
              <v-progress-linear
                v-if="modelsStore.isModelDownloading(model.model_id)"
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
                v-if="!modelsStore.isModelDownloading(model.model_id)"
                color="primary"
                variant="tonal"
                size="small"
                prepend-icon="mdi-download"
                @click="handleDownload(model.model_id)"
              >
                Download
              </v-btn>

              <v-btn
                v-if="modelsStore.isModelDownloading(model.model_id)"
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
              @click="handleRemove(model.model_id)"
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
            v-model="addForm.model_id"
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
                  { title: 'SD 1.x / 2.x — Classic Stable Diffusion', value: 'sd15' as ModelFamily },
                  { title: 'SDXL — Stable Diffusion XL (high-res)', value: 'sdxl' as ModelFamily },
                  { title: 'Flux — Next-gen architecture (500M+ params)', value: 'flux' as ModelFamily },
                  { title: 'Custom — Unknown or modified architecture', value: 'custom' as ModelFamily },
                ]"
                label="Architecture"
                variant="outlined"
              />
            </v-col>
          </v-row>

          <v-textarea
            v-model="addForm.description"
            label="Description (optional)"
            hint="Describe what the model does and its strengths"
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
                v-model="addForm.variant"
                label="Variant (optional)"
                placeholder="e.g. fp16"
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
            :disabled="!addForm.model_id.trim() || !addForm.name.trim()"
            @click="handleAddModel"
          >
            Add Model
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
