<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useModelsStore } from '../stores/models'
import type { ModelFamily, ModelRegistryAddRequest, ModelRegistryEntry, ModelRegistryUpdateRequest, ModelSource } from '../types'

const modelsStore = useModelsStore()

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

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
const activeType = ref<string>('all')
const activeTags = ref<string[]>([])

const MODEL_TYPES = [
  { value: 'all', label: 'All types' },
  { value: 'base_diffusion', label: 'Base' },
  { value: 'controlnet', label: 'ControlNet' },
  { value: 't2i_adapter', label: 'T2I Adapter' },
  { value: 'lora', label: 'LoRA' },
  { value: 'ip_adapter', label: 'IP-Adapter' },
  { value: 'vae', label: 'VAE' },
  { value: 'upscaler', label: 'Upscaler' },
  { value: 'face_restore', label: 'Face Restore' },
  { value: 'vision_language', label: 'Vision LM' },
]

const allTags = computed(() => modelsStore.allTags)

// Maps each UI family key to the family values and compatible_bases values stored in the DB
const FAMILY_NAMES: Record<string, string[]> = {
  sd15: ['sd15', 'stable_diffusion_1', 'stable_diffusion_2', 'stable_diffusion_edit'],
  sdxl: ['sdxl', 'stable_diffusion_xl'],
  flux: ['flux'],
  custom: ['custom'],
}
const FAMILY_COMPAT_BASES: Record<string, string[]> = {
  sd15: ['sd15', 'sd1.5', 'sd1', 'sd2'],
  sdxl: ['sdxl'],
  flux: ['flux'],
}

const filteredModels = computed(() =>
  modelsStore.registry.filter(m => {
    if (activeSource.value !== 'all' && m.source !== activeSource.value) return false
    if (activeFamily.value !== 'all') {
      const fam = activeFamily.value
      const validFamilies = FAMILY_NAMES[fam] ?? [fam]
      const validBases = FAMILY_COMPAT_BASES[fam] ?? []
      const matchesFamily = validFamilies.includes(m.family)
      const matchesBases = validBases.length > 0 && m.compatible_bases?.some(b => validBases.includes(b))
      if (!matchesFamily && !matchesBases) return false
    }
    if (activeDownloaded.value === 'downloaded' && m.status !== 'downloaded') return false
    if (activeDownloaded.value === 'not-downloaded' && m.status === 'downloaded') return false
    if (activeType.value !== 'all' && m.model_type !== activeType.value) return false
    if (activeTags.value.length > 0 && !activeTags.value.every(t => m.tags?.includes(t))) return false
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

// ─── Edit Model dialog ────────────────────────────────────────────────────

const showEditDialog = ref(false)
const editingModelId = ref('')
const editForm = ref<ModelRegistryUpdateRequest>({})
const editTagInput = ref('')

function openEditDialog(model: ModelRegistryEntry) {
  editingModelId.value = model.model_id
  editForm.value = {
    name: model.name,
    preferred_name: model.preferred_name ?? '',
    source: model.source as ModelSource,
    family: model.family as ModelFamily,
    variant: model.variant ?? '',
    description: model.description ?? '',
    source_url: model.source_url ?? '',
    tags: [...(model.tags ?? [])],
    notes: model.notes ?? '',
  }
  editTagInput.value = (model.tags ?? []).join(', ')
  showEditDialog.value = true
}

function handleEditModel() {
  const payload: ModelRegistryUpdateRequest = { ...editForm.value }
  if (editTagInput.value.trim()) {
    payload.tags = editTagInput.value.split(',').map(t => t.trim()).filter(Boolean)
  }
  modelsStore.updateModel(editingModelId.value, payload)
    .then(() => {
      showEditDialog.value = false
    })
    .catch(() => { /* error handled by store */ })
}

// ─── Actions ──────────────────────────────────────────────────────────────

function handleDownload(modelId: string) {
  modelsStore.downloadModel(modelId)
}

// ─── Confirmation dialog ───────────────────────────────────────────────────

type ConfirmAction = 'delete_files' | 'remove'
type BulkConfirmAction = 'purge_all_files' | 'remove_all'

const confirmDialog = ref(false)
const confirmTarget = ref<ModelRegistryEntry | null>(null)
const confirmAction = ref<ConfirmAction>('delete_files')

const confirmConfig = {
  delete_files: {
    title: 'Delete downloaded files?',
    icon: 'mdi-folder-remove',
    iconColor: 'warning',
    confirmColor: 'warning',
    confirmLabel: 'Delete files',
    body: 'This deletes all downloaded files for this model from disk.',
    detail: 'The registry entry is kept. You can re-download the model at any time.',
  },
  remove: {
    title: 'Remove from registry?',
    icon: 'mdi-delete-alert',
    iconColor: 'error',
    confirmColor: 'error',
    confirmLabel: 'Remove permanently',
    body: 'This permanently deletes the model from the registry and from disk.',
    detail: 'All downloaded files will be lost. You will need to re-add and re-download the model to use it. This cannot be undone.',
  },
} as const

function openConfirm(model: ModelRegistryEntry, action: ConfirmAction) {
  confirmTarget.value = model
  confirmAction.value = action
  confirmDialog.value = true
}

function handleConfirm() {
  const model = confirmTarget.value
  if (!model) return
  confirmDialog.value = false

  if (confirmAction.value === 'delete_files') {
    modelsStore.deleteModelFiles(model.model_id)
  } else {
    modelsStore.removeModel(model.model_id)
  }
  confirmTarget.value = null
}

// ─── Bulk confirmation dialog ──────────────────────────────────────────────

const bulkConfirmDialog = ref(false)
const bulkAction = ref<BulkConfirmAction>('purge_all_files')

const bulkConfirmConfig = {
  purge_all_files: {
    title: 'Purge all model files?',
    icon: 'mdi-folder-remove',
    iconColor: 'warning',
    confirmColor: 'warning',
    confirmLabel: 'Purge all files',
    body: 'This deletes downloaded files for every model from disk and resets all statuses to "Not downloaded".',
    detail: 'Registry entries are kept. You can re-download any model at any time.',
  },
  remove_all: {
    title: 'Remove all models?',
    icon: 'mdi-delete-alert',
    iconColor: 'error',
    confirmColor: 'error',
    confirmLabel: 'Remove all permanently',
    body: 'This permanently removes every model from the registry and deletes all files from disk.',
    detail: 'You will need to re-add and re-download every model. This cannot be undone.',
  },
} as const

function openBulkConfirm(action: BulkConfirmAction) {
  bulkAction.value = action
  bulkConfirmDialog.value = true
}

function handleBulkConfirm() {
  bulkConfirmDialog.value = false
  if (bulkAction.value === 'purge_all_files') {
    modelsStore.purgeAllModelFiles()
  } else {
    modelsStore.removeAllModels()
  }
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

function modelTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    base_diffusion: 'Base Model',
    controlnet: 'ControlNet',
    t2i_adapter: 'T2I Adapter',
    lora: 'LoRA',
    ip_adapter: 'IP-Adapter',
    vae: 'VAE',
    upscaler: 'Upscaler',
    face_restore: 'Face Restore',
    vision_language: 'Vision LM',
  }
  return labels[type] ?? type
}

function modelTypeColor(type: string): string {
  const colors: Record<string, string> = {
    base_diffusion: 'primary',
    controlnet: 'cyan',
    t2i_adapter: 'teal',
    lora: 'green',
    ip_adapter: 'indigo',
    vae: 'blue-grey',
    upscaler: 'deep-orange',
    face_restore: 'pink',
    vision_language: 'deep-purple',
  }
  return colors[type] ?? 'grey'
}

function modelTypeIcon(type: string): string {
  const icons: Record<string, string> = {
    base_diffusion: 'mdi-image-auto-adjust',
    controlnet: 'mdi-tune',
    t2i_adapter: 'mdi-pencil-ruler',
    lora: 'mdi-layers-plus',
    ip_adapter: 'mdi-image-frame',
    vae: 'mdi-vector-combine',
    upscaler: 'mdi-arrow-expand-all',
    face_restore: 'mdi-face-recognition',
    vision_language: 'mdi-eye-outline',
  }
  return icons[type] ?? 'mdi-cube-outline'
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
      <div class="d-flex gap-2">
        <v-tooltip location="bottom" text="Delete all downloaded files, keep registry entries">
          <template #activator="{ props: tooltipProps }">
            <v-btn
              v-if="modelsStore.registry.length"
              v-bind="tooltipProps"
              variant="outlined"
              color="warning"
              prepend-icon="mdi-folder-remove"
              @click="openBulkConfirm('purge_all_files')"
            >
              Purge all files
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="bottom" text="Remove all models from registry and disk">
          <template #activator="{ props: tooltipProps }">
            <v-btn
              v-if="modelsStore.registry.length"
              v-bind="tooltipProps"
              variant="outlined"
              color="error"
              prepend-icon="mdi-delete-sweep"
              @click="openBulkConfirm('remove_all')"
            >
              Remove all
            </v-btn>
          </template>
        </v-tooltip>
        <v-btn
          color="primary"
          prepend-icon="mdi-plus"
          @click="showAddDialog = true"
        >
          Add Model
        </v-btn>
      </div>
    </div>
    <p class="text-body-2 text-medium-emphasis mb-6">
      Manage your model catalog — download models to use them in generation. Not-downloaded models appear in generation selects as disabled options.
    </p>

    <!-- ─── Filter bar ──────────────────────────────────────────────── -->
    <v-row class="mb-2" align="center">
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
    </v-row>

    <!-- Model type filter (full row) -->
    <v-row class="mb-2" align="center">
      <v-col cols="12">
        <v-btn-toggle
          v-model="activeType"
          mandatory
          variant="outlined"
          density="compact"
          divided
        >
          <v-btn
            v-for="t in MODEL_TYPES"
            :key="t.value"
            :value="t.value"
          >
            <v-icon
              v-if="t.value !== 'all'"
              :icon="modelTypeIcon(t.value)"
              class="mr-1"
              size="14"
            />
            {{ t.label }}
          </v-btn>
        </v-btn-toggle>
      </v-col>
    </v-row>

    <!-- Tag multi-select filter -->
    <v-row class="mb-4" align="center">
      <v-col cols="12" sm="6" md="4">
        <v-select
          v-model="activeTags"
          :items="allTags"
          label="Filter by tags"
          multiple
          chips
          closable-chips
          clearable
          variant="outlined"
          density="compact"
          hide-details
        />
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
          <!-- Title row: name + status -->
          <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1 d-flex align-center">
            {{ model.preferred_name || model.name }}
            <v-spacer />
            <v-chip
              :color="statusChipColor(model)"
              size="x-small"
              :prepend-icon="statusChipIcon(model)"
            >
              {{ statusChipLabel(model) }}
            </v-chip>
          </v-card-title>

          <!-- Type + architecture row -->
          <v-card-subtitle class="pt-0 pb-2">
            <div class="d-flex align-center flex-wrap gap-2">
              <v-chip
                :color="modelTypeColor(model.model_type)"
                size="small"
                :prepend-icon="modelTypeIcon(model.model_type)"
                label
              >
                {{ modelTypeLabel(model.model_type) }}
              </v-chip>
              <template v-if="model.compatible_bases?.length">
                <v-divider vertical class="mx-1" />
                <v-chip
                  v-for="base in model.compatible_bases"
                  :key="base"
                  :color="familyColor(base)"
                  size="x-small"
                  variant="tonal"
                  label
                >
                  {{ base }}
                </v-chip>
              </template>
            </div>
          </v-card-subtitle>

          <v-card-text class="pt-0 pb-2">
            <!-- VRAM requirements -->
            <div v-if="model.recommended_vram_min_gb || model.recommended_vram_max_gb" class="d-flex align-center flex-wrap gap-2 mb-3">
              <v-chip
                v-if="model.recommended_vram_min_gb"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-memory"
              >
                Min {{ model.recommended_vram_min_gb }} GB
              </v-chip>
              <v-chip
                v-if="model.recommended_vram_max_gb"
                size="x-small"
                variant="tonal"
                prepend-icon="mdi-memory"
              >
                Max {{ model.recommended_vram_max_gb }} GB
              </v-chip>
            </div>

            <!-- Short description -->
            <p class="text-body-2 text-medium-emphasis mb-2">
              {{ model.short_description || model.description || 'No description available.' }}
            </p>

            <p class="mt-2 mb-5">{{ model.description }}</p>

            <!-- Size + source link row -->
            <div class="d-flex align-center flex-wrap gap-2 mb-2">
              <v-chip
                v-if="model.total_size_bytes"
                size="x-small"
                variant="tonal"
                :prepend-icon="model.status === 'downloaded' ? 'mdi-harddisk' : 'mdi-download-outline'"
                :color="model.status === 'downloaded' ? 'success' : undefined"
              >
                {{ model.status === 'downloaded' ? '' : '~' }}{{ formatBytes(model.total_size_bytes) }}
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
            <div v-if="model.tags?.length">
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
              <div
                v-if="modelsStore.isModelDownloading(model.model_id)"
                class="d-flex flex-column gap-1"
                style="width: 200px"
              >
                <v-progress-linear
                  :model-value="modelsStore.downloadProgress.get(model.model_id)?.pct ?? 0"
                  :indeterminate="!modelsStore.downloadProgress.has(model.model_id)"
                  height="8"
                  color="blue"
                  rounded
                />
                <div class="text-caption text-center">
                  <template v-if="modelsStore.downloadProgress.has(model.model_id)">
                    {{ modelsStore.downloadProgress.get(model.model_id)!.pct }}%
                    <template v-if="modelsStore.downloadProgress.get(model.model_id)!.totalFiles">
                      · file {{ modelsStore.downloadProgress.get(model.model_id)!.fileIndex }}/{{ modelsStore.downloadProgress.get(model.model_id)!.totalFiles }}
                    </template>
                  </template>
                  <template v-else>Preparing…</template>
                </div>
              </div>

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
            </div>

            <v-spacer />

            <!-- Edit model metadata -->
            <v-tooltip location="top" text="Edit model metadata">
              <template #activator="{ props: tooltipProps }">
                <v-btn
                  v-bind="tooltipProps"
                  color="primary"
                  variant="text"
                  size="small"
                  icon="mdi-pencil-outline"
                  @click="openEditDialog(model)"
                />
              </template>
            </v-tooltip>

            <!-- Delete files only (only shown when files are present) -->
            <v-tooltip location="top" text="Delete downloaded files — keeps registry entry">
              <template #activator="{ props: tooltipProps }">
                <v-btn
                  v-if="model.status === 'downloaded'"
                  v-bind="tooltipProps"
                  color="warning"
                  variant="text"
                  size="small"
                  icon="mdi-folder-remove"
                  @click="openConfirm(model, 'delete_files')"
                />
              </template>
            </v-tooltip>

            <!-- Remove from registry + delete files (permanent) -->
            <v-tooltip location="top" text="Remove from registry and delete files">
              <template #activator="{ props: tooltipProps }">
                <v-btn
                  v-bind="tooltipProps"
                  color="error"
                  variant="text"
                  size="small"
                  icon="mdi-delete-outline"
                  @click="openConfirm(model, 'remove')"
                />
              </template>
            </v-tooltip>
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

    <!-- ─── Confirm dialog (unload / remove) ─────────────────────── -->
    <v-dialog v-model="confirmDialog" max-width="480" persistent>
      <v-card v-if="confirmTarget">
        <v-card-title class="d-flex align-center gap-2 pt-4 pb-1">
          <v-icon
            :icon="confirmConfig[confirmAction].icon"
            :color="confirmConfig[confirmAction].iconColor"
          />
          {{ confirmConfig[confirmAction].title }}
        </v-card-title>

        <v-card-text class="pb-2">
          <p class="text-body-1 font-weight-medium mb-2">
            {{ confirmTarget.preferred_name || confirmTarget.name }}
          </p>
          <p class="text-body-2 mb-2">{{ confirmConfig[confirmAction].body }}</p>
          <v-alert
            :type="confirmAction === 'remove' ? 'error' : 'warning'"
            variant="tonal"
            density="compact"
            class="text-body-2"
          >
            {{ confirmConfig[confirmAction].detail }}
          </v-alert>
        </v-card-text>

        <v-card-actions class="pa-4 pt-2">
          <v-spacer />
          <v-btn variant="text" @click="confirmDialog = false; confirmTarget = null">
            Cancel
          </v-btn>
          <v-btn
            :color="confirmConfig[confirmAction].confirmColor"
            variant="elevated"
            @click="handleConfirm"
          >
            {{ confirmConfig[confirmAction].confirmLabel }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- ─── Bulk Confirm dialog (purge all / remove all) ───────────── -->
    <v-dialog v-model="bulkConfirmDialog" max-width="480" persistent>
      <v-card>
        <v-card-title class="d-flex align-center gap-2 pt-4 pb-1">
          <v-icon
            :icon="bulkConfirmConfig[bulkAction].icon"
            :color="bulkConfirmConfig[bulkAction].iconColor"
          />
          {{ bulkConfirmConfig[bulkAction].title }}
        </v-card-title>

        <v-card-text class="pb-2">
          <p class="text-body-2 mb-2">{{ bulkConfirmConfig[bulkAction].body }}</p>
          <v-alert
            :type="bulkAction === 'remove_all' ? 'error' : 'warning'"
            variant="tonal"
            density="compact"
            class="text-body-2"
          >
            {{ bulkConfirmConfig[bulkAction].detail }}
          </v-alert>
        </v-card-text>

        <v-card-actions class="pa-4 pt-2">
          <v-spacer />
          <v-btn variant="text" @click="bulkConfirmDialog = false">Cancel</v-btn>
          <v-btn
            :color="bulkConfirmConfig[bulkAction].confirmColor"
            variant="elevated"
            @click="handleBulkConfirm"
          >
            {{ bulkConfirmConfig[bulkAction].confirmLabel }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- ─── Edit Model Dialog ───────────────────────────────────────── -->
    <v-dialog v-model="showEditDialog" max-width="600" persistent>
      <v-card>
        <v-card-title class="text-h6">
          <v-icon icon="mdi-pencil" class="mr-2" />
          Edit Model
        </v-card-title>

        <v-card-text>
          <v-text-field
            :model-value="editingModelId"
            label="Model ID"
            variant="outlined"
            readonly
            class="mb-3"
            hint="Model ID cannot be changed"
            persistent-hint
          />

          <v-text-field
            v-model="editForm.name"
            label="Display Name"
            variant="outlined"
            class="mb-3"
          />

          <v-text-field
            v-model="editForm.preferred_name"
            label="Preferred Name (optional override)"
            variant="outlined"
            class="mb-3"
          />

          <v-row>
            <v-col cols="6">
              <v-select
                v-model="editForm.source"
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
                v-model="editForm.family"
                :items="[
                  { title: 'SD 1.x / 2.x', value: 'sd15' as ModelFamily },
                  { title: 'SDXL', value: 'sdxl' as ModelFamily },
                  { title: 'Flux', value: 'flux' as ModelFamily },
                  { title: 'Custom', value: 'custom' as ModelFamily },
                ]"
                label="Architecture"
                variant="outlined"
              />
            </v-col>
          </v-row>

          <v-textarea
            v-model="editForm.description"
            label="Description (optional)"
            variant="outlined"
            rows="3"
            class="mb-3"
          />

          <v-row>
            <v-col cols="12" sm="6">
              <v-text-field
                v-model="editForm.source_url"
                label="Source URL"
                placeholder="https://huggingface.co/..."
                variant="outlined"
              />
            </v-col>
            <v-col cols="12" sm="6">
              <v-text-field
                v-model="editForm.variant"
                label="Variant (optional)"
                placeholder="e.g. fp16"
                variant="outlined"
              />
            </v-col>
          </v-row>

          <v-text-field
            v-model="editTagInput"
            label="Tags (comma-separated)"
            placeholder="e.g. photorealistic, portrait, fast"
            variant="outlined"
            class="mb-3"
          />

          <v-textarea
            v-model="editForm.notes"
            label="Notes (optional)"
            variant="outlined"
            rows="2"
          />
        </v-card-text>

        <v-card-actions class="pa-4">
          <v-spacer />
          <v-btn variant="text" @click="showEditDialog = false">
            Cancel
          </v-btn>
          <v-btn
            color="primary"
            variant="elevated"
            :disabled="!editForm.name?.trim()"
            @click="handleEditModel"
          >
            Save Changes
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

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
            label="Tags (comma-separated, snake_case)"
            placeholder="e.g. photorealistic, portrait, fast"
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
