<script setup lang="ts">
/**
 * Image Upscale workspace — three-step pipeline UI.
 *
 * Step 1 (optional): Enhancement — img2img quality improvement before upscaling
 * Step 2 (required): Upscale     — SD x4 diffusion upscaler (model select)
 * Step 3 (optional): Face Restore — GFPGAN face sharpening after upscaling
 *
 * Each optional step has its own enable/disable toggle via PipelineStepCard.
 * Model selects are filtered by capability so only relevant models appear.
 */
import { computed, onMounted, ref } from 'vue'
import { diffusionApi } from '../api/diffusion'
import { useModelsStore } from '../stores/models'
import { useNotificationStore } from '../stores/notifications'
import PipelineStepCard from '../components/upscale/PipelineStepCard.vue'
import type { ArtifactEntry, ModelRegistryEntry } from '../types'

const modelsStore = useModelsStore()
const notifications = useNotificationStore()

onMounted(() => modelsStore.fetchRegistry())

// ─── Model lists (filtered by capability) ─────────────────────────────────

// Capabilities that mark a model as specialized (not suitable for general enhancement)
const SPECIALIZED_CAPS = ['sketch_to_ink', 'recolor', 'recolor_image', 'upscale_image', 'face_restore', 'describe']

/** All general-purpose img2img models; not-downloaded shown as disabled options */
const img2imgModels = computed<ModelRegistryEntry[]>(() =>
  modelsStore.registry.filter((m) => {
    const caps = m.capabilities ?? []
    return caps.includes('image_to_image') && !caps.some((c) => SPECIALIZED_CAPS.includes(c))
  }),
)

/** All upscale models; not-downloaded shown as disabled options */
const upscaleModels = computed<ModelRegistryEntry[]>(() =>
  modelsStore.registry.filter((m) => m.capabilities?.includes('upscale_image')),
)

/** All face_restore models; not-downloaded shown as disabled options */
const faceRestoreModels = computed<ModelRegistryEntry[]>(() =>
  modelsStore.registry.filter((m) => m.capabilities?.includes('face_restore')),
)

// ─── Step enable/disable ───────────────────────────────────────────────────

const step1Enabled = ref(false) // Enhancement is off by default — most images don't need it
const step3Enabled = ref(false) // Face restore is off by default — most images have no faces

// ─── Step 1 params ────────────────────────────────────────────────────────

const enhanceModelId = ref<string>('')
const enhanceStrength = ref(0.4)

// ─── Step 2 params (required) ─────────────────────────────────────────────

const upscaleModelId = ref<string>('')
const scaleFactor = ref(2.0)
const prompt = ref('')
const noiseLevel = ref(20)
const numSteps = ref(20)

// ─── Step 3 params ────────────────────────────────────────────────────────

const faceRestoreModelId = ref<string>('')
const faceRestoreFidelity = ref(0.5)

// ─── Image upload ──────────────────────────────────────────────────────────

const imageFile = ref<File | null>(null)
const imagePreviewUrl = ref<string | null>(null)

function onFileChange(files: File[] | File | null) {
  const file = Array.isArray(files) ? files[0] : files
  imageFile.value = file ?? null
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
  imagePreviewUrl.value = file ? URL.createObjectURL(file) : null
}

// ─── Job tracking ──────────────────────────────────────────────────────────

const isSubmitting = ref(false)
const jobId = ref<string | null>(null)
const jobStatus = ref<string | null>(null)
const resultArtifact = ref<ArtifactEntry | null>(null)
let _pollTimer: ReturnType<typeof setTimeout> | null = null

async function submit() {
  if (!imageFile.value || !upscaleModelId.value) return
  isSubmitting.value = true
  jobId.value = null
  jobStatus.value = null
  resultArtifact.value = null

  try {
    const res = await diffusionApi.submitUpscale(upscaleModelId.value, imageFile.value, {
      scaleFactor: scaleFactor.value,
      prompt: prompt.value,
      noiseLevel: noiseLevel.value,
      numInferenceSteps: numSteps.value,
      enhanceModelId: step1Enabled.value ? enhanceModelId.value || undefined : undefined,
      enhanceStrength: enhanceStrength.value,
      faceRestoreModelId: step3Enabled.value ? faceRestoreModelId.value || undefined : undefined,
      faceRestoreFidelity: faceRestoreFidelity.value,
    })
    jobId.value = res.job_id
    jobStatus.value = 'pending'
    notifications.push('info', 'Upscale job submitted — processing…')
    schedulePoll()
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Failed to submit upscale job'
    notifications.push('error', msg)
  } finally {
    isSubmitting.value = false
  }
}

function schedulePoll() {
  _pollTimer = setTimeout(pollJob, 2000)
}

async function pollJob() {
  if (!jobId.value) return
  try {
    const job = await diffusionApi.getJobStatus(jobId.value)
    jobStatus.value = job.status

    if (job.status === 'completed') {
      await loadResult()
    } else if (job.status === 'failed') {
      notifications.push('error', `Upscale failed: ${job.error ?? 'unknown error'}`)
    } else {
      schedulePoll()
    }
  } catch {
    schedulePoll()
  }
}

async function loadResult() {
  if (!jobId.value) return
  try {
    const page = await diffusionApi.getArtifacts({ limit: 50 })
    const artifact = page.items.find((a) => a.job_id === jobId.value)
    if (artifact) {
      resultArtifact.value = artifact
      notifications.push('success', 'Upscale complete!')
    }
  } catch {
    notifications.push('error', 'Job completed but failed to load result')
  }
}

function reset() {
  if (_pollTimer) clearTimeout(_pollTimer)
  jobId.value = null
  jobStatus.value = null
  resultArtifact.value = null
  imageFile.value = null
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
  imagePreviewUrl.value = null
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const artifactUrl = computed(() =>
  resultArtifact.value ? `/api/v1/artifacts/${resultArtifact.value.id}/file` : null,
)

const canSubmit = computed(
  () =>
    !!imageFile.value &&
    !!upscaleModelId.value &&
    (!step1Enabled.value || !!enhanceModelId.value) &&
    (!step3Enabled.value || !!faceRestoreModelId.value) &&
    !isSubmitting.value &&
    jobStatus.value !== 'running',
)

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

</script>

<template>
  <v-row>
    <!-- ── Left panel: pipeline controls ── -->
    <v-col cols="12" md="5" lg="4">
      <div class="d-flex align-center mb-4">
        <v-icon icon="mdi-image-size-select-large" color="deep-purple" class="mr-2" />
        <span class="text-h6 font-weight-bold">Upscale Pipeline</span>
      </div>

      <!-- Image upload (shared across all steps) -->
      <v-card class="pa-4 mb-4" elevation="1">
        <div class="text-caption text-medium-emphasis mb-1">Source image</div>
        <v-file-input
          accept="image/*"
          label="Upload image"
          density="compact"
          variant="outlined"
          prepend-icon=""
          prepend-inner-icon="mdi-image-plus"
          @update:model-value="onFileChange"
        />
        <v-img
          v-if="imagePreviewUrl"
          :src="imagePreviewUrl"
          max-height="140"
          class="rounded mt-2"
          cover
        />
      </v-card>

      <!-- ── Step 1: Enhancement (optional) ── -->
      <PipelineStepCard
        :step="1"
        title="Enhancement"
        icon="mdi-auto-fix"
        :optional="true"
        v-model:enabled="step1Enabled"
        color="indigo"
        class="mb-4"
      >
        <div class="text-caption text-medium-emphasis mb-2">
          Runs img2img before upscaling to clean up noise or blur — useful for very
          degraded sources. Low strength preserves structure.
        </div>

        <v-select
          v-model="enhanceModelId"
          :items="img2imgModels"
          item-title="preferred_name"
          item-value="model_id"
          label="Enhancement model"
          density="compact"
          variant="outlined"
          no-data-text="No img2img models in registry"
          :disabled="!step1Enabled"
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

        <div class="d-flex justify-space-between mb-1 mt-2">
          <span class="text-caption text-medium-emphasis">Strength</span>
          <span class="text-caption font-weight-bold">{{ enhanceStrength }}</span>
        </div>
        <v-slider
          v-model="enhanceStrength"
          :min="0.05"
          :max="0.95"
          :step="0.05"
          density="compact"
          color="indigo"
          :disabled="!step1Enabled"
          thumb-label
          hint="Lower = subtle cleanup, higher = more change"
          persistent-hint
        />
      </PipelineStepCard>

      <!-- ── Step 2: Upscale (required) ── -->
      <PipelineStepCard
        :step="2"
        title="Upscale"
        icon="mdi-image-size-select-large"
        :optional="false"
        color="deep-purple"
        class="mb-4"
      >
        <v-select
          v-model="upscaleModelId"
          :items="upscaleModels"
          item-title="preferred_name"
          item-value="model_id"
          label="Upscale model"
          density="compact"
          variant="outlined"
          no-data-text="No upscale models in registry"
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

        <!-- Scale factor -->
        <div class="d-flex justify-space-between mb-1 mt-2">
          <span class="text-caption text-medium-emphasis">Scale factor</span>
          <span class="text-caption font-weight-bold">{{ scaleFactor }}×</span>
        </div>
        <v-slider
          v-model="scaleFactor"
          :min="1.0"
          :max="8.0"
          :step="0.5"
          density="compact"
          color="deep-purple"
          thumb-label
        />

        <!-- Noise level -->
        <div class="d-flex justify-space-between mb-1 mt-2">
          <span class="text-caption text-medium-emphasis">Noise level</span>
          <span class="text-caption font-weight-bold">{{ noiseLevel }}</span>
        </div>
        <v-slider
          v-model="noiseLevel"
          :min="0"
          :max="350"
          :step="5"
          density="compact"
          color="deep-purple"
          thumb-label
          hint="Higher = more AI-generated detail; lower = more faithful to source"
          persistent-hint
        />

        <!-- Steps -->
        <div class="d-flex justify-space-between mb-1 mt-3">
          <span class="text-caption text-medium-emphasis">Inference steps</span>
          <span class="text-caption font-weight-bold">{{ numSteps }}</span>
        </div>
        <v-slider
          v-model="numSteps"
          :min="5"
          :max="50"
          :step="1"
          density="compact"
          color="deep-purple"
          thumb-label
        />

        <!-- Advanced: prompt (rarely needed) -->
        <v-expansion-panels variant="accordion" class="mt-3" elevation="0">
          <v-expansion-panel>
            <v-expansion-panel-title class="text-caption text-medium-emphasis px-0 py-1" style="min-height: 32px">
              Advanced
            </v-expansion-panel-title>
            <v-expansion-panel-text class="px-0">
              <v-textarea
                v-model="prompt"
                label="Prompt"
                placeholder="e.g. portrait, sharp focus, realistic skin"
                density="compact"
                variant="outlined"
                rows="2"
                auto-grow
                hint="Only useful on very degraded images — tells the upscaler what kind of content to expect so it hallucinates better detail. Leave empty for normal use."
                persistent-hint
              />
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
      </PipelineStepCard>

      <!-- ── Step 3: Face Restore (optional) ── -->
      <PipelineStepCard
        :step="3"
        title="Face Restore"
        icon="mdi-face-recognition"
        :optional="true"
        v-model:enabled="step3Enabled"
        color="teal"
        class="mb-4"
      >
        <div class="text-caption text-medium-emphasis mb-2">
          GFPGAN detects all faces and sharpens them without affecting the rest of
          the image. Only enable if the image contains faces.
        </div>

        <v-select
          v-model="faceRestoreModelId"
          :items="faceRestoreModels"
          item-title="preferred_name"
          item-value="model_id"
          label="Face restore model"
          density="compact"
          variant="outlined"
          no-data-text="No face restore models in registry"
          :disabled="!step3Enabled"
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

        <!-- Fidelity slider -->
        <div class="d-flex justify-space-between mb-1 mt-2">
          <span class="text-caption text-medium-emphasis">Fidelity</span>
          <span class="text-caption font-weight-bold">{{ faceRestoreFidelity }}</span>
        </div>
        <v-slider
          v-model="faceRestoreFidelity"
          :min="0.0"
          :max="1.0"
          :step="0.05"
          density="compact"
          color="teal"
          :disabled="!step3Enabled"
          thumb-label
          hint="0 = max restoration • 1 = preserve original identity"
          persistent-hint
        />
      </PipelineStepCard>

      <!-- Submit + reset -->
      <v-btn
        block
        color="deep-purple"
        :disabled="!canSubmit"
        :loading="isSubmitting || jobStatus === 'running' || jobStatus === 'pending'"
        prepend-icon="mdi-rocket-launch"
        size="large"
        @click="submit"
      >
        Run pipeline
      </v-btn>
      <v-btn
        v-if="jobId"
        block
        variant="text"
        class="mt-2"
        @click="reset"
      >
        Reset
      </v-btn>

      <!-- Job status chip -->
      <div v-if="jobStatus" class="mt-3 text-center">
        <v-chip
          :color="jobStatus === 'completed' ? 'success' : jobStatus === 'failed' ? 'error' : 'warning'"
          size="small"
          variant="tonal"
        >
          {{ jobStatus }}
        </v-chip>
      </div>
    </v-col>

    <!-- ── Right panel: result ── -->
    <v-col cols="12" md="7" lg="8">
      <!-- Result card -->
      <v-card v-if="resultArtifact && artifactUrl" elevation="2" class="pa-4">
        <div class="d-flex align-center justify-space-between mb-3">
          <span class="text-h6 font-weight-bold">Result</span>
          <div class="d-flex align-center gap-2">
            <v-chip size="small" variant="tonal" color="deep-purple">
              {{ resultArtifact.width }}×{{ resultArtifact.height }}
            </v-chip>
            <v-chip size="small" variant="tonal">
              {{ formatBytes(resultArtifact.size_bytes) }}
            </v-chip>
            <v-btn
              :href="artifactUrl"
              download
              size="small"
              variant="tonal"
              color="deep-purple"
              prepend-icon="mdi-download"
            >
              Download
            </v-btn>
          </div>
        </div>
        <v-img
          :src="artifactUrl"
          :alt="`Upscaled ${resultArtifact.width}×${resultArtifact.height}`"
          class="rounded"
          max-height="720"
          contain
        />
      </v-card>

      <!-- Empty state -->
      <div
        v-else
        class="d-flex flex-column align-center justify-center text-center"
        style="min-height: 400px"
      >
        <v-icon
          icon="mdi-image-size-select-large"
          size="64"
          color="medium-emphasis"
          class="mb-4 opacity-30"
        />
        <p class="text-h6 text-medium-emphasis">
          Configure the pipeline and upload an image to start.
        </p>
        <p
          v-if="jobStatus === 'pending' || jobStatus === 'running'"
          class="text-body-2 text-medium-emphasis mt-2"
        >
          <v-progress-circular indeterminate size="16" class="mr-1" />
          Processing job {{ jobId }}…
        </p>
      </div>
    </v-col>
  </v-row>
</template>
