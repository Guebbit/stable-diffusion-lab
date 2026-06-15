<script setup lang="ts">
/**
 * Reusable card for a single step in the upscale pipeline.
 *
 * Props:
 *   step        — step number badge (1, 2, 3)
 *   title       — step name
 *   icon        — MDI icon string
 *   optional    — when true, renders an enable/disable toggle
 *   enabled     — v-model:enabled binding (only meaningful when optional=true)
 *   color       — Vuetify color for badge and accents (default: deep-purple)
 *
 * Slots:
 *   default — the controls inside the card (model select, sliders, etc.)
 */
defineProps<{
  step: number
  title: string
  icon: string
  optional?: boolean
  color?: string
}>()

const enabled = defineModel<boolean>('enabled', { default: true })
</script>

<template>
  <v-card
    :class="['pa-4', optional && !enabled ? 'opacity-60' : '']"
    elevation="2"
    style="transition: opacity 0.2s"
  >
    <!-- Header: step badge + title + optional toggle -->
    <div class="d-flex align-center mb-3">
      <v-avatar
        :color="enabled ? (color ?? 'deep-purple') : 'medium-emphasis'"
        size="28"
        class="mr-2 text-caption font-weight-bold"
        style="flex-shrink: 0"
      >
        {{ step }}
      </v-avatar>
      <v-icon :icon="icon" :color="enabled ? (color ?? 'deep-purple') : 'medium-emphasis'" class="mr-2" size="small" />
      <span class="text-subtitle-1 font-weight-bold flex-grow-1">{{ title }}</span>
      <!-- Toggle only shown when step is optional -->
      <v-switch
        v-if="optional"
        v-model="enabled"
        density="compact"
        hide-details
        :color="color ?? 'deep-purple'"
        class="ml-2"
        style="flex-shrink: 0"
      />
    </div>

    <!-- Content slot — disabled visually when step is off -->
    <div :class="optional && !enabled ? 'pointer-events-none' : ''">
      <slot />
    </div>
  </v-card>
</template>
