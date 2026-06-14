<script setup lang="ts">
/**
 * Original generation page — text2img + img2img + sketch2ink workflows.
 */
import GenerationForm from '../components/GenerationForm.vue'
import ImageGallery from '../components/ImageGallery.vue'
</script>

<template>
  <v-row>
    <!-- Left: Generation Form -->
    <v-col cols="12" md="4" lg="3">
      <GenerationForm />
    </v-col>

    <!-- Right: Image Gallery -->
    <v-col cols="12" md="8" lg="9">
      <ImageGallery />
    </v-col>
  </v-row>
</template>
