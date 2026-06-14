<script setup lang="ts">
/**
 * Landing page - brief presentation of the application.
 * Provides quick navigation to the main features.
 */
import { useRouter } from 'vue-router'

const router = useRouter()

const features = [
  {
    title: 'Image Generation',
    description: 'Create, edit, upscale, and transform images using AI diffusion models.',
    icon: 'mdi-image-sparkle',
    route: '/image',
    color: 'primary',
  },
  {
    title: 'Model Library',
    description: 'Browse, download, and manage AI models from HuggingFace and Civitai.',
    icon: 'mdi-brain',
    route: '/models',
    color: 'secondary',
  },
  {
    title: 'Generation History',
    description: 'Review and revisit all your past generations and their results.',
    icon: 'mdi-history',
    route: '/history',
    color: 'success',
  },
]
</script>

<template>
  <div class="home-view">
    <div class="text-center mb-12">
      <v-icon icon="mdi-atom-variant" size="x-large" color="primary" class="mb-4" />
      <h1 class="text-h3 font-weight-bold mb-2">
        Stable Diffusion Lab
      </h1>
      <p class="text-h6 text-medium-emphasis max-width-600 mx-auto">
        An AI-powered image generation workspace. Create images from text prompts,
        transform existing images, upscale, describe, recolor, and turn sketches into ink &mdash;
        all from a single unified interface.
      </p>
    </div>

    <v-row justify="center" class="mb-8">
      <v-col
        v-for="feature in features"
        :key="feature.route"
        cols="12"
        sm="4"
        class="d-flex"
      >
        <v-card
          class="cursor-pointer pa-4 text-center d-flex flex-column align-center"
          :elevation="2"
          hover
          width="100%"
          @click="router.push(feature.route)"
        >
          <v-icon
            :icon="feature.icon"
            size="x-large"
            :color="feature.color"
            class="mb-4"
          />
          <h2 class="text-h6 font-weight-bold mb-2">
            {{ feature.title }}
          </h2>
          <p class="text-body-2 text-medium-emphasis">
            {{ feature.description }}
          </p>
        </v-card>
      </v-col>
    </v-row>

    <div class="text-center text-body-2 text-medium-emphasis">
      <v-btn
        size="large"
        color="primary"
        elevation="2"
        rounded="lg"
        prepend-icon="mdi-rocket-launch"
        @click="router.push('/image')"
      >
        Start Creating
      </v-btn>
    </div>
  </div>
</template>

<style scoped>
.home-view {
  max-width: 960px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.cursor-pointer {
  cursor: pointer;
  transition: transform 0.15s ease;
}

.cursor-pointer:hover {
  transform: translateY(-4px);
}

.max-width-600 {
  max-width: 600px;
}
</style>