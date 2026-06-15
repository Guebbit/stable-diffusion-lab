import { createRouter, createWebHistory } from 'vue-router'

/** Route definitions — each maps a URL path to a view component. */
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../views/HomeView.vue'),
    },
    {
      path: '/image',
      name: 'image',
      component: () => import('../views/ImageView.vue'),
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('../views/ModelsView.vue'),
    },
    {
      path: '/image-upscale',
      name: 'image-upscale',
      component: () => import('../views/ImageUpscaleView.vue'),
    },
    {
      path: '/history',
      name: 'history',
      component: () => import('../views/HistoryView.vue'),
    },
  ],
})

export default router
