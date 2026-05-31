import { createRouter, createWebHistory } from 'vue-router'

/** Route definitions — each maps a URL path to a view component. */
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'generate',
      component: () => import('../views/GenerateView.vue'),
    },
    {
      path: '/describe',
      name: 'describe',
      component: () => import('../views/DescribeView.vue'),
    },
    {
      path: '/edit',
      name: 'edit',
      component: () => import('../views/EditImageView.vue'),
    },
    {
      path: '/ink',
      name: 'ink',
      component: () => import('../views/InkSketchView.vue'),
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('../views/ModelsView.vue'),
    },
  ],
})

export default router
