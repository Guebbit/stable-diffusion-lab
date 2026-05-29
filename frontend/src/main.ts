/**
 * App bootstrap — creates the Vue app instance and plugs in:
 * - Vuetify (UI component library, dark theme)
 * - Pinia (state management, holds generation state)
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'
import 'vuetify/styles'
import './style.css'
import App from './App.vue'

// Vuetify setup — registers all components/directives globally
const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
  },
})

// Pinia = reactive store (like a global ref() that survives across components)
const pinia = createPinia()

// Mount the app to the #app div in index.html
createApp(App).use(vuetify).use(pinia).mount('#app')
