import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env files from project root (parent of frontend/)
  const env = loadEnv(mode, path.resolve(__dirname, '..'))

  const backendPort = parseInt(env.BACKEND_PORT || '8000', 10)
  const frontendPort = parseInt(env.FRONTEND_PORT || '5173', 10)

  // Detect if running inside Docker (where backend service is reachable by name)
  const isDocker = env.IS_DOCKER === 'true' || process.env.IS_DOCKER === 'true'
  const backendHost = isDocker ? 'backend' : 'localhost'

  return {
    plugins: [
      vue(),
      vuetify({ autoImport: true }),
    ],
    server: {
      host: true,
      port: frontendPort,
      proxy: {
        '/api': {
          target: `http://${backendHost}:${backendPort}`,
          changeOrigin: true,
          secure: false,
        },
        '/sse': {
          target: `http://${backendHost}:${backendPort}`,
          changeOrigin: true,
          secure: false,
        },
      },
    },
  }
})