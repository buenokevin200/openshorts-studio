import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import seo from './vite-plugin-seo'

// Backend target for the dev proxy. Defaults to local 127.0.0.1:8000;
// set VITE_PROXY_TARGET=http://backend:8000 inside docker-compose.
const backend = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000'
const renderer = process.env.VITE_RENDER_TARGET || 'http://127.0.0.1:3100'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), seo()],
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        'tool-transcript': 'tools/transcript.js',
        'tool-metadata': 'tools/metadata.js',
        'tool-vertical': 'tools/vertical.js',
        'tool-paste-link': 'tools/paste-link.js',
      },
    },
  },
  server: {
    allowedHosts: true,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/videos': { target: backend, changeOrigin: true },
      '/thumbnails': { target: backend, changeOrigin: true },
      '/gallery': { target: backend, changeOrigin: true },
      '/video': { target: backend, changeOrigin: true },
      '/render': { target: renderer, changeOrigin: true },
    }
  }
})
