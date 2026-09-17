import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    {
      name: 'clean-urls',
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          const url = new URL(req.url, 'http://localhost')
          const pathname = url.pathname
          if (pathname !== '/' && !pathname.split('/').pop().includes('.')) {
            const cleanPath = pathname.endsWith('/') ? pathname + 'index.html' : pathname + '/index.html'
            req.url = cleanPath + url.search
          }
          next()
        })
      },
    },
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        benchmark: resolve(__dirname, 'rps-benchmark/index.html'),
      },
    },
  },
})
