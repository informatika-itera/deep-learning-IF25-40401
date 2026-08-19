import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    {
      name: 'clean-rps-benchmark-url',
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          if (req.url === '/rps-benchmark') req.url = '/rps-benchmark/index.html'
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
