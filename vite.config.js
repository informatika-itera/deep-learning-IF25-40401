import fs from 'fs'
import path, { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    {
      name: 'clean-urls-and-materials',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === '/rps-benchmark' || req.url === '/rps-benchmark/') req.url = '/rps-benchmark/index.html'
          if (req.url === '/rps' || req.url === '/rps/') req.url = '/rps/index.html'
          if (req.url && req.url.startsWith('/materials/')) {
            const filePath = resolve(__dirname, '.' + req.url.split('?')[0])
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              if (filePath.endsWith('.pdf')) res.setHeader('Content-Type', 'application/pdf')
              if (filePath.endsWith('.tex')) res.setHeader('Content-Type', 'text/plain')
              if (filePath.endsWith('.png')) res.setHeader('Content-Type', 'image/png')
              return fs.createReadStream(filePath).pipe(res)
            }
          }
          next()
        })
      },
      closeBundle() {
        const srcDir = resolve(__dirname, 'materials')
        const destDir = resolve(__dirname, 'dist/materials')
        if (fs.existsSync(srcDir)) {
          if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true })
          fs.cpSync(srcDir, destDir, { recursive: true })
        }
      },
    },
  ],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        rps: resolve(__dirname, 'rps/index.html'),
        benchmark: resolve(__dirname, 'rps-benchmark/index.html'),
      },
    },
  },
})
