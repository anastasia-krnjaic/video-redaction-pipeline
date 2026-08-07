import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

function stripExternalDiagnosticUrls(): Plugin {
  const reactErrorDecoderUrl = 'https://reactjs.org/docs/error-decoder.html?invariant=';

  return {
    name: 'strip-external-diagnostic-urls',
    renderChunk(code) {
      if (!code.includes(reactErrorDecoderUrl)) {
        return null;
      }
      return {
        code: code.replaceAll(reactErrorDecoderUrl, 'ReactErrorInvariant#'),
        map: null,
      };
    },
  };
}

export default defineConfig({
  base: '/app/',
  plugins: [react(), stripExternalDiagnosticUrls()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
});
