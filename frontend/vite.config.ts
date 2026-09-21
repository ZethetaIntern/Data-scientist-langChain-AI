import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend only ever talks to relative `/api/...` URLs. In development Vite proxies
// them to the FastAPI server; in production FastAPI serves the built bundle itself, so
// the browser never needs to know where the backend lives.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: [".e2b.app", "localhost", "127.0.0.1"],
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
