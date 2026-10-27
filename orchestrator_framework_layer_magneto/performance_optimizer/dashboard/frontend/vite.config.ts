import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard backend (see ../backend) runs on 8020 by default in dev —
// see README_OPTIMIZATION_DASHBOARD.md for the full run instructions.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      "/api": "http://localhost:8020",
    },
  },
});
