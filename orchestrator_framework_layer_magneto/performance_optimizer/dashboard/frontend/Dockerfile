# Container build for the Optimization Dashboard's static UI, served by
# nginx (same shape as tool_layer/frontend/Dockerfile). The app calls
# "/api/..." as relative paths (see src/api.ts) — no VITE_*_BASE_URL
# needed at build time, since nginx.conf reverse-proxies /api/ to the
# dashboard-backend container on the same origin.
#
# Build (from repo root):
#   docker build -t dockerhubalgo/algo-kaizen-optimization-dashboard-frontend:latest \
#     orchestrator_framework_layer/performance_optimizer/dashboard/frontend
#
# Run (needs --link (or a shared network) to a container literally named
# "dashboard-backend" — nginx.conf's proxy_pass target):
#   docker run -d --name ak-opt-dashboard-frontend --network ak-net \
#     -p 5175:80 --link ak-opt-dashboard-backend:dashboard-backend \
#     dockerhubalgo/algo-kaizen-optimization-dashboard-frontend:latest
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
