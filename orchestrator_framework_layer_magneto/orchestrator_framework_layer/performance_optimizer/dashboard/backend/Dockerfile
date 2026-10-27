# Container build for the Optimization Dashboard's API — its own image,
# separate from Algo Akanade's (see README_OPTIMIZATION_DASHBOARD.md's
# Architecture section for why this reads Algo Akanade's data directly off a
# shared volume instead of calling Algo Akanade's HTTP API).
#
# Build (from repo root):
#   docker build -t dockerhubalgo/algo-kaizen-optimization-dashboard-backend:latest \
#     orchestrator_framework_layer/performance_optimizer/dashboard/backend
#
# Run (needs Algo Akanade's DATA_DIR mounted read-only — same volume/path
# Algo Akanade's own container already writes to):
#   docker run -d --name ak-opt-dashboard-backend --network ak-net \
#     -p 8020:8000 \
#     -v ak-akanade-data:/srv/akanade-data:ro \
#     -e ALGO_AKANADE_DATA_DIR=/srv/akanade-data \
#     -e ALGO_ENGINE_BASE_URL=http://ak-algo-engine:8080 \
#     -e DASHBOARD_ADMIN_USER=admin -e DASHBOARD_ADMIN_PASSWORD=admin \
#     -e DASHBOARD_JWT_SECRET=<change-me> \
#     dockerhubalgo/algo-kaizen-optimization-dashboard-backend:latest
FROM python:3.12-slim

WORKDIR /srv/app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --default-timeout=180 -r requirements.txt

COPY app ./app

# Non-root, same reasoning Algo Akanade's own Dockerfile documents (a
# container compromise shouldn't get root with no extra step needed).
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
