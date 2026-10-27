FROM python:3.12-slim

WORKDIR /srv/app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
# --default-timeout: torch/google-adk pull large wheels; the default 15s
# pip read-timeout intermittently trips on those over a slow/rate-limited
# connection mid-download, failing the whole build.
RUN pip install --default-timeout=180 -r requirements.txt

COPY app ./app

# Just this ONE file, not the whole scripts/ directory — the rest of
# scripts/ (finetune_local_slm.py, eval/) needs the heavy ML deps
# (torch/transformers/peft) that live ONLY in the separate
# algo-kaizen-finetune image (deploy/Dockerfile.finetune) and are
# deliberately absent from this lightweight API-server image's own
# requirements.txt. This one script only imports this repo's own
# app.agent.finetune_log/app.llm.router/app.framework.agents.coding_agent
# — real, already-installed dependencies here — so it runs fine via
# `docker exec ak-chatbot python -m scripts.log_complex_ground_truth`
# (see README_COMPLEX_EXAMPLE_TRAINING.md) without needing this image to
# grow the training toolchain it will never otherwise use. No
# scripts/__init__.py exists anywhere in this repo (PEP 420 implicit
# namespace packages — the SAME convention scripts.eval.regression_gate/
# scripts.finetune_local_slm already rely on in algo-kaizen-finetune),
# so this needs no package-init file either.
COPY scripts/log_complex_ground_truth.py ./scripts/log_complex_ground_truth.py
# Same "one file, not the whole scripts/ dir" reasoning as above — only
# imports app.framework.agents.coding_agent/app.llm.intent_reasoner/
# app.llm.router, all already installed here.
COPY scripts/run_slm_test_suite.py ./scripts/run_slm_test_suite.py

# data + chroma live on volumes (see docker-compose.yml)
ENV DATA_DIR=/srv/data CHROMA_DIR=/srv/chroma_db

# Run as a non-root user — flagged by cloud-config-agent (CFG-NO-USER-DIRECTIVE):
# without this the process runs as root by default, so a container
# compromise gets root inside the container with no privilege-escalation
# step needed. The volume dirs are created (and chowned) here, before
# Docker mounts ak-chatbot-data/ak-chatbot-chroma over them, so a fresh
# named volume inherits appuser ownership instead of root's.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /srv/data /srv/chroma_db \
    && chown -R appuser:appuser /srv/app /srv/data /srv/chroma_db
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
