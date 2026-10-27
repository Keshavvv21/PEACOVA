FROM python:3.12-slim

WORKDIR /srv/app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# This service is CPU-only (see generic_architecture.py's own docstring on
# why "causal_llm" is excluded — no GPU here at all) — PyPI's default torch
# wheel bundles a full CUDA toolkit (14 nvidia-* packages, ~6GB uncompressed)
# that this deployment can never use. Installing torch from PyTorch's own
# CPU-only index FIRST means routellm's later `pip install -r
# requirements.txt` sees a torch already satisfying its version constraint
# and never pulls the CUDA-bundled one — same functionality (bert/mf both
# already run on CPU with zero code changes), a fraction of the size.
RUN pip install --default-timeout=180 torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --default-timeout=180 -r requirements.txt

COPY app ./app

ENV DATA_DIR=/srv/data

RUN useradd -m -u 1000 appuser \
    && mkdir -p /srv/data \
    && chown -R appuser:appuser /srv/app /srv/data
USER appuser

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
