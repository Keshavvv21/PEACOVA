"""The GPU/kernel panel's data source — a best-effort live read of the
Algo Engine's own GET /stats (now including kernel_tier, see
internal/api/api.go + internal/quant/quant.go's Kernel.Name()), plus a
static description of the three kernel tiers that's true regardless of
whether an engine happens to be running right now. "Deep, CUDA/C++
level" here means: which tier is ACTUALLY active (not just configured),
and what real acceleration mechanism each tier is, in enough detail to
be useful to someone deciding whether to build the cuda tag on a real
GPU box — not a generic "GPU acceleration: on/off" toggle.
"""
from __future__ import annotations

import httpx

from .config import settings

KERNEL_TIERS = [
    {
        "id": "pure-go",
        "build": "go build (no tags)",
        "requires": "nothing — CGO_ENABLED=0, a static binary",
        "mechanism": "A plain Go double loop per MatVec/MatVecQ8 call. The "
                    "correctness oracle every other tier is equivalence-"
                    "tested against (internal/quant/equiv_test.go) — always "
                    "correct, never the fast path.",
    },
    {
        "id": "cpu-accel",
        "build": "make accel  (-tags asmkernel)",
        "requires": "a C compiler (gcc/clang) — no GPU",
        "mechanism": "A cgo C loop compiled -O3 -march=native, autovectorized "
                    "by the compiler into AVX2 (x86-64) or NEON (arm64) SIMD "
                    "instructions — real vector-width parallelism per "
                    "matvec row, not a bigger constant factor.",
    },
    {
        "id": "cuda",
        "build": "make cuda  (-tags cuda)",
        "requires": "an NVIDIA GPU + the CUDA toolkit (nvcc)",
        "mechanism": "internal/quant/cuda/kernels.cu, cgo-linked via "
                    "libaekernels.a. Weights are uploaded to device memory "
                    "ONCE per tensor and cached by host pointer (residency, "
                    "not per-call PCIe shuttling) — ae_cuda_upload/"
                    "ae_cuda_matvec_f32/ae_cuda_matvec_q8, one thread block "
                    "row-parallelizing the matvec (thread i owns output row "
                    "i). The naive-but-correct first rung — fused gate/up + "
                    "silu_mul kernels are the documented next step on this "
                    "same seam.",
    },
]


async def fetch_engine_stats() -> dict:
    """Never raises — an unreachable/unconfigured engine is a normal,
    expected state for this panel (same fail-soft convention as every
    other optional integration in this codebase), not a dashboard error."""
    base_url = settings().ALGO_ENGINE_BASE_URL
    if not base_url:
        return {"configured": False, "reachable": False, "stats": None}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url.rstrip('/')}/stats")
            r.raise_for_status()
            return {"configured": True, "reachable": True, "stats": r.json()}
    except Exception as e:  # noqa: BLE001 — genuinely any failure mode here
        return {"configured": True, "reachable": False, "error": str(e)[:200], "stats": None}
