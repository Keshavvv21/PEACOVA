"""Vector storage + retrieval optimization — two real, mathematically-
grounded techniques layered on top of rag.py's InMemoryVectorStoreAdapter,
both driven by the same "optimize once, scaled to corpus size" idea as
document_pipeline.py's retrieval-breadth scaling:

1. **Scalar quantization** (storage) — asymmetric per-vector uint8
   quantization. For a raw embedding vector v = (v_1, ..., v_d):

       v_min = min(v),  v_max = max(v)
       scale = (v_max - v_min) / 255           (1.0 if v_max == v_min)
       q_i   = round((v_i - v_min) / scale)    clamped to [0, 255], uint8
       dequantize:  v_i' = q_i * scale + v_min

   Trades a small, BOUNDED approximation error (each dequantized
   component is within scale/2 of the original) for a real 4x memory
   reduction per vector (float32/float64 -> uint8) — the same technique
   FAISS's ScalarQuantizer and most production vector DBs use as their
   cheapest compression tier.

2. **IVF (Inverted File Index) clustering** (retrieval speed) — brute-
   force search is O(n·d): every query compares against every stored
   vector. IVF instead partitions the n stored vectors into
   K = round(sqrt(n)) clusters via Lloyd's k-means (K = sqrt(n) is the
   standard heuristic: it makes both the number of clusters AND the
   average cluster size ~sqrt(n), balancing the two costs below), then
   a query only computes exact distance against the n_probe nearest
   clusters' members instead of all n vectors:

       search cost  ≈ O(K·d)              [scan every centroid]
                    + O((n/K)·n_probe·d)  [scan members of the probed clusters]

   Minimized at K = sqrt(n), giving O(sqrt(n)·d) total — the standard
   first-level technique real ANN systems (FAISS's IndexIVFFlat, etc.)
   use before adding further compression on top. n_probe=1 (search only
   the single best-matching cluster) trades a little recall for the
   full sqrt(n) speedup; a larger n_probe recovers more recall at
   proportionally more cost.

Lloyd's k-means, the clustering step IVF needs:

    assignment:  cluster(x_i) = argmin_k ||x_i - μ_k||²
    update:      μ_k = (1/|C_k|) Σ_{x_i ∈ C_k} x_i

repeated until assignments stop changing or max_iters is hit.

Both techniques are opt-in and used together by
InMemoryVectorStoreAdapter only once a corpus crosses
IVF_MIN_VECTORS (see rag.py) — a handful of vectors gets zero benefit
from clustering overhead and stays on plain brute-force scan, mirroring
document_pipeline.py's own "small corpus = simple defaults" philosophy.
"""
from __future__ import annotations

import math

import numpy as np


def quantize_vector(v: list[float]) -> tuple[bytes, float, float]:
    """Returns (uint8 bytes, v_min, scale) — dequantize_vector() needs
    all three. v_min/scale are per-vector (asymmetric quantization),
    not a single global scale, since different embeddings can have
    very different value ranges."""
    arr = np.asarray(v, dtype=np.float64)
    v_min, v_max = float(arr.min()), float(arr.max())
    scale = (v_max - v_min) / 255.0 if v_max > v_min else 1.0
    q = np.clip(np.round((arr - v_min) / scale), 0, 255).astype(np.uint8)
    return q.tobytes(), v_min, scale


def dequantize_vector(data: bytes, v_min: float, scale: float) -> list[float]:
    q = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    return (q * scale + v_min).tolist()


def recommended_cluster_count(n_vectors: int) -> int:
    """K = round(sqrt(n)) — see this module's docstring for why sqrt(n)
    minimizes total IVF search cost. Never more clusters than vectors."""
    return max(1, min(n_vectors, round(math.sqrt(n_vectors))))


def kmeans(vectors: np.ndarray, k: int, *, max_iters: int = 25,
          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm — see module docstring for the assignment/update
    formulas. Returns (centroids [k x d], assignments [n], one cluster
    id per input vector). Deterministic given `seed` (no wall-clock/
    system randomness) so results are reproducible."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    k = min(k, n)
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[centroid_idx].copy()
    assignments = np.full(n, -1, dtype=int)
    for iteration in range(max_iters):
        dists = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = dists.argmin(axis=1)
        if iteration > 0 and np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroids[c] = vectors[mask].mean(axis=0)
    return centroids, assignments


class IVFIndex:
    """Built once over a fixed set of vectors, queried many times.
    search() only scans the n_probe nearest clusters' members rather
    than every stored vector — see module docstring for the real
    complexity analysis. Assumes unit-normalized input vectors (true
    for everything HashEmbedder produces), so dot product IS cosine
    similarity — no separate normalization step needed here."""

    def __init__(self, vectors: list[list[float]], *, n_clusters: int | None = None, seed: int = 0):
        if not vectors:
            raise ValueError("IVFIndex needs at least one vector")
        self._vectors = np.asarray(vectors, dtype=np.float64)
        self.n_clusters = n_clusters or recommended_cluster_count(len(vectors))
        self.centroids, self.assignments = kmeans(self._vectors, self.n_clusters, seed=seed)
        self._cluster_members = {
            c: np.where(self.assignments == c)[0] for c in range(len(self.centroids))
        }

    def search(self, query: list[float], k: int, *, n_probe: int = 1) -> list[tuple[int, float]]:
        """Returns [(original_index, cosine_score), ...], best first,
        up to k results. n_probe=len(self.centroids) degenerates to an
        exhaustive (exact, brute-force-equivalent) search — useful as a
        correctness check against InMemoryVectorStoreAdapter's own
        plain scan."""
        q = np.asarray(query, dtype=np.float64)
        n_probe = min(n_probe, len(self.centroids))
        centroid_scores = self.centroids @ q
        probe_clusters = np.argsort(-centroid_scores)[:n_probe]
        member_arrays = [self._cluster_members[c] for c in probe_clusters
                        if len(self._cluster_members[c])]
        if not member_arrays:
            return []
        candidate_idx = np.concatenate(member_arrays)
        scores = self._vectors[candidate_idx] @ q
        order = np.argsort(-scores)[:k]
        return [(int(candidate_idx[i]), float(scores[i])) for i in order]
