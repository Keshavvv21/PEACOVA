"""app/gmaofn/vector_storage_optimizer.py — scalar quantization (storage)
and IVF clustering (retrieval speed), plus their wiring into
rag.InMemoryVectorStoreAdapter for corpora above IVF_MIN_VECTORS.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.framework.rag import IVF_MIN_VECTORS, InMemoryVectorStoreAdapter, RagDocument
from app.framework.vector_storage_optimizer import (IVFIndex, dequantize_vector, kmeans,
                                                 quantize_vector, recommended_cluster_count)


# ── scalar quantization ──────────────────────────────────────────────────

def test_quantize_dequantize_round_trip_stays_within_the_scale_bound():
    v = [0.1, -0.5, 0.9, 0.0, 0.33, -0.87, 1.0, -1.0]
    data, v_min, scale = quantize_vector(v)
    restored = dequantize_vector(data, v_min, scale)
    max_error = max(abs(a - b) for a, b in zip(v, restored))
    assert max_error <= scale / 2 + 1e-9


def test_quantize_uses_4_bytes_per_component_reduction():
    v = [0.0] * 100  # arbitrary length, just checking the storage format
    data, _, _ = quantize_vector(v)
    assert len(data) == len(v)  # 1 byte/component (uint8) vs 8 (float64) or 4 (float32)


def test_quantize_handles_a_constant_vector_without_dividing_by_zero():
    v = [0.5, 0.5, 0.5]
    data, v_min, scale = quantize_vector(v)
    restored = dequantize_vector(data, v_min, scale)
    assert all(abs(x - 0.5) < 1e-9 for x in restored)


# ── k-means / IVF ─────────────────────────────────────────────────────────

def test_recommended_cluster_count_is_sqrt_n_and_never_exceeds_n():
    assert recommended_cluster_count(100) == 10
    assert recommended_cluster_count(4) == 2
    assert recommended_cluster_count(1) == 1
    assert recommended_cluster_count(0) == 1


def _unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, d))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_kmeans_produces_the_requested_number_of_clusters_with_all_points_assigned():
    vectors = _unit_vectors(80, 8)
    centroids, assignments = kmeans(vectors, k=5)
    assert centroids.shape == (5, 8)
    assert len(assignments) == 80
    assert set(assignments.tolist()) <= {0, 1, 2, 3, 4}


def test_kmeans_is_deterministic_given_the_same_seed():
    vectors = _unit_vectors(50, 6)
    c1, a1 = kmeans(vectors, k=4, seed=7)
    c2, a2 = kmeans(vectors, k=4, seed=7)
    assert np.array_equal(a1, a2)
    assert np.allclose(c1, c2)


def test_ivf_exhaustive_probe_exactly_matches_brute_force_top_k():
    vectors = _unit_vectors(200, 16)
    vectors_list = vectors.tolist()
    index = IVFIndex(vectors_list)

    query = vectors_list[42]
    brute_scores = vectors @ np.asarray(query)
    brute_top5 = set(np.argsort(-brute_scores)[:5].tolist())

    exhaustive = index.search(query, k=5, n_probe=index.n_clusters)
    assert {i for i, _ in exhaustive} == brute_top5


def test_ivf_fast_probe_still_finds_an_exact_self_match():
    vectors = _unit_vectors(200, 16)
    vectors_list = vectors.tolist()
    index = IVFIndex(vectors_list)
    query = vectors_list[10]
    fast = index.search(query, k=3, n_probe=1)
    assert fast[0][0] == 10
    assert fast[0][1] == pytest.approx(1.0, abs=1e-9)


def test_ivf_index_rejects_an_empty_vector_list():
    with pytest.raises(ValueError):
        IVFIndex([])


# ── wiring into InMemoryVectorStoreAdapter ───────────────────────────────

def test_small_corpus_never_builds_an_ivf_index():
    store = InMemoryVectorStoreAdapter(name="small-test")
    store.add_documents([RagDocument(text=f"topic {i} document", metadata={"i": i})
                        for i in range(IVF_MIN_VECTORS - 1)])
    store.search("topic 3", k=3)
    assert store._ivf_index is None


def test_large_corpus_builds_and_uses_an_ivf_index():
    store = InMemoryVectorStoreAdapter(name="large-test")
    store.add_documents([RagDocument(text=f"topic {i} document about various subjects",
                                     metadata={"i": i}) for i in range(IVF_MIN_VECTORS + 10)])
    hits = store.search("topic 5 document", k=3)
    assert store._ivf_index is not None
    assert len(hits) == 3


def test_adding_more_documents_marks_the_ivf_index_stale_for_a_rebuild():
    store = InMemoryVectorStoreAdapter(name="rebuild-test")
    store.add_documents([RagDocument(text=f"topic {i} document about various subjects",
                                     metadata={"i": i}) for i in range(IVF_MIN_VECTORS + 5)])
    store.search("topic 1 document", k=3)
    assert store._ivf_stale is False

    store.add_documents([RagDocument(text="a brand new document", metadata={})])
    assert store._ivf_stale is True  # invalidated, not yet rebuilt

    store.search("brand new", k=3)
    assert store._ivf_stale is False  # rebuilt on the next search
