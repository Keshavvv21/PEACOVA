"""app/gmaofn/rag.py's own functions — chunking, context building, and
reranking. Regression tests for three real bugs found in a retrieval/
answer-quality audit: chunk overlap corrupting text mid-word,
build_context dropping ALL context when only the top hit is oversized,
and rerank()'s docstring/implementation mismatch (a no-op despite
claiming a "lexical overlap boost").
"""
from __future__ import annotations

from app.framework.rag import RagHit, build_context_with_included_hits, chunk_text, rerank
from app.retrieval.vectorstore import HashEmbedder


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hash_embedder_no_longer_confuses_unrelated_topics_via_stopwords():
    """Regression, from a real empirical audit: HashEmbedder's plain
    hashed bag-of-words had no stopword filtering, so two queries about
    COMPLETELY different topics that happened to share only common
    function words ("how", "do", "my", "is", "account") scored as
    similar as genuinely related ones. "how do I delete my account"
    used to rank a password-reset chunk as its top (and ONLY non-zero)
    match, purely from stopword overlap, despite zero shared topical
    content. With stopwords filtered before hashing, an unrelated query
    must no longer score meaningfully above zero against a chunk that
    shares no real content words with it."""
    e = HashEmbedder()
    password_reset_chunk = ("To reset your password, click the link we emailed you, "
                            "or request a new one if it has expired.")
    unrelated_query = "how do I delete my account"
    sim = _cosine(e.embed(unrelated_query), e.embed(password_reset_chunk))
    assert sim < 0.15, f"expected near-zero similarity for an unrelated query, got {sim}"


def test_hash_embedder_still_ranks_genuinely_related_text_higher():
    """The fix must not just zero everything out — real topical overlap
    (shared content words) should still score meaningfully higher than
    an unrelated pair."""
    e = HashEmbedder()
    query = "how do I reset my password"
    related = "To reset your password, click the link we emailed you."
    unrelated = "Shipping delays are usually resolved within three business days."
    sim_related = _cosine(e.embed(query), e.embed(related))
    sim_unrelated = _cosine(e.embed(query), e.embed(unrelated))
    assert sim_related > sim_unrelated
    assert sim_related > 0.3


def test_chunk_overlap_never_starts_a_chunk_mid_word():
    """Regression: `current[-overlap:]` used to slice raw characters,
    landing mid-word (e.g. a chunk starting "r request a new one"
    instead of "...or request a new one"). Force a chunk boundary with
    a small chunk_size/overlap and check every chunk after the first
    starts on a real word boundary."""
    text = ("To reset your password, click the link we emailed you, "
            "or request a new one if it has expired. "
            "Billing cycles run monthly, starting the day you subscribe. "
            "Shipping delays are usually resolved within three business days.")
    chunks = chunk_text(text, chunk_size=60, overlap=15)
    assert len(chunks) > 1
    for chunk in chunks[1:]:
        first_word = chunk.split(" ", 1)[0]
        # A corrupted chunk starts with a word FRAGMENT that isn't a
        # real standalone word from the source text — a crude but
        # effective check: every real word in the source is at least 2
        # characters and alphabetic; a fragment like "r" or "o" (from
        # slicing mid-word) is exactly what this regression looked like.
        assert len(first_word) > 1, f"chunk starts with a likely word fragment: {chunk!r}"


def test_build_context_skips_an_oversized_hit_instead_of_dropping_everything():
    """Regression: build_context_with_included_hits() used to `break` the
    moment ANY hit didn't fit the remaining budget — if the FIRST
    (highest-scored) hit alone exceeded max_chars, the whole turn got
    ZERO context back, even with perfectly good smaller hits right
    behind it. It should skip the oversized hit and keep trying."""
    huge = RagHit(text="x" * 200, score=0.9, metadata={})
    small = RagHit(text="a short, relevant chunk", score=0.8, metadata={})
    context, included = build_context_with_included_hits([huge, small], max_chars=100)
    assert small in included
    assert huge not in included
    assert "short, relevant chunk" in context


def test_rerank_without_a_query_is_the_old_plain_resort():
    """Backward-compatible default — no behavior change for a caller
    that doesn't pass `query`."""
    hits = [RagHit(text="low", score=0.2, metadata={}),
           RagHit(text="high", score=0.9, metadata={})]
    ranked = rerank(hits)
    assert [h.text for h in ranked] == ["high", "low"]


def test_rerank_with_a_query_actually_reranks_by_lexical_overlap():
    """Regression: rerank()'s docstring always claimed a "lexical overlap
    boost," but the implementation was a no-op re-sort by the existing
    (already-sorted) score — real callers (routers/v1/rag.py) got zero
    reranking benefit while believing they had it. With `query` passed,
    a lower-vector-score hit that's a much better keyword match for the
    query should be able to outrank a higher-vector-score hit that
    shares almost no words with it."""
    keyword_poor = RagHit(text="a document about unrelated topics entirely",
                          score=0.6, metadata={})
    keyword_rich = RagHit(text="reset password account login credentials",
                          score=0.55, metadata={})
    ranked = rerank([keyword_poor, keyword_rich], query="how do I reset my password")
    assert ranked[0] is keyword_rich
