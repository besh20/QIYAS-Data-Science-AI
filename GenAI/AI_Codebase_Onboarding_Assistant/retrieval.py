"""
Stage 3: Retrieval module - vector, keyword, and hybrid search
------------------------------------------------------------------
Stage 2 only had vector (semantic) search. This adds keyword search
(BM25 - a classic, fast lexical ranking algorithm) and combines the two
using Reciprocal Rank Fusion (RRF).

Why hybrid, specifically for code?
Vector search is good at "what does X mean" but can miss exact
identifiers - if you ask about `chunk_file`, a semantic model might
retrieve a related-but-different function because it's matching on
meaning, not the literal token. BM25 is the opposite: it nails exact
term matches but has no notion of meaning. Fusing both catches more
cases than either alone - this is standard practice in production RAG,
not something most tutorial projects implement.

Why a code-aware tokenizer?
Naive tokenization would treat "chunk_file" as one opaque token, so a
query for "file chunking" wouldn't match it at all. Splitting on
snake_case/camelCase boundaries as well as whitespace means the token
"chunk_file" also indexes as "chunk" and "file" separately - a small
detail, but it's the difference between BM25 being useful for code
versus useless for it.
"""

import re

from rank_bm25 import BM25Okapi

# Splits on: whitespace/punctuation, underscores, and camelCase boundaries.
_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str):
    tokens = []
    for raw in _TOKEN_SPLIT_RE.split(text):
        if not raw:
            continue
        for piece in _CAMEL_SPLIT_RE.split(raw):
            if piece:
                tokens.append(piece.lower())
    return tokens


def build_bm25_index(chunks: list):
    """chunks: list of dicts with at least a 'code' field."""
    corpus = [tokenize(c["code"]) for c in chunks]
    return BM25Okapi(corpus)


def vector_search(collection, query: str, k: int):
    results = collection.query(query_texts=[query], n_results=k)
    ranked = []
    for doc, meta, chunk_id in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0]
    ):
        ranked.append({"id": chunk_id, "code": doc, **meta})
    return ranked


def keyword_search(bm25: BM25Okapi, chunks: list, query: str, k: int):
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in ranked_indices if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list, k: int, rrf_constant: int = 60):
    """Merge multiple ranked lists of chunk dicts into one ranking.

    Each chunk's fused score = sum over lists of 1 / (rrf_constant + rank).
    A chunk that shows up near the top of BOTH lists wins; a chunk that
    only one method found still gets credit, just less of it. This is a
    standard, dependency-free way to combine rankings from very different
    scoring scales (BM25 scores and vector distances aren't comparable
    directly - RRF sidesteps that by using rank position, not raw score).
    """
    fused_scores = {}
    chunk_by_id = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_by_id[chunk["id"]] = chunk
            fused_scores[chunk["id"]] = fused_scores.get(chunk["id"], 0.0) + 1.0 / (rrf_constant + rank)

    sorted_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in sorted_ids[:k]]


def hybrid_search(collection, bm25, chunks: list, query: str, k: int, fetch_k: int = 10):
    """Retrieve top-k chunks using both vector and keyword search, fused."""
    vec_results = vector_search(collection, query, fetch_k)
    kw_results = keyword_search(bm25, chunks, query, fetch_k)
    return reciprocal_rank_fusion([vec_results, kw_results], k)
