"""
Stage 3: Retrieval evaluation
--------------------------------
Runs every question in eval_set.json through all three retrieval modes
(vector, keyword, hybrid) and measures whether the known-correct chunk
was actually retrieved.

Metrics:
  - Hit Rate@k: fraction of questions where the correct chunk appeared
    ANYWHERE in the top-k results. Simple, easy to explain.
  - MRR (Mean Reciprocal Rank): rewards ranking the correct chunk HIGHER,
    not just present. If the right chunk is result #1, that's worth more
    than if it's result #4, even though both count as a "hit" for Hit Rate.

Why this file matters more than any other in this stage:
Without this, you have three retrieval modes and a vague sense that
"hybrid seems fine." With this, you have numbers: which mode actually
performs best on YOUR codebase, for YOUR kinds of questions. That's the
difference between an opinion and a measurement - and it's very likely
the single most interview-worthy artifact in this whole project.

Usage:
    python eval.py
    python eval.py --k 3
"""

import argparse
import json
import os
import sys

import chromadb

from retrieval import build_bm25_index, vector_search, keyword_search, hybrid_search

DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"
EVAL_SET_PATH = "eval_set.json"


def chunk_matches(chunk: dict, expected_file: str, expected_symbol: str) -> bool:
    return chunk["file"] == expected_file and chunk["symbol"] == expected_symbol


def evaluate_mode(mode_name: str, retrieve_fn, eval_cases: list, k: int):
    hits = 0
    reciprocal_ranks = []
    per_question_results = []

    for case in eval_cases:
        results = retrieve_fn(case["question"], k)
        rank = None
        for i, chunk in enumerate(results):
            if chunk_matches(chunk, case["expected_file"], case["expected_symbol"]):
                rank = i + 1  # 1-indexed
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        per_question_results.append({
            "question": case["question"],
            "expected": f"{case['expected_file']}::{case['expected_symbol']}",
            "found_at_rank": rank,
        })

    n = len(eval_cases)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    return {
        "mode": mode_name,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "hits": hits,
        "total": n,
        "per_question": per_question_results,
    }


def print_summary(all_results: list):
    print("\n" + "=" * 70)
    print(f"{'Mode':<10} {'Hit Rate':<12} {'MRR':<10} {'Hits/Total'}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['mode']:<10} {r['hit_rate']:<12.1%} {r['mrr']:<10.3f} {r['hits']}/{r['total']}")
    print("=" * 70)


def print_misses(all_results: list, verbose: bool):
    if not verbose:
        return
    for r in all_results:
        misses = [q for q in r["per_question"] if q["found_at_rank"] is None]
        if misses:
            print(f"\n[{r['mode']}] questions where the correct chunk was NOT retrieved:")
            for m in misses:
                print(f"  - \"{m['question']}\" (expected {m['expected']})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against eval_set.json.")
    parser.add_argument("--k", type=int, default=4, help="Top-k to evaluate against (default: 4).")
    parser.add_argument("--verbose", action="store_true", help="List questions that missed, per mode.")
    args = parser.parse_args()

    if not os.path.isdir(DB_PATH):
        print(f"No index found at {DB_PATH}/. Run 'python ingest.py .' first.")
        sys.exit(1)

    chunks_path = os.path.join(DB_PATH, "chunks.json")
    if not os.path.isfile(chunks_path):
        print(f"No chunk sidecar found at {chunks_path}. Re-run 'python ingest.py .'.")
        sys.exit(1)

    if not os.path.isfile(EVAL_SET_PATH):
        print(f"No eval set found at {EVAL_SET_PATH}.")
        sys.exit(1)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    bm25 = build_bm25_index(chunks)

    modes = {
        "vector": lambda q, k: vector_search(collection, q, k),
        "keyword": lambda q, k: keyword_search(bm25, chunks, q, k),
        "hybrid": lambda q, k: hybrid_search(collection, bm25, chunks, q, k),
    }

    all_results = [evaluate_mode(name, fn, eval_cases, args.k) for name, fn in modes.items()]

    print(f"\nEvaluating {len(eval_cases)} questions at k={args.k}...")
    print_summary(all_results)
    print_misses(all_results, args.verbose)


if __name__ == "__main__":
    main()
