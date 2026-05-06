#!/usr/bin/env python3
"""Layer 5: Search — Hybrid semantic search with reranking.

Combines dense cosine similarity (70%) and sparse lexical matching (30%)
for robust multilingual search. Optional cross-encoder reranking.

Requires: pip install FlagEmbedding numpy

Usage:
  python3 search.py "AI agent orchestration"
  python3 search.py "knowledge management" --rerank
  python3 search.py "LLM scoring" -k 5 --json
  python3 search.py "Docker" --domain github.com
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from config import DB_PATH, EMBED_DIM, EMBED_MODEL, get_db_connection, init_db

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 10
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3


def load_embeddings(conn, filters: dict | None = None):
    """Load embedding matrix and sparse weights from DB."""
    where = ["embedding IS NOT NULL", "length(embedding) > 2"]
    params = []

    if filters:
        if filters.get("domain"):
            where.append("domain = ?")
            params.append(filters["domain"])
        if filters.get("min_score"):
            where.append("signal_score >= ?")
            params.append(filters["min_score"])

    query = f"SELECT id, url, domain, title, core_insight, signal_score, route_to, embedding, sparse_weights FROM items WHERE {' AND '.join(where)}"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        return [], np.array([]), []

    matrix = np.zeros((len(rows), EMBED_DIM), dtype=np.float32)
    sparse_list = []
    for i, row in enumerate(rows):
        vec = json.loads(row["embedding"])
        matrix[i] = np.array(vec, dtype=np.float32)
        sw = json.loads(row["sparse_weights"]) if row["sparse_weights"] else {}
        sparse_list.append(sw)

    return rows, matrix, sparse_list


def hybrid_search(query_text: str, model, rows, matrix, sparse_list, top_k: int = 10, dense_only: bool = False):
    """Perform hybrid search (dense + sparse). Returns scored results."""
    q_output = model.encode(
        [query_text], return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    q_dense = q_output["dense_vecs"][0]
    q_sparse = {str(k): float(v) for k, v in q_output["lexical_weights"][0].items()}

    # Dense cosine similarity
    q_norm = q_dense / (np.linalg.norm(q_dense) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    dense_scores = (matrix / norms) @ q_norm

    if dense_only or not sparse_list:
        final_scores = dense_scores
    else:
        # Sparse dot product
        sparse_scores = np.zeros(len(rows), dtype=np.float32)
        for i, sw in enumerate(sparse_list):
            score = sum(q_sparse.get(k, 0) * v for k, v in sw.items())
            sparse_scores[i] = score
        # Normalize sparse scores
        s_max = sparse_scores.max()
        if s_max > 0:
            sparse_scores /= s_max
        final_scores = DENSE_WEIGHT * dense_scores + SPARSE_WEIGHT * sparse_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        idx = int(idx)
        if final_scores[idx] <= 0:
            continue
        row = rows[idx]
        results.append({
            "id": row["id"],
            "url": row["url"],
            "domain": row["domain"],
            "title": row["title"],
            "core_insight": row["core_insight"],
            "signal_score": row["signal_score"],
            "route_to": row["route_to"],
            "similarity": round(float(final_scores[idx]), 4),
        })

    return results


_reranker = None


def _get_reranker():
    """Lazy-load and cache the reranker model (avoid reloading on every call)."""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(RERANKER_NAME, use_fp16=True)
    return _reranker


def rerank(query: str, results: list[dict], top_k: int = 10) -> list[dict]:
    """Rerank results using cross-encoder."""
    reranker = _get_reranker()

    pairs = [(query, r.get("core_insight") or r.get("title") or r["url"]) for r in results]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    for i, s in enumerate(scores):
        results[i]["rerank_score"] = round(float(s), 4)

    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid semantic search over your knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python3 search.py "AI agent orchestration"\n'
               '  python3 search.py "LLM scoring" --rerank -k 5\n'
               '  python3 search.py "Docker" --domain github.com --json',
    )
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of results (default {DEFAULT_TOP_K})")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--min-score", type=int, default=0, help="Minimum signal_score")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--dense-only", action="store_true", help="Disable hybrid, use dense only")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output as JSON")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    init_db()
    conn = get_db_connection()

    print(f"Loading model {EMBED_MODEL}...", file=sys.stderr)
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)

    filters = {}
    if args.domain:
        filters["domain"] = args.domain
    if args.min_score:
        filters["min_score"] = args.min_score

    rows, matrix, sparse_list = load_embeddings(conn, filters)
    if not rows:
        print("No embedded items found. Run embed.py first.", file=sys.stderr)
        sys.exit(1)

    rerank_top = min(50, len(rows))
    results = hybrid_search(query_text, model, rows, matrix, sparse_list, top_k=rerank_top if args.rerank else args.top_k, dense_only=args.dense_only)

    if args.rerank:
        results = rerank(query_text, results, top_k=args.top_k)

    if args.output_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\nSearch: \"{query_text}\" ({len(results)} results)\n")
        for i, r in enumerate(results, 1):
            score_str = f"signal={r['signal_score']}" if r["signal_score"] else ""
            rerank_str = f" rerank={r['rerank_score']}" if "rerank_score" in r else ""
            print(f"  {i}. [{r['similarity']:.3f}{rerank_str}] {score_str} [{r['route_to'] or '?'}]")
            print(f"     {r['title'] or r['url']}")
            if r["core_insight"]:
                print(f"     > {r['core_insight'][:100]}")
            print()

    conn.close()


if __name__ == "__main__":
    main()
