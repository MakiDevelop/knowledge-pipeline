#!/usr/bin/env python3
"""Layer 4: Embed — Generate dense + sparse vectors for semantic search.

Uses BAAI/bge-m3 to produce both dense (1024-dim) and learned sparse
embeddings, stored in SQLite for the search layer.

Requires: pip install FlagEmbedding

Usage:
  python3 embed.py                # Embed all unembedded items
  python3 embed.py --rebuild      # Re-embed everything (destructive!)
  python3 embed.py --remote URL   # Use a remote embedding server
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from config import EMBED_DIM, EMBED_MODEL, EMBED_REMOTE_URL, get_db_connection, init_db


def get_embedding_text(row) -> str:
    """Build the text to embed from item fields."""
    parts = []
    if row["core_insight"]:
        parts.append(row["core_insight"])
    if row["summary"]:
        parts.append(row["summary"])
    if row["full_text"]:
        parts.append(row["full_text"][:1000])
    if not parts and row["url"]:
        parts.append(row["url"])
    return " ".join(parts)


_local_model = None


def _get_local_model():
    """Lazy-load and cache the embedding model (avoid reloading on every call)."""
    global _local_model
    if _local_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _local_model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)
    return _local_model


def embed_local(texts: list[str]) -> list[dict]:
    """Embed texts using local bge-m3 model. Returns list of {dense, sparse}."""
    model = _get_local_model()
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for i in range(len(texts)):
        dense = output["dense_vecs"][i].tolist()
        sparse = {str(k): float(v) for k, v in output["lexical_weights"][i].items()}
        results.append({"dense": dense, "sparse": sparse})
    return results


def embed_remote(texts: list[str], remote_url: str) -> list[dict]:
    """Embed texts via a remote HTTP embedding server."""
    from urllib.request import Request, urlopen

    body = json.dumps({"texts": texts, "return_dense": True, "return_sparse": True}).encode()
    req = Request(remote_url, data=body, method="POST", headers={"Content-Type": "application/json"})

    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    results = []
    for i in range(len(texts)):
        dense = data["dense"][i]
        if len(dense) != EMBED_DIM:
            raise ValueError(
                f"Remote server returned {len(dense)}-dim vector, expected {EMBED_DIM}"
            )
        results.append({
            "dense": dense,
            "sparse": data.get("sparse", [{}])[i] if "sparse" in data else {},
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for scored items")
    parser.add_argument("--rebuild", action="store_true", help="Clear and re-embed all items")
    parser.add_argument("--remote", type=str, default=EMBED_REMOTE_URL, help="Remote embedding server URL")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    args = parser.parse_args()

    init_db()
    conn = get_db_connection()

    if args.rebuild:
        print("WARNING: --rebuild will clear ALL existing embeddings.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        conn.execute("UPDATE items SET embedding = NULL, sparse_weights = NULL, embedded_at = NULL")
        conn.commit()

    rows = conn.execute(
        "SELECT id, url, core_insight, summary, full_text FROM items "
        "WHERE fetch_status = 'fetched' AND embedding IS NULL "
        "ORDER BY added_at"
    ).fetchall()

    print(f"[Embed] {len(rows)} items to embed")
    if not rows:
        return

    texts = [get_embedding_text(row) for row in rows]
    ids = [row["id"] for row in rows]
    now = datetime.now(timezone.utc).isoformat()

    use_remote = bool(args.remote)
    batch_size = args.batch_size

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]

        print(f"  Embedding batch {start+1}-{end}/{len(texts)}...", end=" ", flush=True)
        t0 = time.time()

        if use_remote:
            results = embed_remote(batch_texts, args.remote)
        else:
            results = embed_local(batch_texts)

        for j, item_id in enumerate(batch_ids):
            conn.execute(
                "UPDATE items SET embedding = ?, sparse_weights = ?, embedded_at = ? WHERE id = ?",
                (
                    json.dumps(results[j]["dense"]),
                    json.dumps(results[j]["sparse"]),
                    now,
                    item_id,
                ),
            )

        conn.commit()
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

    conn.close()
    print(f"Done: {len(rows)} items embedded")


if __name__ == "__main__":
    main()
