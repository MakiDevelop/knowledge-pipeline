"""Tests for search.py — embedding loading, hybrid search, and reranking."""

import json
import importlib
import os
import sqlite3
import sys
import types

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

flagembedding = types.ModuleType("FlagEmbedding")
flagembedding.BGEM3FlagModel = object
flagembedding.FlagReranker = object
sys.modules.setdefault("FlagEmbedding", flagembedding)

search = importlib.import_module("search")


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.sql")) as f:
        conn.executescript(f.read())
    return conn


def insert_item(conn, url, domain, score, dense, sparse=None):
    conn.execute(
        """
        INSERT INTO items (
            url, domain, title, core_insight, signal_score, route_to,
            embedding, sparse_weights
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            url,
            domain,
            f"title {url}",
            f"insight {url}",
            score,
            "research",
            json.dumps(dense),
            json.dumps(sparse or {}),
        ),
    )
    conn.commit()


class FakeModel:
    def __init__(self, dense, sparse=None):
        self.dense = np.array(dense, dtype=np.float32)
        self.sparse = sparse or {}

    def encode(self, texts, return_dense, return_sparse, return_colbert_vecs):
        assert texts
        assert return_dense is True
        assert return_sparse is True
        assert return_colbert_vecs is False
        return {"dense_vecs": [self.dense], "lexical_weights": [self.sparse]}


def rows_for_search():
    return [
        {
            "id": 1,
            "url": "https://a.example/1",
            "domain": "a.example",
            "title": "A",
            "core_insight": "A insight",
            "signal_score": 80,
            "route_to": "research",
        },
        {
            "id": 2,
            "url": "https://b.example/2",
            "domain": "b.example",
            "title": "B",
            "core_insight": "B insight",
            "signal_score": 70,
            "route_to": "writer",
        },
    ]


def test_load_embeddings_filters_and_empty(monkeypatch):
    monkeypatch.setattr(search, "EMBED_DIM", 3)
    conn = make_conn()
    insert_item(conn, "https://a.example/1", "a.example", 80, [1, 0, 0], {"42": 1.5})
    insert_item(conn, "https://b.example/1", "b.example", 40, [0, 1, 0], {"7": 2.0})
    insert_item(conn, "https://a.example/2", "a.example", 20, [0, 0, 1], {"42": 0.5})

    rows, matrix, sparse = search.load_embeddings(conn, {"domain": "a.example", "min_score": 50})

    assert [row["url"] for row in rows] == ["https://a.example/1"]
    assert matrix.tolist() == [[1.0, 0.0, 0.0]]
    assert sparse == [{"42": 1.5}]

    rows, matrix, sparse = search.load_embeddings(conn, {"domain": "missing.example"})
    assert rows == []
    assert matrix.tolist() == []
    assert sparse == []


def test_hybrid_search_dense_cosine_sorting_and_fields():
    rows = rows_for_search()
    matrix = np.array([[1, 0], [0.8, 0.6]], dtype=np.float32)
    sparse = [{"kw": 0}, {"kw": 10}]
    model = FakeModel([1, 0], {"kw": 1})

    results = search.hybrid_search("query", model, rows, matrix, sparse, top_k=2, dense_only=True)

    assert [result["id"] for result in results] == [1, 2]
    assert set(results[0]) == {
        "id",
        "url",
        "domain",
        "title",
        "core_insight",
        "signal_score",
        "route_to",
        "similarity",
    }


def test_hybrid_search_combines_dense_and_sparse_weights():
    rows = rows_for_search()
    matrix = np.array([[1, 0], [0.8, 0.6]], dtype=np.float32)
    sparse = [{"kw": 0}, {"kw": 10}]
    model = FakeModel([1, 0], {"kw": 1})

    results = search.hybrid_search("query", model, rows, matrix, sparse, top_k=2)

    assert [result["id"] for result in results] == [2, 1]
    assert results[0]["similarity"] == 0.86
    assert results[1]["similarity"] == 0.7


def test_hybrid_search_top_k_and_non_positive_filtering():
    rows = rows_for_search()
    matrix = np.array([[1, 0], [0, 1]], dtype=np.float32)
    sparse = [{}, {}]
    model = FakeModel([1, 0])

    results = search.hybrid_search("query", model, rows, matrix, sparse, top_k=1)

    assert [result["id"] for result in results] == [1]

    results = search.hybrid_search("query", model, rows, matrix, sparse, top_k=2)
    assert [result["id"] for result in results] == [1]


def test_rerank_sorts_adds_scores_and_truncates(monkeypatch):
    class FakeReranker:
        def compute_score(self, pairs, normalize):
            assert pairs == [("query", "A insight"), ("query", "B insight")]
            assert normalize is True
            return [0.1, 0.9]

    monkeypatch.setattr(search, "_get_reranker", lambda: FakeReranker())
    results = rows_for_search()

    reranked = search.rerank("query", results, top_k=1)

    assert [result["id"] for result in reranked] == [2]
    assert reranked[0]["rerank_score"] == 0.9
