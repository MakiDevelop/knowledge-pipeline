"""Tests for serve.py — HTTP handler contract without loading real models."""

import json
import importlib
import os
import sys
import time
import types
from io import BytesIO

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

flagembedding = types.ModuleType("FlagEmbedding")
flagembedding.BGEM3FlagModel = object
flagembedding.FlagReranker = object
sys.modules.setdefault("FlagEmbedding", flagembedding)

serve = importlib.import_module("serve")


class FakeModel:
    def encode(self, texts, return_dense, return_sparse, return_colbert_vecs):
        assert texts == ["hello"]
        assert return_dense is True
        assert return_sparse is True
        assert return_colbert_vecs is False
        return {"dense_vecs": [np.array([1, 0], dtype=np.float32)], "lexical_weights": [{}]}


def configure_globals():
    serve._model = FakeModel()
    serve._conn = None
    serve._rows = [
        {
            "id": 1,
            "url": "https://example.com/1",
            "domain": "example.com",
            "title": "Example",
            "core_insight": "Useful result",
            "signal_score": 90,
            "route_to": "research",
        }
    ]
    serve._matrix = np.array([[1, 0]], dtype=np.float32)
    serve._sparse = [{}]
    serve._use_rerank = False
    serve._stats = {"start_time": time.time(), "queries": 0, "avg_latency_ms": 0, "_latency_sum": 0}


def invoke_get(path):
    handler = serve.SearchHandler.__new__(serve.SearchHandler)
    handler.path = path
    handler.wfile = BytesIO()
    handler.status = None
    handler.headers = []

    def send_response(status):
        handler.status = status

    def send_header(key, value):
        handler.headers.append((key, value))

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = lambda: None

    handler.do_GET()
    return handler.status, json.loads(handler.wfile.getvalue())


def test_search_handler_contract():
    configure_globals()

    status, data = invoke_get("/search")
    assert status == 400
    assert data == {"error": "missing ?q= parameter"}

    status, data = invoke_get("/search?q=hello&k=abc")
    assert status == 400
    assert data == {"error": "k and min_score must be integers"}

    status, data = invoke_get("/health")
    assert status == 200
    assert data == {"status": "ok", "items": 1}

    status, data = invoke_get("/stats")
    assert status == 200
    assert set(data) == {"items_total", "queries_served", "avg_latency_ms", "uptime_seconds"}
    assert data["items_total"] == 1

    status, data = invoke_get("/search?q=hello")
    assert status == 200
    assert data["query"] == "hello"
    assert data["count"] == 1
    assert data["results"][0]["id"] == 1
    assert "latency_ms" in data

    status, data = invoke_get("/missing")
    assert status == 404
    assert data == {"error": "not found"}
