"""Tests for embed.py — text selection and embedding adapters."""

import json
import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import embed


def test_get_embedding_text_uses_priority_content_and_truncates_full_text():
    row = {
        "core_insight": "core",
        "summary": "summary",
        "full_text": "x" * 1200,
        "url": "https://example.com",
    }

    text = embed.get_embedding_text(row)

    assert text == f"core summary {'x' * 1000}"


def test_get_embedding_text_falls_back_to_url_when_content_missing():
    row = {
        "core_insight": None,
        "summary": "",
        "full_text": None,
        "url": "https://example.com/fallback",
    }

    assert embed.get_embedding_text(row) == "https://example.com/fallback"


def test_embed_local_converts_dense_and_sparse(monkeypatch):
    class FakeModel:
        def encode(self, texts, return_dense, return_sparse, return_colbert_vecs):
            assert texts == ["one", "two"]
            assert return_dense is True
            assert return_sparse is True
            assert return_colbert_vecs is False
            return {
                "dense_vecs": [
                    np.array([1.0, 2.0], dtype=np.float32),
                    np.array([3.0, 4.0], dtype=np.float32),
                ],
                "lexical_weights": [{1: 0.5}, {"token": 2}],
            }

    monkeypatch.setattr(embed, "_get_local_model", lambda: FakeModel())

    assert embed.embed_local(["one", "two"]) == [
        {"dense": [1.0, 2.0], "sparse": {"1": 0.5}},
        {"dense": [3.0, 4.0], "sparse": {"token": 2.0}},
    ]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_embed_remote_parses_dense_and_sparse(monkeypatch):
    monkeypatch.setattr(embed, "EMBED_DIM", 3)

    def fake_urlopen(req, timeout):
        assert timeout == 120
        assert json.loads(req.data) == {
            "texts": ["hello"],
            "return_dense": True,
            "return_sparse": True,
        }
        return FakeResponse({"dense": [[1, 2, 3]], "sparse": [{"kw": 0.5}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert embed.embed_remote(["hello"], "http://embed.test") == [
        {"dense": [1, 2, 3], "sparse": {"kw": 0.5}}
    ]


def test_embed_remote_rejects_wrong_dense_dimension(monkeypatch):
    monkeypatch.setattr(embed, "EMBED_DIM", 3)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout: FakeResponse({"dense": [[1, 2]]}),
    )

    try:
        embed.embed_remote(["hello"], "http://embed.test")
    except ValueError as e:
        assert "2-dim vector, expected 3" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_embed_remote_defaults_missing_sparse_to_empty_dict(monkeypatch):
    monkeypatch.setattr(embed, "EMBED_DIM", 3)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout: FakeResponse({"dense": [[1, 2, 3]]}),
    )

    assert embed.embed_remote(["hello"], "http://embed.test") == [
        {"dense": [1, 2, 3], "sparse": {}}
    ]
