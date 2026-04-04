"""Tests for ingest.py — URL normalization, extraction, and ingestion."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingest import extract_urls, ingest_urls, normalize_url


class TestNormalizeUrl:
    def test_strips_utm_params(self):
        url = "https://example.com/article?utm_source=twitter&utm_medium=social&id=42"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=42" in result

    def test_strips_fbclid(self):
        url = "https://example.com/page?fbclid=abc123&real=yes"
        result = normalize_url(url)
        assert "fbclid" not in result
        assert "real=yes" in result

    def test_strips_fragment(self):
        url = "https://example.com/page#section-2"
        result = normalize_url(url)
        assert "#" not in result

    def test_strips_trailing_slash(self):
        url = "https://example.com/page/"
        result = normalize_url(url)
        assert not result.endswith("/")

    def test_preserves_meaningful_params(self):
        url = "https://example.com/search?q=knowledge+pipeline&page=2"
        result = normalize_url(url)
        assert "q=" in result
        assert "page=2" in result

    def test_handles_no_params(self):
        url = "https://example.com/article"
        result = normalize_url(url)
        assert result == "https://example.com/article"


class TestExtractUrls:
    def test_extracts_from_text(self):
        text = "Check out https://example.com and also https://test.org/page"
        urls = extract_urls(text)
        assert len(urls) == 2

    def test_deduplicates(self):
        text = "Visit https://example.com and https://example.com again"
        urls = extract_urls(text)
        assert len(urls) == 1

    def test_normalizes_during_extraction(self):
        text = "Link: https://example.com/a?utm_source=x and https://example.com/a"
        urls = extract_urls(text)
        assert len(urls) == 1  # Same URL after normalization

    def test_ignores_non_urls(self):
        text = "No URLs here, just text."
        urls = extract_urls(text)
        assert len(urls) == 0

    def test_handles_urls_in_angle_brackets(self):
        text = "See <https://example.com/page> for details"
        urls = extract_urls(text)
        assert len(urls) == 1
        assert ">" not in urls[0]


class TestIngestUrls:
    def setup_method(self):
        """Use a temp DB for each test."""
        self._tmpdir = tempfile.mkdtemp()
        self._orig_db = None

    def _patch_db(self):
        """Patch DB_PATH to use temp directory."""
        import config
        self._orig_db = config.DB_PATH
        config.DB_PATH = type(config.DB_PATH)(self._tmpdir) / "test.db"

    def teardown_method(self):
        if self._orig_db:
            import config
            config.DB_PATH = self._orig_db

    def test_ingest_new_urls(self):
        self._patch_db()
        stats = ingest_urls(["https://example.com/a", "https://example.com/b"])
        assert stats["added"] == 2
        assert stats["skipped"] == 0

    def test_dedup_on_second_ingest(self):
        self._patch_db()
        ingest_urls(["https://example.com/a"])
        stats = ingest_urls(["https://example.com/a", "https://example.com/b"])
        assert stats["added"] == 1
        assert stats["skipped"] == 1

    def test_empty_list(self):
        self._patch_db()
        stats = ingest_urls([])
        assert stats["total"] == 0
