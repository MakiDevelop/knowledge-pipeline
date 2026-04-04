"""Tests for enrich.py — HTML extraction and SSRF protection."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from enrich import _is_private_url, extract_text_from_html, extract_title_from_html


class TestIsPrivateUrl:
    def test_blocks_localhost(self):
        assert _is_private_url("http://localhost:8080/api") is True

    def test_blocks_127(self):
        assert _is_private_url("http://127.0.0.1/secret") is True

    def test_blocks_private_ip(self):
        assert _is_private_url("http://192.168.1.100/admin") is True
        assert _is_private_url("http://10.0.0.1/internal") is True

    def test_blocks_aws_metadata(self):
        assert _is_private_url("http://169.254.169.254/latest/meta-data/") is True

    def test_blocks_gcp_metadata(self):
        assert _is_private_url("http://metadata.google.internal/computeMetadata/v1/") is True

    def test_allows_public_urls(self):
        assert _is_private_url("https://example.com/page") is False
        assert _is_private_url("https://github.com/repo") is False

    def test_allows_public_ip(self):
        assert _is_private_url("http://8.8.8.8/dns") is False


class TestExtractTextFromHtml:
    def test_basic_extraction(self):
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        assert "Hello world" in text

    def test_strips_script_tags(self):
        html = "<html><body><script>var x = 1;</script><p>Content</p></body></html>"
        text = extract_text_from_html(html)
        assert "var x" not in text
        assert "Content" in text

    def test_strips_style_tags(self):
        html = "<html><body><style>body{color:red}</style><p>Visible</p></body></html>"
        text = extract_text_from_html(html)
        assert "color:red" not in text
        assert "Visible" in text

    def test_strips_nav_footer(self):
        html = "<html><body><nav>Menu</nav><main>Article</main><footer>Copyright</footer></body></html>"
        text = extract_text_from_html(html)
        assert "Menu" not in text
        assert "Article" in text
        assert "Copyright" not in text

    def test_handles_malformed_html(self):
        html = "<p>Unclosed tag<div>Nested"
        text = extract_text_from_html(html)
        assert "Unclosed tag" in text


class TestExtractTitleFromHtml:
    def test_extracts_title(self):
        html = "<html><head><title>My Page</title></head></html>"
        assert extract_title_from_html(html) == "My Page"

    def test_returns_none_when_missing(self):
        html = "<html><head></head></html>"
        assert extract_title_from_html(html) is None

    def test_strips_whitespace(self):
        html = "<html><head><title>  Spaced Title  </title></head></html>"
        assert extract_title_from_html(html) == "Spaced Title"
