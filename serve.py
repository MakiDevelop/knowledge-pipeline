#!/usr/bin/env python3
"""Layer 6: Serve — HTTP API for AI agents to query your knowledge base.

A zero-dependency HTTP server (stdlib only) that exposes semantic search
as a JSON API. Any AI agent (Claude, GPT, Gemini, etc.) can use this
as a knowledge source via MCP, function calling, or plain HTTP.

Usage:
  python3 serve.py                  # Start on default port (8780)
  python3 serve.py --port 9000      # Custom port
  python3 serve.py --rerank         # Enable cross-encoder reranking

Endpoints:
  GET  /search?q=...&k=10&domain=...&min_score=0
  GET  /stats
  GET  /health
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from FlagEmbedding import BGEM3FlagModel

from config import EMBED_MODEL, SERVE_PORT, get_db_connection, init_db
from search import hybrid_search, load_embeddings, rerank

# ── Global state (loaded at startup) ──

_model = None
_conn = None
_rows = None
_matrix = None
_sparse = None
_use_rerank = False
_stats = {"start_time": 0, "queries": 0, "avg_latency_ms": 0, "_latency_sum": 0}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Search</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 20px;
        }
        .result {
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
        }
        .score {
            font-weight: bold;
        }
        .route {
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Knowledge Search</h1>
    <form id="search-form">
        <input type="text" id="query" name="q" placeholder="Enter your search query">
        <button type="submit">Search</button>
    </form>
    <div id="results"></div>

    <script>
        const form = document.getElementById('search-form');
        const resultsDiv = document.getElementById('results');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const query = document.getElementById('query').value;

            const response = await fetch(`/search?q=${query}`);
            const data = await response.json();

            resultsDiv.innerHTML = '';
            if (data.results && data.results.length > 0) {
                data.results.forEach(result => {
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'result';
                    resultDiv.innerHTML = `
                        <div class="score">Score: ${result.signal_score.toFixed(3)}</div>
                        <div class="route">Route: ${result.route}</div>
                        <div>Title: ${result.title}</div>
                        <div>Core Insight: ${result.core_insight}</div>
                    `;
                    resultsDiv.appendChild(resultDiv);
                });
            } else {
                resultsDiv.innerHTML = '<p>No results found.</p>';
            }
        });
    </script>
</body>
</html>
"""


def _reload():
    """Load/reload embeddings from DB."""
    global _rows, _matrix, _sparse
    _rows, _matrix, _sparse = load_embeddings(_conn)
    print(f"  Loaded {len(_rows)} items with embeddings")


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._json_response({"status": "ok", "items": len(_rows) if _rows else 0})
        elif path == "/stats":
            self._json_response({
                "items_total": len(_rows) if _rows else 0,
                "queries_served": _stats["queries"],
                "avg_latency_ms": round(_stats["avg_latency_ms"], 1),
                "uptime_seconds": int(time.time() - _stats["start_time"]),
            })
        elif path == "/search":
            self._handle_search(params)
        elif path == "/reload":
            _reload()
            self._json_response({"status": "reloaded", "items": len(_rows)})
        elif path == "/":
            self._html_response()
        else:
            self._json_response({"error": "not found"}, status=404)

    def _html_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def _handle_search(self, params):
        q = params.get("q", [""])[0]
        if not q:
            self._json_response({"error": "missing ?q= parameter"}, status=400)
            return

        try:
            k = min(int(params.get("k", ["10"])[0]), 100)  # cap at 100
            min_score = int(params.get("min_score", ["0"])[0])
        except (ValueError, TypeError):
            self._json_response({"error": "k and min_score must be integers"}, status=400)
            return
        domain = params.get("domain", [None])[0]

        t0 = time.time()

        # Filter if needed
        if domain or min_score:
            filters = {}
            if domain:
                filters["domain"] = domain
            if min_score:
                filters["min_score"] = min_score
            rows, matrix, sparse = load_embeddings(_conn, filters)
        else:
            rows, matrix, sparse = _rows, _matrix, _sparse

        if not rows:
            self._json_response({"query": q, "results": [], "count": 0})
            return

        pool_size = min(50, len(rows)) if _use_rerank else k
        results = hybrid_search(q, _model, rows, matrix, sparse, top_k=pool_size)

        if _use_rerank and results:
            results = rerank(q, results, top_k=k)

        elapsed_ms = (time.time() - t0) * 1000
        _stats["queries"] += 1
        _stats["_latency_sum"] += elapsed_ms
        _stats["avg_latency_ms"] = _stats["_latency_sum"] / _stats["queries"]

        self._json_response({
            "query": q,
            "results": results,
            "count": len(results),
            "latency_ms": round(elapsed_ms, 1),
        })

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        if "/health" not in str(args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def main():
    global _model, _conn, _rows, _matrix, _sparse, _use_rerank

    parser = argparse.ArgumentParser(description="Knowledge search HTTP API server")
    parser.add_argument("--port", type=int, default=SERVE_PORT, help=f"Port (default {SERVE_PORT})")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    args = parser.parse_args()

    _use_rerank = args.rerank

    print("Starting knowledge-pipeline server...")
    init_db()
    _conn = get_db_connection()

    print(f"  Loading embedding model: {EMBED_MODEL}")
    _model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)
    _reload()

    _stats["start_time"] = time.time()

    server = HTTPServer(("0.0.0.0", args.port), SearchHandler)
    print(f"  Listening on http://0.0.0.0:{args.port}")
    print("  NOTE: This is a development server. Use a reverse proxy for production.")
    print("  Endpoints: /search?q=... /stats /health /reload")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()