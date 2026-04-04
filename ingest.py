#!/usr/bin/env python3
"""Layer 1: Ingest — Add URLs to the knowledge pipeline.

Accepts URLs from CLI args, a file (one URL per line), or stdin.
Normalizes URLs, deduplicates, and stores in SQLite.

Usage:
  python3 ingest.py https://example.com/article
  python3 ingest.py urls.txt
  echo "https://example.com" | python3 ingest.py --stdin
"""

import argparse
import hashlib
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from config import DB_PATH, get_db_connection, init_db

# Tracking parameters to strip
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "ref_src", "ref_url",
    "igsh", "si", "xmt", "slof", "hsLang",
}

URL_RE = re.compile(r"https?://[^\s<>\"']+")


def normalize_url(raw_url: str) -> str:
    """Strip tracking params and fragments."""
    parsed = urlparse(raw_url.strip().rstrip("/"))
    params = parse_qs(parsed.query, keep_blank_values=False)
    cleaned = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS}
    new_query = urlencode(cleaned, doseq=True) if cleaned else ""
    return urlunparse(parsed._replace(query=new_query, fragment=""))


def extract_urls(text: str) -> list[str]:
    """Extract and normalize all URLs from text."""
    raw = URL_RE.findall(text)
    seen = set()
    result = []
    for url in raw:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def ingest_urls(urls: list[str], source: str = "cli") -> dict:
    """Insert URLs into the database. Returns stats."""
    init_db()
    conn = get_db_connection()
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    skipped = 0

    for url in urls:
        domain = urlparse(url).netloc
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        try:
            conn.execute(
                "INSERT INTO items (url, domain, source, added_at, url_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (url, domain, source, now, url_hash),
            )
            added += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    conn.close()
    return {"added": added, "skipped": skipped, "total": len(urls)}


def main():
    parser = argparse.ArgumentParser(
        description="Ingest URLs into the knowledge pipeline",
        epilog="Examples:\n"
               "  python3 ingest.py https://example.com/article\n"
               "  python3 ingest.py urls.txt\n"
               '  echo "https://example.com" | python3 ingest.py --stdin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="URLs or path to a file containing URLs")
    parser.add_argument("--stdin", action="store_true", help="Read URLs from stdin")
    args = parser.parse_args()

    urls = []

    if args.stdin:
        urls = extract_urls(sys.stdin.read())
    elif args.inputs:
        for inp in args.inputs:
            path = Path(inp)
            if path.is_file():
                urls.extend(extract_urls(path.read_text()))
            elif inp.startswith("http"):
                urls.append(normalize_url(inp))
            else:
                print(f"Warning: skipping unrecognized input: {inp}", file=sys.stderr)
    else:
        parser.print_help()
        sys.exit(1)

    if not urls:
        print("No URLs found.", file=sys.stderr)
        sys.exit(1)

    stats = ingest_urls(urls)
    print(f"Ingested: {stats['added']} new, {stats['skipped']} duplicates (of {stats['total']} URLs)")


if __name__ == "__main__":
    main()
