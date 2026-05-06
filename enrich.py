#!/usr/bin/env python3
"""Layer 2: Enrich — Fetch full text and generate LLM summaries.

For each pending URL: fetch HTML, extract text, call LLM for
core_insight and summary. Pure stdlib HTTP, no external dependencies.

Usage:
  python3 enrich.py              # Process all pending items
  python3 enrich.py --limit 10   # Process up to 10 items
"""

import argparse
import html.parser
import ipaddress
import json
import re
import socket
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
    get_db_connection,
    init_db,
)

USER_AGENT = (
    "Mozilla/5.0 (compatible; knowledge-pipeline/1.0; "
    "+https://github.com/makifordevelop/knowledge-pipeline)"
)

SKIP_DOMAINS = {"apps.apple.com", "drive.google.com", "play.google.com"}

MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5MB limit to prevent OOM

# Hostnames to always block (SSRF protection)
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, link-local, or reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        return True  # unparseable = block


def _is_private_url(url: str) -> bool:
    """Block requests to private/internal IPs (SSRF protection).

    Resolves DNS first to prevent rebinding attacks. Checks ALL resolved
    IPs, not just the hostname string.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTNAMES:
        return True
    # Resolve DNS and check ALL resolved IPs
    try:
        for _, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            if _is_private_ip(sockaddr[0]):
                return True
    except socket.gaierror:
        return True  # unresolvable = block
    return False


class _SSRFSafeRedirectHandler(HTTPRedirectHandler):
    """Validate redirect targets against SSRF blocklist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _is_private_url(newurl):
            raise ValueError(f"Redirect to private/internal URL blocked: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_safe_opener = build_opener(_SSRFSafeRedirectHandler)


# ── HTML text extraction (zero dependencies) ──

class _HTMLTextExtractor(html.parser.HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self):
        return "\n".join(self.result)


def extract_text_from_html(html_content: str) -> str:
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_content)
    except Exception:
        pass
    return extractor.get_text()


def extract_title_from_html(html_content: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


# ── URL fetching ──

def fetch_url(url: str, timeout: int = 30) -> dict:
    """Fetch a URL and return {html, title, text, status}."""
    if _is_private_url(url):
        return {"status": "skipped", "reason": "blocked: private/internal URL"}
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with _safe_opener.open(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {"status": "skipped", "reason": f"non-html: {content_type}"}
            # Read with size limit to prevent OOM
            raw = resp.read(MAX_CONTENT_BYTES + 1)
            if len(raw) > MAX_CONTENT_BYTES:
                return {"status": "skipped", "reason": f"too large (>{MAX_CONTENT_BYTES // 1024 // 1024}MB)"}
            # Decode: try charset from header, then UTF-8, fallback latin-1
            charset = None
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].strip().split(";")[0]
            try:
                html_content = raw.decode(charset or "utf-8")
            except (UnicodeDecodeError, LookupError):
                html_content = raw.decode("latin-1")
        title = extract_title_from_html(html_content)
        text = extract_text_from_html(html_content)
        return {"status": "fetched", "html": html_content, "title": title, "text": text}
    except Exception as e:
        return {"status": "failed", "reason": str(e)[:200]}


# ── LLM enrichment ──

_ENRICH_PROMPT = """Analyze this web content and provide a structured summary.

Title: {title}
URL: {url}
Content (first 3000 chars):
{content}

Respond in strict JSON (no other text):
{{
  "core_insight": "<one sentence: the single most important takeaway>",
  "summary": "<2-3 sentences: what this content is about and why it matters>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}}"""


def call_llm(prompt: str) -> dict | None:
    """Call an OpenAI-compatible LLM API. Returns parsed JSON or None."""
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        req = Request(
            LLM_BASE_URL,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=LLM_TIMEOUT) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"]
        # Strip markdown code fences if present
        text = re.sub(r"^```json\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        return json.loads(text)
    except Exception:
        return None


def enrich_item(item_id: int, url: str, domain: str, conn) -> str:
    """Enrich a single item. Returns status string."""
    if domain in SKIP_DOMAINS:
        conn.execute(
            "UPDATE items SET fetch_status = 'skipped' WHERE id = ?", (item_id,)
        )
        return "skipped"

    result = fetch_url(url)
    now = datetime.now(timezone.utc).isoformat()

    if result["status"] != "fetched":
        conn.execute(
            "UPDATE items SET fetch_status = ?, fetched_at = ? WHERE id = ?",
            (result["status"], now, item_id),
        )
        return result["status"]

    title = result.get("title", "")
    full_text = result.get("text", "")

    # LLM enrichment
    prompt = _ENRICH_PROMPT.format(
        title=title or "(no title)",
        url=url,
        content=full_text[:3000] if full_text else "(empty)",
    )
    llm_result = call_llm(prompt)

    core_insight = ""
    summary = ""
    tags = "[]"
    if llm_result:
        core_insight = llm_result.get("core_insight", "")
        summary = llm_result.get("summary", "")
        tags = json.dumps(llm_result.get("tags", []), ensure_ascii=False)

    conn.execute(
        "UPDATE items SET title = ?, full_text = ?, summary = ?, "
        "core_insight = ?, tags = ?, fetch_status = 'fetched', fetched_at = ? "
        "WHERE id = ?",
        (title, full_text[:50000], summary, core_insight, tags, now, item_id),
    )
    return "fetched"


def main():
    parser = argparse.ArgumentParser(description="Enrich pending items with full text and LLM summaries")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process (0 = all)")
    args = parser.parse_args()

    init_db()
    conn = get_db_connection()

    query = (
        "SELECT id, url, domain FROM items "
        "WHERE fetch_status = 'pending' ORDER BY added_at"
    )
    query_params = []
    if args.limit > 0:
        query += " LIMIT ?"
        query_params.append(args.limit)

    rows = conn.execute(query, query_params).fetchall()
    print(f"[Enrich] {len(rows)} pending items")

    try:
        for i, row in enumerate(rows, 1):
            print(f"  [{i}/{len(rows)}] {row['domain']} — {row['url'][-40:]}", end=" ", flush=True)
            status = enrich_item(row["id"], row["url"], row["domain"], conn)
            print(status)
            if i % 10 == 0:
                conn.commit()
            time.sleep(0.5)

        conn.commit()
    finally:
        conn.close()
    print(f"Done: {len(rows)} items processed")


if __name__ == "__main__":
    main()
