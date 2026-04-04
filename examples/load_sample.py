#!/usr/bin/env python3
"""Load pre-enriched sample data into the database.

This lets you explore scoring, searching, and the API without needing
an LLM or embedding model. Great for a quick demo.

Usage:
  python3 examples/load_sample.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_db_connection, init_db


def main():
    sample_path = Path(__file__).parent / "sample_enriched.json"
    if not sample_path.exists():
        print("Error: sample_enriched.json not found", file=sys.stderr)
        sys.exit(1)

    with open(sample_path) as f:
        items = json.load(f)

    init_db()
    conn = get_db_connection()
    now = datetime.now(timezone.utc).isoformat()
    added = 0

    for item in items:
        try:
            conn.execute(
                "INSERT INTO items "
                "(url, domain, title, core_insight, summary, tags, "
                "fetch_status, fetched_at, "
                "knowledge_density, novelty, evidence_strength, actionability, "
                "risk_level, time_horizon, emotional_noise, source_credibility, "
                "signal_score, route_to, decision_reason, scored_at, "
                "source, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item["url"], item["domain"], item["title"],
                    item["core_insight"], item["summary"],
                    json.dumps(item.get("tags", []), ensure_ascii=False),
                    "fetched", now,
                    item["knowledge_density"], item["novelty"],
                    item["evidence_strength"], item["actionability"],
                    item["risk_level"], item["time_horizon"],
                    item["emotional_noise"], item["source_credibility"],
                    item["signal_score"], item["route_to"],
                    item["decision_reason"], now,
                    "sample", now,
                ),
            )
            added += 1
            print(f"  + [{item['signal_score']:3d}] [{item['route_to']:9s}] {item['title'][:60]}")
        except Exception as e:
            print(f"  ! Skipped {item['url']}: {e}")

    conn.commit()
    conn.close()

    print(f"\nLoaded {added} sample items into knowledge.db")
    print("\nTry:")
    print("  python3 -c \"from config import get_db_connection; c=get_db_connection(); [print(f'  [{r[\\\"signal_score\\\"]:3d}] {r[\\\"route_to\\\"]:9s} {r[\\\"title\\\"]}') for r in c.execute('SELECT title, signal_score, route_to FROM items ORDER BY signal_score DESC')]\"")


if __name__ == "__main__":
    main()
