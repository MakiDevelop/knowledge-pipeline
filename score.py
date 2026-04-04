#!/usr/bin/env python3
"""Layer 3: Score — Multi-dimensional LLM scoring and routing.

Scores each enriched item on 8 dimensions using an LLM, computes a
composite signal_score (0-100), and assigns a route (what to do with it).

This is the core differentiator of knowledge-pipeline: instead of
treating all bookmarks equally, we triage them automatically.

Usage:
  python3 score.py              # Score all unscored items
  python3 score.py --rescore    # Re-score everything
  python3 score.py --dry-run    # Preview without writing
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
    SCORING_PROMPT_VERSION,
    get_db_connection,
    init_db,
)

# ── Scoring dimensions ──
#
# Each dimension is scored 0-5 by the LLM.
# The prompt includes calibration baselines to ensure consistency.

SCORING_PROMPT = """You are a knowledge analyst. Score the following content on multiple dimensions.
Respond in strict JSON only (no other text).

Title: {title}
Core Insight: {core_insight}
Content snippet: {content}
Source domain: {domain}

Return this exact JSON structure:
{{
  "knowledge_density": <0-5, how structured and information-rich>,
  "novelty": <0-5, new perspectives or counter-intuitive insights>,
  "evidence_strength": <0-5, supported by data, examples, or logic>,
  "actionability": <0-5, can you act on this immediately>,
  "risk_level": <0-5, technical or societal risk involved>,
  "time_horizon": "<short|mid|long>",
  "emotional_noise": <0-5, how much emotion vs substance>,
  "source_credibility": <0-5, trustworthiness of the source>,
  "decision_reason": "<one sentence, why this matters or doesn't>"
}}

Calibration (follow strictly):
- evidence_strength: 1=speculation, 2=clear logic, 3=concrete examples, 4=data/benchmarks, 5=reproducible experiments
- emotional_noise: 0=purely objective, 1=opinionated but rational, 2=clearly subjective, 3=clickbait, 4+=empty hype
- knowledge_density: 1=single opinion, 2=thesis+support, 3=framework, 4=reusable framework, 5=systematic knowledge
- novelty: 1=common knowledge, 2=known idea new angle, 3=new framework, 4=counter-intuitive, 5=structural shift
- actionability: 1=info only, 2=changes thinking, 3=changes decisions, 4=can build/test now, 5=complete implementation guide
- source_credibility: 1=anonymous, 2=social media, 3=tech blog, 4=major publication/official, 5=academic paper/gov report"""

DEFAULT_SCORES = {
    "knowledge_density": 2,
    "novelty": 2,
    "evidence_strength": 2,
    "actionability": 2,
    "risk_level": 0,
    "time_horizon": "short",
    "emotional_noise": 2,
    "source_credibility": 2,
    "decision_reason": "Unable to score (LLM unavailable)",
}

VALID_TIME_HORIZONS = {"short", "mid", "long"}

# ── Routes ──
# Each item is routed to a destination based on its scores.
# Customize these for your workflow.

ROUTE_RESEARCH = "research"    # Needs deeper investigation
ROUTE_WRITER = "writer"        # Good for writing/publishing
ROUTE_ACTION = "action"        # Directly actionable
ROUTE_VALIDATOR = "validator"  # Needs fact-checking
ROUTE_ARCHIVE = "archive"      # Low priority, file away


def compute_route(scores: dict) -> str:
    """Determine where this item should go based on scores."""
    act = scores.get("actionability", 0)
    rl = scores.get("risk_level", 0)
    nov = scores.get("novelty", 0)
    es = scores.get("evidence_strength", 0)
    en = scores.get("emotional_noise", 0)
    kd = scores.get("knowledge_density", 0)

    # Validator: high risk or low evidence + high emotion
    if rl >= 3 or (es <= 2 and en >= 3):
        return ROUTE_VALIDATOR

    # Research: high novelty or needs more evidence
    if nov >= 4 or (es <= 2 and kd >= 3):
        return ROUTE_RESEARCH

    # Writer: high density + strong evidence + low noise
    if kd >= 3 and es >= 3 and en <= 2:
        return ROUTE_WRITER

    # Action: highly actionable and low risk
    if act >= 4 and rl <= 2:
        return ROUTE_ACTION

    return ROUTE_ARCHIVE


def compute_signal_score(scores: dict) -> int:
    """Composite signal score (0-100) for ranking and thresholds.

    Higher = more valuable knowledge. Used to decide publishing,
    memory storage, or deep-reading thresholds.
    """
    kd = scores.get("knowledge_density", 0)
    nov = scores.get("novelty", 0)
    es = scores.get("evidence_strength", 0)
    act = scores.get("actionability", 0)
    sc = scores.get("source_credibility", 2)
    en = scores.get("emotional_noise", 0)

    raw = (
        sc * 5 +        # Source credibility (trust matters most)
        nov * 4 +       # Novelty
        es * 4 +        # Evidence strength
        act * 3 +       # Actionability
        kd * 3 -        # Knowledge density
        en * 3          # Emotional noise penalty
    )
    # Theoretical max = 5*5 + 5*4 + 5*4 + 5*3 + 5*3 - 0*3 = 95
    return max(0, min(100, int(raw * 100 / 95)))


def _call_llm(prompt: str) -> dict | None:
    """Call LLM and return parsed JSON scores."""
    from urllib.request import Request, urlopen

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
        text = re.sub(r"^```json\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        # Strip <think> blocks (Qwen/reasoning models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return json.loads(text)
    except Exception as e:
        print(f"    LLM error: {e}", file=sys.stderr)
        return None


def validate_scores(raw: dict) -> dict:
    """Clamp and validate LLM output, fallback to defaults."""
    scores = {}
    for key in ("knowledge_density", "novelty", "evidence_strength",
                "actionability", "risk_level", "emotional_noise", "source_credibility"):
        val = raw.get(key, DEFAULT_SCORES[key])
        scores[key] = max(0, min(5, int(val))) if isinstance(val, (int, float)) else DEFAULT_SCORES[key]

    th = raw.get("time_horizon", "short")
    scores["time_horizon"] = th if th in VALID_TIME_HORIZONS else "short"
    scores["decision_reason"] = str(raw.get("decision_reason", ""))[:200]
    return scores


def score_item(item, conn, dry_run: bool = False) -> dict:
    """Score a single item. Returns the scores dict."""
    prompt = SCORING_PROMPT.format(
        title=item["title"] or "(no title)",
        core_insight=item["core_insight"] or "(none)",
        content=(item["full_text"] or "")[:2000],
        domain=item["domain"] or "(unknown)",
    )

    raw = _call_llm(prompt)
    scores = validate_scores(raw) if raw else DEFAULT_SCORES.copy()
    scores["signal_score"] = compute_signal_score(scores)
    scores["route_to"] = compute_route(scores)

    if not dry_run:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE items SET "
            "knowledge_density=?, novelty=?, evidence_strength=?, "
            "actionability=?, risk_level=?, time_horizon=?, "
            "emotional_noise=?, source_credibility=?, signal_score=?, "
            "route_to=?, decision_reason=?, scored_at=?, prompt_version=? "
            "WHERE id=?",
            (
                scores["knowledge_density"], scores["novelty"],
                scores["evidence_strength"], scores["actionability"],
                scores["risk_level"], scores["time_horizon"],
                scores["emotional_noise"], scores["source_credibility"],
                scores["signal_score"], scores["route_to"],
                scores["decision_reason"], now, SCORING_PROMPT_VERSION,
                item["id"],
            ),
        )

    return scores


def main():
    parser = argparse.ArgumentParser(description="Score enriched items with multi-dimensional LLM analysis")
    parser.add_argument("--limit", type=int, default=0, help="Max items to score")
    parser.add_argument("--rescore", action="store_true", help="Re-score already scored items")
    parser.add_argument("--dry-run", action="store_true", help="Preview scores without saving")
    args = parser.parse_args()

    init_db()
    conn = get_db_connection()

    where = "fetch_status = 'fetched'"
    if not args.rescore:
        where += " AND signal_score IS NULL"

    query = f"SELECT id, url, domain, title, core_insight, full_text FROM items WHERE {where} ORDER BY added_at"
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    rows = conn.execute(query).fetchall()
    print(f"[Score] {len(rows)} items to score")

    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {row['domain']}", end=" ", flush=True)
        scores = score_item(row, conn, dry_run=args.dry_run)
        sig = scores["signal_score"]
        route = scores["route_to"]
        print(f"signal={sig} route={route}")

        if i % 10 == 0:
            conn.commit()
        time.sleep(0.3)

    conn.commit()
    conn.close()
    print(f"Done: {len(rows)} items scored")


if __name__ == "__main__":
    main()
