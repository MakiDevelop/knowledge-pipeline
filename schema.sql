-- knowledge-pipeline schema v1.0
-- Single-file SQLite database for the full pipeline.

CREATE TABLE IF NOT EXISTS items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    domain          TEXT,
    title           TEXT,
    -- Ingestion
    added_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    source          TEXT    DEFAULT 'cli',      -- cli / file / api
    -- Enrichment (Layer 2)
    full_text       TEXT,
    summary         TEXT,
    core_insight    TEXT,
    fetch_status    TEXT    DEFAULT 'pending',   -- pending / fetched / failed / skipped
    fetched_at      TEXT,
    -- Scoring (Layer 3)
    knowledge_density   INTEGER,
    novelty             INTEGER,
    evidence_strength   INTEGER,
    actionability       INTEGER,
    risk_level          INTEGER,
    time_horizon        TEXT,       -- short / mid / long
    emotional_noise     INTEGER,
    source_credibility  INTEGER,
    signal_score        INTEGER,    -- composite 0-100
    route_to            TEXT,       -- research / writer / action / validator / archive
    decision_reason     TEXT,
    scored_at           TEXT,
    prompt_version      TEXT,
    -- Embedding (Layer 4)
    embedding           TEXT,       -- JSON array (dense vector)
    sparse_weights      TEXT,       -- JSON object (sparse vector)
    embedded_at         TEXT,
    -- Metadata
    url_hash            TEXT,       -- SHA256 of URL for dedup
    tags                TEXT        -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_items_domain ON items(domain);
CREATE INDEX IF NOT EXISTS idx_items_signal ON items(signal_score);
CREATE INDEX IF NOT EXISTS idx_items_route ON items(route_to);
CREATE INDEX IF NOT EXISTS idx_items_fetch ON items(fetch_status);
