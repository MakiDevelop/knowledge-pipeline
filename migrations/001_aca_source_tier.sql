-- 001: Add ACA source_tier column
-- ACA Layer 2 (Trust): every knowledge entry carries a trust tier.
-- Existing items are human-authored web content → raw_source.
-- Idempotent: ALTER TABLE with IF NOT EXISTS pattern via try/catch in caller.

ALTER TABLE items ADD COLUMN source_tier TEXT DEFAULT 'raw_source';
