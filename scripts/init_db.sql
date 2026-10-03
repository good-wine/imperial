-- Provenance events (SOTA hybrid schema). pgvector extension used when
-- available; Hamming/cosine fusion in Python keeps parity without it.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS provenance_events (
    event_id       TEXT PRIMARY KEY,
    description    TEXT NOT NULL,
    event_date     TEXT,
    source         TEXT,
    phash_hex      TEXT NOT NULL,
    dhash_hex      TEXT,
    pdq_hex        TEXT,
    clip_embedding vector(512),
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION
);

-- Cosine HNSW for semantic tier (IF NOT EXISTS guards older pgvector).
CREATE INDEX IF NOT EXISTS prov_clip_hnsw
  ON provenance_events USING hnsw (clip_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS prov_phash_idx ON provenance_events (phash_hex);
