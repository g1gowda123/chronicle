-- Chronicle VFM vault schema v3 additions.
-- The v2 tables stay intact so older readers keep working.

PRAGMA journal_mode = WAL;

-- Dense timestamped captions produced once by an offline VLM.
-- One row holds a zlib-compressed JSON document:
--   {version, model, prompt_version, window_s, windows:[{t_start,t_end,caption,
--    objects,actions,zone,tags,frames:[{t,caption}]}]}
CREATE TABLE IF NOT EXISTS caption_ledger (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    model      TEXT NOT NULL DEFAULT '',
    window_s   REAL NOT NULL DEFAULT 30.0,
    blob       BLOB NOT NULL,
    n          INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

-- Cached query answers. Present in schema_v2; repeated here for v3-only vaults.
CREATE TABLE IF NOT EXISTS qa_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question      TEXT NOT NULL,
    question_hash TEXT NOT NULL UNIQUE,
    answer        TEXT NOT NULL,
    citations     TEXT NOT NULL DEFAULT '[]',
    evidence      TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.0,
    method        TEXT NOT NULL DEFAULT 'vault',
    created_at    REAL NOT NULL,
    hits          INTEGER NOT NULL DEFAULT 0
);
