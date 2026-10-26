-- Chronicle VFM vault schema v2 (compressed layout)
-- No video. Events are a packed struct-of-arrays blob; captions are zlib JSON.

PRAGMA page_size = 1024;
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    source_path    TEXT,
    zone_id        INTEGER,
    duration_s     REAL,
    fps            REAL,
    video_retained INTEGER NOT NULL DEFAULT 0,
    compiled_at    REAL NOT NULL,
    compiler       TEXT NOT NULL DEFAULT 'vfm-v2'
);

-- Dictionary encoding for every repeated string.
CREATE TABLE IF NOT EXISTS strings (
    id    INTEGER PRIMARY KEY,
    value TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS event_pack (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    blob BLOB NOT NULL,
    n    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id         TEXT PRIMARY KEY,
    label_id   INTEGER NOT NULL,
    state_id   INTEGER NOT NULL,
    zone_id    INTEGER NOT NULL,
    first_seen REAL,
    last_seen  REAL,
    attrs      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS captions_blob (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    blob BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id        TEXT PRIMARY KEY,
    t         REAL NOT NULL,
    kind      TEXT NOT NULL,
    label     TEXT,
    file      TEXT NOT NULL,
    rel_path  TEXT NOT NULL,
    bytes     INTEGER NOT NULL DEFAULT 0,
    width     INTEGER NOT NULL DEFAULT 0,
    height    INTEGER NOT NULL DEFAULT 0,
    meta      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evidence_t ON evidence(t);

CREATE TABLE IF NOT EXISTS embeddings (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind   TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    model  TEXT NOT NULL,
    dim    INTEGER NOT NULL,
    scale  REAL NOT NULL,
    data   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_ref ON embeddings(kind, ref_id);

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
