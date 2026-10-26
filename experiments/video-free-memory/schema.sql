-- Chronicle VFM vault schema (v0)
-- One SQLite file holds the compiled memory for one episode.
-- No video is referenced; `video_retained` is expected to be 0.

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
    zone           TEXT,
    duration_s     REAL,
    fps            REAL,
    video_retained INTEGER NOT NULL DEFAULT 0,
    compiled_at    REAL NOT NULL,
    compiler       TEXT NOT NULL DEFAULT 'vfm-v0'
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    state       TEXT,
    zone        TEXT,
    first_seen  REAL,
    last_seen   REAL,
    attributes  TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    t          REAL NOT NULL,
    kind       TEXT NOT NULL,          -- state | interaction | anchor | relation | zone | keyframe
    subject    TEXT,
    relation   TEXT,
    object     TEXT,
    zone       TEXT,
    confidence REAL,
    source     TEXT,
    caption    TEXT NOT NULL DEFAULT '',
    evidence_id TEXT,
    meta       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_t ON events(t);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject, t);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, t);

CREATE TABLE IF NOT EXISTS captions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start    REAL NOT NULL,
    t_end      REAL NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'window',  -- window | event
    text       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    evidence_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_captions_time ON captions(t_start, t_end);

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
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind    TEXT NOT NULL,       -- text | image
    ref_id  TEXT NOT NULL,
    model   TEXT NOT NULL,
    vector  BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_ref ON embeddings(kind, ref_id);

CREATE TABLE IF NOT EXISTS qa_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    question_hash TEXT NOT NULL UNIQUE,
    answer      TEXT NOT NULL,
    citations   TEXT NOT NULL DEFAULT '[]',
    evidence    TEXT NOT NULL DEFAULT '[]',
    confidence  REAL NOT NULL DEFAULT 0.0,
    method      TEXT NOT NULL DEFAULT 'vault',
    created_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_qa_hash ON qa_cache(question_hash);
