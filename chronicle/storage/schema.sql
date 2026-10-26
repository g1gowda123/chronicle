-- Chronicle episodic memory store (Role B).
-- Invariant: rows in `anchors` are never deleted. Bounding-box trajectories in
-- `observations` are the only rows subject to the 10-minute retention purge.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS zones (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    embedding   BLOB,
    created_at  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    attrs       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS furniture (
    id         TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    zone_id    TEXT REFERENCES zones(id) ON DELETE RESTRICT,
    created_at REAL NOT NULL,
    attrs      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    state       TEXT NOT NULL,
    state_since REAL NOT NULL,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    bbox        TEXT,
    embedding   BLOB,
    track_id    INTEGER,
    zone_id     TEXT,
    attrs       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS state_transitions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    from_state TEXT NOT NULL,
    to_state   TEXT NOT NULL,
    t_event    REAL NOT NULL,
    evidence   TEXT NOT NULL DEFAULT '{}',
    clip_ref   TEXT
);

CREATE INDEX IF NOT EXISTS idx_transitions_entity_time
    ON state_transitions(entity_id, t_event);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    relation    TEXT NOT NULL,
    t_start     REAL,
    t_end       REAL,
    is_terminal INTEGER NOT NULL DEFAULT 0,
    props       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, relation);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, relation);
CREATE INDEX IF NOT EXISTS idx_edges_time ON edges(t_start);

CREATE TABLE IF NOT EXISTS anchors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    anchor_id  TEXT NOT NULL,
    relation   TEXT NOT NULL DEFAULT 'RESTING_ON',
    t_start    REAL NOT NULL,
    t_end      REAL,
    is_terminal INTEGER NOT NULL DEFAULT 1,
    retained   INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_anchors_entity
    ON anchors(entity_id, is_terminal DESC, t_start DESC);

CREATE TABLE IF NOT EXISTS observations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    t         REAL NOT NULL,
    bbox      TEXT,
    state     TEXT
);

CREATE INDEX IF NOT EXISTS idx_observations_time ON observations(t);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id, t);
