"""NiceGUI page assembly for the Chronicle mission-control dashboard (zine edition).

The page stays mostly server-driven: every connected tab owns a timer that pulls
the shared :class:`~dashboard.runtime.Runtime` snapshot and pushes a compact JSON
payload into the client runtime in ``static/chronicle.js``. The layout mirrors the
hackathon zine: big feed + chat, right-rail graph / dossier / entity list, and a
bottom strip of state changes, actions and vitals.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nicegui import ui

from ..replay_db import events_from_db, has_memory, write_events
from ..runtime import Runtime, clip_title, default_data_dir, probe_duration, runtime
from ..store import STATE_COLORS, MemoryStore
from . import render, theme

LOG_HISTORY = 180
CAPTION_HOLD_S = 3.4
MARKERS_LIMIT = 80


# --------------------------------------------------------------------------- #
# Clip registry (uses the existing chronicle.sources backend + uploads)
# --------------------------------------------------------------------------- #

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


def _session_title(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("kind") == "session":
                    return str((payload.get("data") or {}).get("title") or "")
    except (OSError, json.JSONDecodeError):
        return ""
    return ""


def _clip_entry(video: Path, base: Path) -> dict[str, str]:
    rel = video.relative_to(base).as_posix()
    events = video.with_name(video.stem + ".events.jsonl")
    if events.is_file():
        return {
            "value": "events:" + events.relative_to(base).as_posix(),
            "title": _session_title(events) or clip_title(video),
        }
    return {"value": "video:" + rel, "title": clip_title(video)}


def _clip_options() -> list[dict[str, str]]:
    base = default_data_dir()
    options: list[dict[str, str]] = []
    seen: set[str] = set()

    demo = base / "demo_events.jsonl"
    if demo.is_file():
        options.append(
            {"value": "events:demo_events.jsonl", "title": _session_title(demo) or "Synthetic demo"}
        )
        seen.add("demo_events.jsonl")

    try:
        from chronicle.sources import available_sources

        for source in available_sources(base):
            video = source.video_path(base)
            if not video.is_file() or video == base / "demo" / "chronicle_demo.mp4":
                continue
            entry = _clip_entry(video, base)
            if entry["value"].startswith("video:"):
                # The clip already has a Role B memory DB: replay it instead of
                # playing a memory-less video.
                try:
                    db_path = source.db_path(base)
                except (OSError, ValueError):
                    db_path = None
                if db_path is not None and has_memory(db_path):
                    events_path = video.with_name(video.stem + ".events.jsonl")
                    if not events_path.is_file():
                        rel_video = video.relative_to(base).as_posix()
                        events = events_from_db(
                            db_path,
                            title=source.label,
                            video_url="/media/" + rel_video,
                            video_path=video,
                        )
                        if events:
                            write_events(events, events_path)
                    if events_path.is_file():
                        entry = {
                            "value": "events:" + events_path.relative_to(base).as_posix(),
                            "title": source.label,
                        }
            if entry["value"].split(":", 1)[1] in seen:
                continue
            if not entry["value"].startswith("events:"):
                entry["title"] = source.label
            options.append(entry)
            seen.add(entry["value"].split(":", 1)[1])
    except Exception:  # registry must never take the dashboard down
        pass

    stems = {Path(option["value"].split(":", 1)[1]).stem for option in options}
    for folder in (base / "real", base / "uploads"):
        if not folder.is_dir():
            continue
        for video in sorted(folder.rglob("*")):
            if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            entry = _clip_entry(video, base)
            rel = entry["value"].split(":", 1)[1]
            if rel in seen or rel == "demo/chronicle_demo.mp4" or video.stem in stems:
                continue
            options.append(entry)
            seen.add(rel)
            stems.add(video.stem)

    options.sort(key=lambda option: not option["value"].startswith("events:"))
    return options


def _active_clip(options: list[dict[str, str]]) -> str:
    env = os.getenv("CHRONICLE_REPLAY") or ""
    if env:
        try:
            rel = Path(env).resolve().relative_to(default_data_dir().resolve()).as_posix()
            for option in options:
                if option["value"] == "events:" + rel:
                    return option["value"]
        except (OSError, ValueError):
            pass
    return options[0]["value"] if options else ""


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _latest_interaction(store: MemoryStore, t: float) -> tuple[Any, Any] | None:
    """Return ``(record, entity)`` for the most recent VLM interaction at *t*."""
    best: tuple[Any, Any] | None = None
    for entity in store.entities.values():
        for record in entity.interactions:
            if record.t > t + 0.05:
                break
            if t - record.t > CAPTION_HOLD_S:
                continue
            if best is None or record.t > best[0].t:
                best = (record, entity)
    return best


def _hud(store: MemoryStore, snapshot: dict[str, Any], boxes: list[dict[str, Any]]) -> dict[str, Any]:
    t = float(snapshot.get("playhead") or 0.0)
    stats = snapshot.get("stats") or {}
    camera = stats.get("camera") or {}
    zone_id = stats.get("active_zone")
    zone = store.zones.get(str(zone_id)) if zone_id else None

    caption = None
    latest = _latest_interaction(store, t)
    if latest is not None:
        record, entity = latest
        caption = {
            "t": round(record.t, 2),
            "action": record.action.replace("_", " "),
            "entity": entity.label,
            "text": record.caption or "no caption",
        }

    return {
        "camera": camera if isinstance(camera, dict) else {},
        "zone": zone.label if zone is not None else "—",
        "tracks": len(boxes),
        "total": stats.get("entities") or 0,
        "latency": stats.get("latency_ms"),
        "caption": caption,
    }


def _entity_colour(entity: Any) -> str:
    attributes = entity.attributes or {}
    for key in ("colour_hex", "color_hex", "swatch_hex", "dominant_hex"):
        value = attributes.get(key)
        if isinstance(value, str) and value.startswith("#"):
            return value
    swatch = attributes.get("swatch")
    if isinstance(swatch, str) and swatch.startswith("#"):
        return swatch
    return STATE_COLORS.get(entity.state_at(float(runtime.playhead)), "#ffd34d")


def _entities_payload(store: MemoryStore) -> list[dict[str, Any]]:
    t = float(runtime.playhead)
    out: list[dict[str, Any]] = []
    for entity in store.entities.values():
        state = entity.state_at(t)
        zone = store.zones.get(entity.zone_id or "") if entity.zone_id else None
        out.append(
            {
                "id": entity.id,
                "label": entity.label,
                "cls": entity.cls or "object",
                "state": state,
                "zone": zone.label if zone is not None else "—",
                "conf": round(float(entity.confidence or 0.0), 3),
                "seen": bool(entity.first_seen > 0 or not entity.is_stub),
                "first": entity.first_seen,
                "last": entity.last_seen,
                "color": _entity_colour(entity),
                "interactions": len(entity.interactions),
            }
        )
    out.sort(key=lambda item: item["seen"], reverse=True)
    return out


def _markers_payload(rt: Runtime) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for marker in rt.timeline(include_future=False):
        kind = str(marker.get("kind") or "")
        if kind not in {"state", "interaction", "edge", "zone"}:
            continue
        markers.append(
            {
                "t": float(marker.get("t") or 0.0),
                "kind": kind,
                "label": str(marker.get("label") or kind),
                "entity_id": marker.get("entity_id"),
            }
        )
    return markers[-MARKERS_LIMIT:]


def _sync_payload(
    store: MemoryStore,
    snapshot: dict[str, Any],
    *,
    graph: dict[str, Any] | None,
    entities: list[dict[str, Any]] | None,
    markers: list[dict[str, Any]] | None,
    replay: str | None = None,
) -> dict[str, Any]:
    t = float(snapshot.get("playhead") or 0.0)
    boxes = store.boxes_at(t)
    stats = dict(snapshot.get("stats") or {})
    stats["active_entities"] = len(boxes)
    return {
        "t": round(t, 3),
        "duration": snapshot.get("duration"),
        "playing": snapshot.get("playing"),
        "rate": snapshot.get("rate"),
        "mode": snapshot.get("mode"),
        "source_label": snapshot.get("source_label"),
        "title": snapshot.get("title"),
        "video_url": snapshot.get("video_url"),
        "fps": snapshot.get("fps"),
        "width": snapshot.get("width"),
        "height": snapshot.get("height"),
        "boxes": boxes,
        "hud": _hud(store, snapshot, boxes),
        "stats": stats,
        "graph": graph,
        "entities": entities,
        "markers": markers,
        "replay": replay,
        "suggestions": list(render.SUGGESTIONS),
        "query_provider": snapshot.get("query_provider"),
    }


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def build_dashboard() -> None:
    """Render the dashboard into the current NiceGUI page."""
    ui.add_head_html(theme.HEAD_HTML)

    store = runtime.store
    graph_mode = "live"
    last_log_seq = 0
    last_epoch = -1
    last_graph_key: tuple[Any, ...] | None = None
    last_entities_key: tuple[Any, ...] | None = None
    last_markers_key: tuple[Any, ...] | None = None

    def snapshot() -> dict[str, Any]:
        return runtime.snapshot(mode=graph_mode)

    def push_log(record: Any) -> None:
        log.push(render.format_log(record), classes=render.log_classes(record))

    def sync(force_graph: bool = False) -> None:
        nonlocal last_graph_key, last_entities_key, last_markers_key
        snap = snapshot()
        t = float(snap.get("playhead") or 0.0)
        revision = snap.get("revision")
        epoch = store.epoch

        key = (revision, round(t * 2) / 2, graph_mode)
        graph = None
        if force_graph or key != last_graph_key:
            graph = snap.get("graph")
            last_graph_key = key

        entities = None
        ekey = (epoch, revision, round(t, 1))
        if ekey != last_entities_key:
            entities = _entities_payload(store)
            last_entities_key = ekey

        markers = None
        mkey = (epoch, revision, round(t, 1))
        if mkey != last_markers_key:
            markers = _markers_payload(runtime)
            last_markers_key = mkey

        payload = _sync_payload(
            store, snap, graph=graph, entities=entities, markers=markers, replay=active_clip
        )
        ui.run_javascript(f"window.Chronicle && window.Chronicle.sync({_json(payload)});")

    def flush_logs() -> None:
        nonlocal last_log_seq, last_epoch
        snap_epoch = store.epoch
        if snap_epoch != last_epoch:
            last_epoch = snap_epoch
            log.clear()
            recent = store.recent_logs(LOG_HISTORY)
            for record in recent:
                push_log(record)
            last_log_seq = recent[-1].seq if recent else 0
            return
        new = store.log_since(last_log_seq)
        if new:
            for record in new:
                push_log(record)
            last_log_seq = new[-1].seq

    # ---- actions ------------------------------------------------------ #

    async def run_query(value: str | None = None) -> None:
        text = (value if value is not None else "") or ""
        text = str(text).strip()
        if not text:
            return
        ui.run_javascript(
            f"window.Chronicle && window.Chronicle.chatQuestion({_json(text)});"
        )
        try:
            answer = await runtime.query.answer_async(text, runtime.playhead)
        except Exception as error:  # a query must never take down the dashboard
            answer = {
                "route": "none",
                "answer": f"Query failed: {error}",
                "provider": "local-graphrag",
                "model": "chronicle-local",
                "citations": [],
            }
        ui.run_javascript(
            f"window.Chronicle && window.Chronicle.chatAnswer({_json(answer)});"
        )
        entity_id = answer.get("matched_entity")
        store.note(
            "QUERY",
            f'Q "{text}" → {answer.get("route", "none")} · '
            f'{len(answer.get("citations") or [])} citations · '
            f'{answer.get("duration_ms", 0)}ms',
            runtime.playhead,
            entity_id=entity_id,
            query=text,
            route=answer.get("route"),
        )
        flush_logs()

    def show_insp(node_key: str) -> None:
        resolved = store.resolve_node(str(node_key))
        if resolved is None:
            return
        kind, real_id = resolved
        t = runtime.playhead
        if kind == "entity":
            detail = store.entity_detail(real_id, t)
            if detail is None:
                return
            fragment = render.entity_drawer_html(detail, t, store)
            store.note(
                "GRAPH",
                f"inspect {detail.get('label', real_id)} · state {detail.get('state_at_playhead')} "
                f"· {len(detail.get('history') or [])} transitions",
                t,
                entity_id=real_id,
            )
        elif kind == "zone":
            detail = store.zone_detail(real_id)
            if detail is None:
                return
            fragment = render.zone_drawer_html(detail, t, store)
            store.note(
                "GRAPH",
                f"inspect zone {detail.get('label', real_id)} · "
                f"{len(detail.get('entities') or [])} objects",
                t,
            )
        elif kind == "anchor":
            fragment = render.anchor_drawer_html(real_id, store, t)
            store.note("GRAPH", f"inspect anchor {real_id}", t)
        else:
            fragment = render.user_drawer_html(store)
        script = (
            "(()=>{const el=document.getElementById('chron-inspector-content');"
            f"if(!el)return;el.innerHTML={_json(fragment)};"
            "const card=el.firstElementChild;if(card)card.classList.add('popin');"
            "el.scrollIntoView({behavior:'smooth',block:'nearest'});})();"
        )
        ui.run_javascript(script)

    # ---- events ------------------------------------------------------- #

    def on_seek(event: Any) -> None:
        args = event.args if isinstance(event.args, list) else [event.args]
        try:
            t = float(args[0])
        except (TypeError, ValueError, IndexError):
            return
        pause = bool(args[1]) if len(args) > 1 else False
        runtime.seek(t, playing=False if pause else None)
        flush_logs()
        ui.run_javascript(
            "(()=>{const el=document.getElementById('chron-play-icon');"
            f"if(el)el.textContent={_json('❚❚' if runtime.playing else '▶')};}})();"
        )
        sync(force_graph=True)

    def on_toggle(_: Any = None) -> None:
        runtime.toggle_playing()
        sync()

    def on_restart(_: Any = None) -> None:
        runtime.restart_demo(0.0)
        flush_logs()
        sync(force_graph=True)

    def on_replay(event: Any) -> None:
        nonlocal active_clip, last_graph_key, last_entities_key, last_markers_key
        args = event.args if isinstance(event.args, list) else [event.args]
        value = str(args[0]) if args and args[0] else ""
        if not value:
            return
        kind, _, rel = value.partition(":")
        if kind == "events":
            ok = runtime.load_replay(rel)
        elif kind == "video":
            path = (default_data_dir() / rel).resolve()
            try:
                path.relative_to(default_data_dir().resolve())
            except ValueError:
                return
            if not path.is_file():
                return
            runtime.play_video("/media/" + rel, clip_title(path), probe_duration(path))
            ok = True
        else:
            return
        if not ok:
            return
        active_clip = value
        last_graph_key = None
        last_entities_key = None
        last_markers_key = None
        flush_logs()
        ui.run_javascript("window.Chronicle && window.Chronicle.clipChanged();")
        sync(force_graph=True)

    def on_rate(_: Any = None) -> None:
        rates = (1.0, 0.5, 1.5, 2.0)
        index = rates.index(runtime.rate) if runtime.rate in rates else 0
        runtime.set_rate(rates[(index + 1) % len(rates)])
        sync()

    def on_graph_mode(event: Any) -> None:
        nonlocal graph_mode
        args = event.args if isinstance(event.args, list) else [event.args]
        graph_mode = "full" if str(args[0]).lower() == "full" else "live"
        sync(force_graph=True)

    def on_node(event: Any) -> None:
        args = event.args if isinstance(event.args, list) else [event.args]
        if args and args[0]:
            show_insp(str(args[0]))

    async def on_query(event: Any) -> None:
        args = event.args if isinstance(event.args, list) else [event.args]
        if args and args[0]:
            await run_query(str(args[0]))

    # ---- layout ------------------------------------------------------- #

    initial = snapshot()
    clip_options = _clip_options()
    active_clip = _active_clip(clip_options)

    with ui.element("div").classes("chron-app"):
        ui.html(theme.topbar_html(store.session, initial), sanitize=False)

        with ui.element("div").classes("row-main"):
            with ui.element("div").classes("col-l"):
                ui.html(
                    theme.video_html(initial, clip_options, active_clip), sanitize=False
                )
                ui.html(theme.chat_head_html(), sanitize=False)
            with ui.element("div").classes("rail"):
                ui.html(theme.graph_body_html(), sanitize=False)
                ui.html(theme.inspector_shell_html(), sanitize=False)
                ui.html(theme.entity_card_html(), sanitize=False)

        with ui.element("div").classes("row-events-actions"):
            ui.html(theme.events_card_html(), sanitize=False)
            ui.html(theme.actions_card_html(), sanitize=False)

        with ui.element("div").classes("row-status"):
            ui.html(theme.vitals_card_html(), sanitize=False)
            with ui.element("div").classes("zcard card-pad telemetry-card rot-r2"):
                ui.html(theme.telemetry_head_html(), sanitize=False)
                log = ui.log(max_lines=700).classes("chron-log")

        ui.html(theme.marquee_html(), sanitize=False)
        ui.html(theme.footer_html(), sanitize=False)

    ui.on("chron-seek", on_seek)
    ui.on("chron-toggle", on_toggle)
    ui.on("chron-restart", on_restart)
    ui.on("chron-replay", on_replay)
    ui.on("chron-rate", on_rate)
    ui.on("chron-graph-mode", on_graph_mode)
    ui.on("chron-node", on_node)
    ui.on("chron-query", on_query)

    # ---- initial paint ------------------------------------------------ #
    recent = store.recent_logs(LOG_HISTORY)
    for record in recent:
        push_log(record)
    last_log_seq = recent[-1].seq if recent else 0
    last_epoch = store.epoch
    sync(force_graph=True)

    def tick() -> None:
        flush_logs()
        sync()

    ui.timer(0.16, tick)
