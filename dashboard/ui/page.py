"""NiceGUI page assembly for the Chronicle mission-control dashboard.

The page is intentionally mostly server-driven: every connected tab owns a
timer that pulls the shared :class:`~dashboard.runtime.Runtime` snapshot and
pushes a compact JSON payload into the client runtime in ``static/chronicle.js``.
"""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from ..runtime import runtime
from ..store import MemoryStore
from . import render, theme

RATES = (1.0, 0.5, 1.5, 2.0)
LOG_HISTORY = 180
CAPTION_HOLD_S = 3.4


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


def _sync_payload(
    store: MemoryStore,
    snapshot: dict[str, Any],
    *,
    graph: dict[str, Any] | None,
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
        "query_provider": snapshot.get("query_provider"),
    }


def _video_html(snapshot: dict[str, Any]) -> str:
    src = snapshot.get("video_url") or ""
    return f"""
<video class="chron-video" src="{render.esc(src)}" muted playsinline loop preload="auto"></video>
<div class="video-overlay" id="chron-overlay"></div>
{theme.VIDEO_HUD_HTML}
""".strip()


def _graph_body_html() -> str:
    return f"""
<div id="chron-graph"></div>
{theme.GRAPH_LEGEND_HTML}
""".strip()


def _drawer_shell_html() -> str:
    return f"""
<div class="drawer-backdrop" id="chron-backdrop" onclick="Chronicle.closeDrawer()"></div>
<aside class="inspector" id="chron-inspector" aria-label="Node inspector">
  <div id="chron-drawer-content">{theme.EMPTY_DRAWER_HTML}</div>
</aside>
""".strip()


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

    def snapshot() -> dict[str, Any]:
        return runtime.snapshot(mode=graph_mode)

    def push_log(record: Any) -> None:
        log.push(render.format_log(record), classes=render.log_classes(record))

    def sync(force_graph: bool = False) -> None:
        nonlocal last_graph_key
        snap = snapshot()
        key = (snap.get("revision"), round(float(snap.get("playhead") or 0.0) * 2) / 2, graph_mode)
        graph = None
        if force_graph or key != last_graph_key:
            graph = snap.get("graph")
            last_graph_key = key
        payload = _sync_payload(store, snap, graph=graph)
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
        text = (value if value is not None else query_input.value) or ""
        text = str(text).strip()
        if not text:
            return
        query_input.value = ""
        result.content = render.query_answer_html(
            {
                "route": "none",
                "answer": "Searching the episodic graph…",
                "provider": "local-graphrag",
                "model": "chronicle-local",
                "citations": [],
            }
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
        result.content = render.query_answer_html(answer)
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
        ui.run_javascript("window.Chronicle && window.Chronicle.scrollTelemetry();")

    async def submit_query() -> None:
        await run_query(query_input.value)

    def show_drawer(node_key: str) -> None:
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
            "(()=>{const el=document.getElementById('chron-drawer-content');"
            f"if(!el)return;el.innerHTML={_json(fragment)};"
            "window.Chronicle&&window.Chronicle.openDrawer();})();"
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

    def on_rate(_: Any = None) -> None:
        index = RATES.index(runtime.rate) if runtime.rate in RATES else 0
        runtime.set_rate(RATES[(index + 1) % len(RATES)])
        sync()

    def on_graph_mode(event: Any) -> None:
        nonlocal graph_mode
        args = event.args if isinstance(event.args, list) else [event.args]
        graph_mode = "full" if str(args[0]).lower() == "full" else "live"
        sync(force_graph=True)

    def on_node(event: Any) -> None:
        args = event.args if isinstance(event.args, list) else [event.args]
        if args and args[0]:
            show_drawer(str(args[0]))

    async def on_query(event: Any) -> None:
        args = event.args if isinstance(event.args, list) else [event.args]
        if args and args[0]:
            await run_query(str(args[0]))

    # ---- layout ------------------------------------------------------- #

    initial = snapshot()

    with ui.element("div").classes("chron-app"):
        ui.html(theme.topbar_html(store.session, initial), sanitize=False)

        with ui.element("div").classes("workspace"):
            with ui.element("div").classes("stage-col"):
                with ui.element("div").classes("panel video-panel"):
                    ui.html(theme.video_head_html(), sanitize=False)
                    with ui.element("div").classes("video-shell"):
                        ui.html(_video_html(initial), sanitize=False)
                with ui.element("div").classes("panel timeline-panel"):
                    ui.html(
                        render.timeline_html(
                            runtime.timeline(),
                            runtime.duration,
                            runtime.playhead,
                            playing=runtime.playing,
                            rate=runtime.rate,
                            mode=runtime.mode,
                            source_label=runtime.source_label,
                        ),
                        sanitize=False,
                    )

            with ui.element("div").classes("panel graph-panel"):
                ui.html(theme.graph_head_html(), sanitize=False)
                with ui.element("div").classes("graph-body"):
                    ui.html(_graph_body_html(), sanitize=False)
                ui.html(theme.GRAPH_FOOT_HTML, sanitize=False)

            with ui.element("div").classes("panel telemetry-panel"):
                ui.html(theme.TELEMETRY_HEAD_HTML, sanitize=False)
                log = ui.log(max_lines=700).classes("chron-log")

            with ui.element("div").classes("panel query-panel"):
                ui.html(theme.QUERY_HEAD_HTML, sanitize=False)
                with ui.element("div").classes("query-body"):
                    with ui.element("div").classes("query-row"):
                        query_input = (
                            ui.input(placeholder="Ask the episodic memory…")
                            .classes("chron-query-input")
                            .props("dense borderless")
                        )
                        ask_button = ui.button("ASK").classes("chron-ask")
                    ui.html(render.suggestions_html(), sanitize=False)
                    result = ui.html(render.query_hint_html(), sanitize=False)

        ui.html(_drawer_shell_html(), sanitize=False)

    ask_button.on("click", submit_query)
    query_input.on("keydown.enter", submit_query)

    ui.on("chron-seek", on_seek)
    ui.on("chron-toggle", on_toggle)
    ui.on("chron-restart", on_restart)
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
