"""Server-side HTML renderers for the Chronicle dashboard."""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping

from ..contracts import STATE_COLORS, clean_text, pretty_ref
from ..store import ACTION_RELATIONS, MemoryStore

CHANNEL_COLORS = {
    "GEOM": "#38bdf8",
    "TRACK": "#a78bfa",
    "STATE": "#fbbf24",
    "VLM": "#e879f9",
    "GRAPH": "#34d399",
    "QUERY": "#f472b6",
    "SYSTEM": "#94a3b8",
}

MARKER_COLORS = {
    "state": "#fbbf24",
    "interaction": "#e879f9",
    "edge": "#34d399",
    "zone": "#22d3ee",
    "session": "#94a3b8",
}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def safe_url(value: Any) -> str:
    url = str(value or "").strip()
    if url.startswith(("/media/", "/chronicle/", "http://", "https://", "data:image/")):
        return esc(url)
    return ""


def fmt_time(value: Any, suffix: str = "s") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1f}{suffix}"


def fmt_clock(value: Any) -> str:
    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return "00:00.00"
    minutes = int(number // 60)
    rest = number - minutes * 60
    prefix = f"{minutes:02d}:" if minutes else ""
    return f"{prefix}{rest:05.2f}"


def _pct(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value))) * 100.0
    except (TypeError, ValueError):
        return 0.0


def _score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- #
# Transport / timeline
# --------------------------------------------------------------------------- #


def timeline_html(
    markers: Iterable[Mapping[str, Any]],
    duration: float,
    playhead: float,
    *,
    playing: bool,
    rate: float,
    mode: str,
    source_label: str,
) -> str:
    duration = max(0.01, float(duration or 0.01))
    playhead = max(0.0, min(float(playhead or 0.0), duration))

    marker_html: list[str] = []
    for marker in markers:
        kind = str(marker.get("kind") or "session")
        if kind not in MARKER_COLORS:
            continue
        try:
            t = float(marker.get("t") or 0.0)
        except (TypeError, ValueError):
            continue
        if t < 0 or t > duration + 0.05:
            continue
        label = str(marker.get("label") or kind).replace("_", " ")
        color = MARKER_COLORS.get(kind, "#94a3b8")
        future = " future" if marker.get("future") else ""
        left = max(0.0, min(100.0, (t / duration) * 100.0))
        marker_html.append(
            f'<span class="tl-marker kind-{esc(kind)}{future}" '
            f'style="left:{left:.3f}%;--marker:{color};" '
            f'title="{esc(label)} @ {t:.2f}s"></span>'
        )

    ticks: list[str] = []
    for index in range(6):
        value = duration * index / 5
        ticks.append(f"<span>{value:.0f}s</span>")

    loop_text = "LOOP" if mode == "demo" else "LIVE"
    icon = "❚❚" if playing else "▶"
    label = "SYNCED PLAYBACK" if mode == "demo" else "STREAMING"
    return f"""
<div class="transport">
  <button class="t-btn play" id="chron-play" onclick="Chronicle.togglePlay()" title="Play / pause (Space)">
    <span id="chron-play-icon">{icon}</span>
  </button>
  <button class="t-btn restart" onclick="Chronicle.restart()" title="Replay from 0:00 (R)">↺</button>
  <span class="t-time" id="chron-time">{fmt_clock(playhead)} <span class="total">/ {fmt_clock(duration)}</span></span>
  <span class="t-sep"></span>
  <span class="transport-label">{esc(label)} · {esc(loop_text)}</span>
  <span class="spacer"></span>
  <span class="t-loop">{esc(source_label)}</span>
  <button class="t-btn t-rate" id="chron-rate" onclick="Chronicle.cycleRate()" title="Playback speed">{(float(rate or 1.0)):.1f}×</button>
</div>
<div class="chron-timeline">
  <div class="tl-track">
    <div class="tl-fill" id="chron-fill" style="width:{(playhead / duration) * 100:.3f}%"></div>
    <div class="tl-scan"></div>
    {''.join(marker_html)}
  </div>
  <div class="tl-head" id="chron-head" style="left:{(playhead / duration) * 100:.3f}%"></div>
  <input class="chron-scrub" id="chron-scrub" type="range" min="0" max="{duration:.3f}"
         step="0.01" value="{playhead:.3f}" aria-label="Scrub episode timeline">
  <div class="tl-labels">{''.join(ticks)}</div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Query console
# --------------------------------------------------------------------------- #

SUGGESTIONS = (
    "Where are my keys?",
    "Where is my wallet?",
    "When did I last touch my phone?",
    "What did I do in this episode?",
)


def suggestions_html() -> str:
    buttons = "".join(
        f'<button class="suggestion" data-query="{esc(query)}" '
        f'onclick="Chronicle.ask(this.dataset.query)">{esc(query)}</button>'
        for query in SUGGESTIONS
    )
    return f'<div class="suggestions">{buttons}</div>'


def query_hint_html() -> str:
    return """
<div class="query-result">
  <div class="query-hint">
    <strong>Ask the episodic memory</strong>
    <span>Resolve natural language against the dual graph, then jump to the
    timestamped evidence behind every answer.</span>
    <span>Local GraphRAG is active. Set CHRONICLE_QUERY_URL to attach Role C's LLM.</span>
  </div>
</div>
""".strip()


def query_answer_html(result: Mapping[str, Any]) -> str:
    route = str(result.get("route") or "none")
    route_label = {
        "entity": "Object memory",
        "zone": "Place memory",
        "timeline": "Episode recall",
        "none": "No match",
    }.get(route, route.upper())
    answer = str(result.get("answer") or "No answer was produced.")
    provider = str(result.get("provider") or "local-graphrag")
    model = str(result.get("model") or "chronicle-local")
    duration = result.get("duration_ms")
    confidence = result.get("confidence")
    citations = result.get("citations") or []

    meta = f"{provider} · {model}"
    if duration is not None:
        meta += f" · {float(duration):.1f}ms"
    if confidence is not None:
        meta += f' · <span class="conf">conf {_score(confidence)}</span>'

    citation_html: list[str] = []
    for citation in citations:
        try:
            t = float(citation.get("t") or 0.0)
        except (TypeError, ValueError):
            t = 0.0
        kind = str(citation.get("kind") or "state")
        label = str(citation.get("label") or kind).replace("_", " ")
        detail = str(citation.get("detail") or "")
        citation_html.append(
            f'<button class="citation kind-{esc(kind)}" onclick="Chronicle.seek({t:.3f}, true)" '
            f'title="{esc(detail)}"><span class="cit-t">{t:.1f}s</span>'
            f"<span>{esc(label)}</span></button>"
        )

    evidence = "".join(citation_html) if citation_html else '<span class="empty-note">No timestamped evidence.</span>'
    error = result.get("llm_error")
    error_html = f'<div class="llm-error">LLM fallback: {esc(error)}</div>' if error else ""
    return f"""
<div class="query-result">
  <div class="answer-wrap">
    <div class="answer-top">
      <span class="answer-route">{esc(route_label)}</span>
      <span class="answer-provider">{meta}</span>
    </div>
    <p class="answer-text">{esc(answer)}</p>
    <div class="answer-citations">
      <span class="citations-label">Evidence</span>
      {evidence}
    </div>
    {error_html}
  </div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Inspector drawer
# --------------------------------------------------------------------------- #


def _state_chip(state: str) -> str:
    color = STATE_COLORS.get(state, "#94a3b8")
    return (
        f'<span class="state-chip" style="--state-color:{color}">'
        f"{esc(state.replace('_', ' '))}</span>"
    )


def _ref_label(store: MemoryStore, ref: Any) -> str:
    text = str(ref or "")
    if text.upper() in {"USER", "ME", "PERSON", "OPERATOR"}:
        return "YOU"
    entity = store.entities.get(text)
    if entity is not None:
        return entity.label
    zone = store.zones.get(text)
    if zone is not None:
        return zone.label
    return store.anchor_labels.get(text) or pretty_ref(text)


def _score_card(name: str, value: Any, *, accent: str = "#22d3ee") -> str:
    percentage = _pct(value) if value is not None else 0.0
    return f"""
<div class="score-card">
  <div class="score-name"><span>{esc(name)}</span><strong>{_score(value)}</strong></div>
  <div class="score-track"><div class="score-fill" style="width:{percentage:.1f}%;background:linear-gradient(90deg,{accent}55,{accent})"></div></div>
</div>
""".strip()


def _meta_cell(key: str, value: Any) -> str:
    return f'<div class="meta-cell"><div class="meta-key">{esc(key)}</div><div class="meta-val">{esc(value)}</div></div>'


def _state_history_html(history: list[Mapping[str, Any]], playhead: float) -> str:
    if not history:
        return '<div class="empty-note">No state transitions recorded yet.</div>'

    records = sorted(history, key=lambda item: float(item.get("t") or 0.0))
    current_index = -1
    for index, record in enumerate(records):
        if float(record.get("t") or 0.0) <= playhead + 0.05:
            current_index = index
    # Keep the drawer scannable while retaining the full early history.
    window_start = max(0, current_index - 10) if current_index >= 0 else max(0, len(records) - 12)
    shown = records[window_start:]
    if window_start:
        header = '<div class="empty-note">earlier transitions hidden</div>'
    else:
        header = ""

    steps: list[str] = []
    for index, record in enumerate(shown, start=window_start):
        from_state = str(record.get("from") or "—")
        to_state = str(record.get("to") or "—")
        t = float(record.get("t") or 0.0)
        color = STATE_COLORS.get(to_state, "#94a3b8")
        current = " current" if index == current_index else ""
        reason = clean_text(record.get("reason") or "", 220)
        triggers = record.get("trigger") or {}
        trigger_html = ""
        if isinstance(triggers, Mapping) and triggers:
            chips = []
            for key, value in list(triggers.items())[:4]:
                if isinstance(value, float):
                    rendered = f"{value:.2f}"
                else:
                    rendered = str(value)
                chips.append(f'<span class="trigger">{esc(key)}={esc(rendered)}</span>')
            trigger_html = f'<div class="trigger-row">{"".join(chips)}</div>'
        steps.append(
            f'<div class="state-step{current}" style="--step-color:{color}" '
            f'onclick="Chronicle.seek({t:.3f}, true)">'
            f'<div class="step-line"><span class="step-t">{t:.1f}s</span>'
            f'<span class="step-state">{esc(from_state.replace("_", " "))}'
            f'<span class="step-arrow"> → </span>{esc(to_state.replace("_", " "))}</span></div>'
            f'<div class="step-reason">{esc(reason or "no reason recorded")}</div>'
            f"{trigger_html}</div>"
        )
    return header + f'<div class="timeline-list">{"".join(steps)}</div>'


def _interactions_html(interactions: list[Mapping[str, Any]]) -> str:
    if not interactions:
        return '<div class="empty-note">No VLM interaction has been recorded for this node.</div>'

    cards: list[str] = []
    for interaction in reversed(interactions[-8:]):
        t = float(interaction.get("t") or 0.0)
        action = str(interaction.get("action") or "INTERACTION")
        caption = str(interaction.get("caption") or "")
        clip = interaction.get("clip_sim")
        model = str(interaction.get("model") or "Florence-2")
        latency = interaction.get("latency_ms")
        strip = safe_url(interaction.get("strip_url"))
        crop = safe_url(interaction.get("crop_url"))
        image = strip or crop
        image_html = (
            f'<img class="interaction-strip" src="{image}" alt="{esc(action)} evidence strip" '
            f'onclick="Chronicle.seek({t:.3f}, true)" loading="lazy">'
            if image
            else ""
        )
        latency_html = f" · {float(latency):.0f}ms" if latency is not None else ""
        cards.append(
            f'<div class="interaction-card">'
            f"{image_html}"
            f'<div class="interaction-body">'
            f'<div class="interaction-head"><span class="action-tag">{esc(action)}</span>'
            f'<span class="interaction-time">{t:.2f}s</span>'
            f'<span class="interaction-clip">CLIP {_score(clip)} · {esc(model)}{latency_html}</span></div>'
            f'<div class="interaction-caption">{esc(caption)}</div>'
            f"</div></div>"
        )
    return "".join(cards)


def _edges_html(store: MemoryStore, edges: list[Mapping[str, Any]]) -> str:
    if not edges:
        return '<div class="empty-note">No relations have been committed for this node.</div>'
    rows: list[str] = []
    for edge in sorted(edges, key=lambda item: float(item.get("t_start") or 0.0), reverse=True)[:12]:
        relation = str(edge.get("relation") or "")
        action = " action" if relation in ACTION_RELATIONS else ""
        source = _ref_label(store, edge.get("source"))
        target = _ref_label(store, edge.get("target"))
        t_start = float(edge.get("t_start") or 0.0)
        t_end = edge.get("t_end")
        end = "active" if t_end is None else f"{float(t_end):.1f}s"
        rows.append(
            f'<div class="edge-row"><span class="edge-rel{action}">{esc(relation)}</span>'
            f'<span class="edge-nodes">{esc(source)} → {esc(target)}</span>'
            f'<span class="edge-time">{t_start:.1f}s → {esc(end)}</span></div>'
        )
    return "".join(rows)


def entity_drawer_html(detail: Mapping[str, Any], playhead: float, store: MemoryStore) -> str:
    entity_id = str(detail.get("id") or "")
    label = str(detail.get("label") or pretty_ref(entity_id))
    state = str(detail.get("state_at_playhead") or detail.get("state") or "NEW")
    color = STATE_COLORS.get(state, "#94a3b8")
    zone = detail.get("zone") or {}
    zone_label = zone.get("label") if isinstance(zone, Mapping) else None
    crop = safe_url(detail.get("crop_url"))
    cls = str(detail.get("cls") or "object")
    track = detail.get("track_id") or "—"
    clip = detail.get("clip_sim")
    conf = detail.get("confidence")
    first_seen = detail.get("first_seen")
    last_seen = detail.get("last_seen")
    state_since = detail.get("state_since")
    last_interaction = detail.get("last_interaction_t")
    history = detail.get("history") or []
    interactions = detail.get("interactions") or []
    edges = detail.get("edges") or []

    crop_html = (
        f'<div class="crop-frame"><img src="{crop}" alt="{esc(label)} crop" loading="lazy">'
        f'<span class="crop-corner tl"></span><span class="crop-corner tr"></span>'
        f'<span class="crop-corner bl"></span><span class="crop-corner br"></span>'
        f'<span class="crop-cap">object crop · {esc(cls)}</span></div>'
        if crop
        else '<div class="crop-frame"><div class="empty-note">no crop available for this node</div></div>'
    )

    meta = "".join(
        [
            _meta_cell("First seen", fmt_time(first_seen)),
            _meta_cell("Last seen", fmt_time(last_seen)),
            _meta_cell("State since", fmt_time(state_since)),
            _meta_cell("Last interaction", fmt_time(last_interaction) if last_interaction is not None else "—"),
            _meta_cell("Track id", track),
            _meta_cell("Anchor zone", zone_label or "—"),
            _meta_cell("Class", cls),
            _meta_cell("Node id", entity_id),
        ]
    )

    return f"""
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Object node · live memory</div>
    <div class="inspector-title">{esc(label)} {_state_chip(state)}</div>
    <div class="inspector-sub">{esc(entity_id)} · {esc(cls)} · tracked {esc(track)}</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeDrawer()" aria-label="Close inspector">×</button>
</div>
<div class="inspector-body">
  {crop_html}
  <div class="score-grid">
    {_score_card("CLIP Re-ID", clip, accent=color)}
    {_score_card("Detection conf", conf, accent="#35d07f")}
  </div>
  <div class="meta-grid">{meta}</div>

  <div class="section-head">State history <span class="count">{len(history)} transitions</span></div>
  {_state_history_html(history, playhead)}

  <div class="section-head">VLM interactions <span class="count">{len(interactions)} events</span></div>
  {_interactions_html(interactions)}

  <div class="section-head">Relations <span class="count">{len(edges)} edges</span></div>
  {_edges_html(store, edges)}
</div>
""".strip()


def zone_drawer_html(detail: Mapping[str, Any], playhead: float, store: MemoryStore) -> str:
    zone_id = str(detail.get("id") or "")
    label = str(detail.get("label") or pretty_ref(zone_id))
    clip = detail.get("clip_sim")
    keyframe = safe_url(detail.get("keyframe_url"))
    entities = detail.get("entities") or []
    bbox = detail.get("bbox")

    keyframe_html = (
        f'<div class="crop-frame"><img src="{keyframe}" alt="{esc(label)} keyframe" loading="lazy">'
        f'<span class="crop-corner tl"></span><span class="crop-corner tr"></span>'
        f'<span class="crop-corner bl"></span><span class="crop-corner br"></span>'
        f'<span class="crop-cap">VPR keyframe · {esc(zone_id)}</span></div>'
        if keyframe
        else '<div class="crop-frame"><div class="empty-note">no keyframe captured for this zone</div></div>'
    )

    rows: list[str] = []
    for entity in entities:
        entity_id = str(entity.get("id") or "")
        state = str(entity.get("state") or "NEW")
        color = STATE_COLORS.get(state, "#94a3b8")
        rows.append(
            f'<div class="zone-entity" onclick="Chronicle.node(\'{esc(entity_id)}\')">'
            f'<span class="zone-entity-dot" style="--state-color:{color}"></span>'
            f'<span class="zone-entity-name">{esc(entity.get("label") or pretty_ref(entity_id))}</span>'
            f'<span class="zone-entity-state">{esc(state.replace("_", " "))}</span></div>'
        )
    entity_html = "".join(rows) if rows else '<div class="empty-note">No tracked objects currently map to this zone.</div>'

    bbox_text = "—"
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        bbox_text = ", ".join(f"{float(value):.2f}" for value in bbox[:4])

    return f"""
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Topological anchor · place</div>
    <div class="inspector-title">{esc(label)} {_state_chip("ANCHOR")}</div>
    <div class="inspector-sub">{esc(zone_id)} · visual place recognition</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeDrawer()" aria-label="Close inspector">×</button>
</div>
<div class="inspector-body">
  {keyframe_html}
  <div class="score-grid">
    {_score_card("VPR CLIP", clip, accent="#22d3ee")}
    {_score_card("Zone occupancy", min(len(entities) / 12.0, 1.0), accent="#35d07f")}
  </div>
  <div class="meta-grid">
    {_meta_cell("First seen", fmt_time(detail.get("first_seen")))}
    {_meta_cell("Last seen", fmt_time(detail.get("last_seen")))}
    {_meta_cell("Bounding box", bbox_text)}
    {_meta_cell("Furniture", "yes" if detail.get("is_furniture") else "no")}
  </div>
  <div class="section-head">Objects in zone <span class="count">{len(entities)} nodes</span></div>
  {entity_html}
</div>
""".strip()


def anchor_drawer_html(ref: str, store: MemoryStore, playhead: float) -> str:
    label = store.anchor_labels.get(ref) or pretty_ref(ref)
    edges = [edge for edge in store.edges.values() if edge.source == ref or edge.target == ref]
    return f"""
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Anchor node · spatial memory</div>
    <div class="inspector-title">{esc(label)} {_state_chip("ANCHOR")}</div>
    <div class="inspector-sub">{esc(ref)} · structural landmark</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeDrawer()" aria-label="Close inspector">×</button>
</div>
<div class="inspector-body">
  <div class="empty-note">
    This anchor is not a tracked object. It is a structural node in the dual graph
    used to keep object locations stable across camera pans, occlusion and
    detection gating.
  </div>
  <div class="section-head">Connected relations <span class="count">{len(edges)}</span></div>
  {_edges_html(store, [edge.as_dict() for edge in edges])}
</div>
""".strip()


def user_drawer_html(store: MemoryStore) -> str:
    counts = {}
    for entity in store.entities.values():
        counts[entity.state] = counts.get(entity.state, 0) + 1
    state_rows = "".join(
        f'<div class="zone-entity"><span class="zone-entity-dot" style="--state-color:{STATE_COLORS.get(state, "#94a3b8")}"></span>'
        f'<span class="zone-entity-name">{esc(state.replace("_", " "))}</span>'
        f'<span class="zone-entity-state">{count}</span></div>'
        for state, count in sorted(counts.items())
    )
    return f"""
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Agent node · egocentric observer</div>
    <div class="inspector-title">You {_state_chip("AGENT")}</div>
    <div class="inspector-sub">first-person camera · hand pose source</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeDrawer()" aria-label="Close inspector">×</button>
</div>
<div class="inspector-body">
  <div class="empty-note">
    The USER node is the origin of action edges. Each pick, place and put-down is
    committed against this node with an evidence timestamp.
  </div>
  <div class="section-head">Object state census <span class="count">{len(store.entities)} nodes</span></div>
  {state_rows or '<div class="empty-note">No objects loaded.</div>'}
</div>
""".strip()


def format_log(record: Any) -> str:
    channel = str(getattr(record, "channel", "SYSTEM") or "SYSTEM").upper()
    level = str(getattr(record, "level", "info") or "info").lower()
    marker = "!" if level in {"warn", "warning", "error", "critical"} else " "
    message = str(getattr(record, "message", "") or "")
    t = float(getattr(record, "t", 0.0) or 0.0)
    return f"{t:6.2f} {marker}[{channel:<6}] {message}"


def log_classes(record: Any) -> str:
    channel = str(getattr(record, "channel", "SYSTEM") or "SYSTEM").lower()
    if channel not in {key.lower() for key in CHANNEL_COLORS}:
        channel = "system"
    return f"line-{channel}"
