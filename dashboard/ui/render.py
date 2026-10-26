"""Server-side HTML renderers for the Chronicle dashboard (zine edition)."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from ..contracts import STATE_COLORS, clean_text, pretty_ref
from ..store import ACTION_RELATIONS, MemoryStore

CHANNEL_COLORS = {
    "GEOM": "#9bd9f2",
    "TRACK": "#cfc0f7",
    "STATE": "#ffe08a",
    "VLM": "#f7b8e8",
    "GRAPH": "#a8ecc9",
    "QUERY": "#ffc7d4",
    "SYSTEM": "#cfc7b8",
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


def fmt_t(value: Any) -> str:
    """M/S style chip text such as ``12.4s``."""
    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1f}s"


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
# Chat
# --------------------------------------------------------------------------- #

SUGGESTIONS = (
    "Where are my keys?",
    "Where is my wallet?",
    "When did I last touch my phone?",
    "What did I do in this episode?",
)


def suggestions_html() -> str:
    chips = "".join(
        f'<button class="chipz sug{" first" if index == 0 else ""}" data-query="{esc(query)}" '
        f'onclick="Chronicle.chatSend(this.dataset.query)">{esc(query)}</button>'
        for index, query in enumerate(SUGGESTIONS)
    )
    return chips


# --------------------------------------------------------------------------- #
# Inspector dossier
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


def _score_card(name: str, value: Any, *, accent: str = "#56b8d8") -> str:
    percentage = _pct(value) if value is not None else 0.0
    return f"""
<div class="score-card">
  <div class="score-name"><span>{esc(name)}</span><strong>{_score(value)}</strong></div>
  <div class="score-track"><div class="score-fill" style="width:{percentage:.1f}%;background:{accent}"></div></div>
</div>
""".strip()


def _meta_cell(key: str, value: Any) -> str:
    return f'<div class="meta-cell"><div class="meta-key">{esc(key)}</div><div class="meta-val">{esc(value)}</div></div>'


def _state_history_html(history: list[Mapping[str, Any]], playhead: float) -> str:
    if not history:
        return '<div class="empty-note">no state transitions recorded yet</div>'

    records = sorted(history, key=lambda item: float(item.get("t") or 0.0))
    current_index = -1
    for index, record in enumerate(records):
        if float(record.get("t") or 0.0) <= playhead + 0.05:
            current_index = index
    window_start = max(0, current_index - 10) if current_index >= 0 else max(0, len(records) - 12)
    shown = records[window_start:]
    header = '<div class="empty-note">earlier transitions hidden</div>' if window_start else ""

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
                rendered = f"{value:.2f}" if isinstance(value, float) else str(value)
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
        return '<div class="empty-note">no VLM interaction recorded for this node</div>'

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
        return '<div class="empty-note">no relations committed for this node yet</div>'
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
        else '<div class="crop-frame"><div class="empty-note" style="padding:18px 12px">no crop available<br>for this node yet</div></div>'
    )

    meta = "".join(
        [
            _meta_cell("First seen", fmt_time(first_seen)),
            _meta_cell("Last seen", fmt_time(last_seen)),
            _meta_cell("State since", fmt_time(state_since)),
            _meta_cell("Last touch", fmt_time(last_interaction) if last_interaction is not None else "—"),
            _meta_cell("Track id", track),
            _meta_cell("Zone", zone_label or "—"),
            _meta_cell("Class", cls),
            _meta_cell("Node id", entity_id),
        ]
    )

    return f"""
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Object dossier · live memory</div>
    <div class="inspector-title">{esc(label)} {_state_chip(state)}</div>
    <div class="inspector-sub">{esc(entity_id)} · {esc(cls)} · tracked {esc(track)}</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>
</div>
<div class="inspector-body">
  {crop_html}
  <div class="score-grid">
    {_score_card("CLIP Re-ID", clip, accent=color)}
    {_score_card("Detection conf", conf, accent="#37c585")}
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
        else '<div class="crop-frame"><div class="empty-note" style="padding:18px 12px">no keyframe captured for this zone</div></div>'
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
    <div class="inspector-eyebrow">Place dossier · topological anchor</div>
    <div class="inspector-title">{esc(label)} {_state_chip("ANCHOR")}</div>
    <div class="inspector-sub">{esc(zone_id)} · visual place recognition</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>
</div>
<div class="inspector-body">
  {keyframe_html}
  <div class="score-grid">
    {_score_card("VPR CLIP", clip)}
    {_score_card("Occupancy", min(len(entities) / 12.0, 1.0), accent="#37c585")}
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
    <div class="inspector-eyebrow">Anchor dossier · spatial memory</div>
    <div class="inspector-title">{esc(label)} {_state_chip("ANCHOR")}</div>
    <div class="inspector-sub">{esc(ref)} · structural landmark</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>
</div>
<div class="inspector-body">
  <div class="empty-note">
    This anchor is not a tracked object. It is a structural node in the dual graph
    that keeps object locations stable across camera pans, occlusion and
    detection gating.
  </div>
  <div class="section-head">Connected relations <span class="count">{len(edges)}</span></div>
  {_edges_html(store, [edge.as_dict() for edge in edges])}
</div>
""".strip()


def user_drawer_html(store: MemoryStore) -> str:
    counts: dict[str, int] = {}
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
    <div class="inspector-eyebrow">Agent dossier · egocentric observer</div>
    <div class="inspector-title">You {_state_chip("AGENT")}</div>
    <div class="inspector-sub">first-person camera · hand pose source</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>
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


# --------------------------------------------------------------------------- #
# Telemetry log lines
# --------------------------------------------------------------------------- #


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
