"""Static page chrome for the Chronicle dashboard — zine edition.

Warm "field notebook" presentation: paper cards, ink borders, washi tape,
highlighter chips and hand-written notes, driven by the same server sync
runtime as before (:meth:`dashboard.ui.page.build_dashboard`).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _asset_version(filename: str) -> str:
    """Return a cache-buster that changes whenever a static asset changes.

    ``/chronicle`` files are served with ``cache-control: max-age=3600``, so a
    hard-coded ``?v=`` tag goes stale as soon as the asset is edited. Deriving
    the tag from the file's mtime keeps browsers from running old JavaScript.
    """
    try:
        return str((_STATIC_DIR / filename).stat().st_mtime_ns)
    except OSError:
        return "1"


_HEAD_TEMPLATE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&family=Caveat:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap">
<link rel="stylesheet" href="/chronicle/chronicle.css?v=__CSS_VERSION__">
<script defer src="/chronicle/vis-network.min.js?v=1"></script>
<script defer src="/chronicle/chronicle.js?v=__JS_VERSION__"></script>
<meta name="theme-color" content="#faf3e3">
<meta name="color-scheme" content="light">
<style>html,body{background:#faf3e3;color:#26221a}</style>
"""

HEAD_HTML = (
    _HEAD_TEMPLATE.strip()
    .replace("__CSS_VERSION__", _asset_version("chronicle.css"))
    .replace("__JS_VERSION__", _asset_version("chronicle.js"))
)

EYE_LOGO = """
<svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
  <path d="M4 24 C 12 10, 36 10, 44 24 C 36 38, 12 38, 4 24 Z"
        fill="#fffcf4" stroke="#26221a" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="24" cy="24" r="8.5" fill="#ffd34d" stroke="#26221a" stroke-width="3"/>
  <circle cx="24" cy="24" r="3.4" fill="#26221a"/>
  <circle cx="27" cy="21" r="1.4" fill="#fffcf4"/>
</svg>
""".strip()

STAR_DOODLE = """
<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" class="floaty">
  <path d="M12 2 L14 9 L21 9.5 L15.5 14 L17.5 21 L12 17 L6.5 21 L8.5 14 L3 9.5 L10 9 Z"
        stroke="#ffd34d" stroke-width="1.8" fill="#ffd34d55" stroke-linejoin="round"/>
</svg>
""".strip()

DOODLE_ARROW = """
<svg viewBox="0 0 60 34" fill="none" aria-hidden="true" class="floaty">
  <path d="M4 6 C 22 4, 42 10, 50 24" stroke="#ff6b4a" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M50 24 L 41 21 M50 24 L 49 14" stroke="#ff6b4a" stroke-width="2.5" stroke-linecap="round"/>
</svg>
""".strip()


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


# --------------------------------------------------------------------------- #
# Topbar
# --------------------------------------------------------------------------- #


def topbar_html(session: dict[str, Any], snapshot: dict[str, Any]) -> str:
    title = _e(session.get("title") or "Chronicle Episode")
    duration = float(snapshot.get("duration") or 0.0)
    fps = session.get("fps") or snapshot.get("fps") or "—"
    width = session.get("width") or snapshot.get("width") or 1280
    height = session.get("height") or snapshot.get("height") or 720
    mode = str(snapshot.get("mode") or "demo")
    provider = _e(snapshot.get("query_provider") or "local-graphrag")
    pill_cls = "is-live" if mode == "live" else "is-demo"
    pill_label = "live ingest" if mode == "live" else "demo replay"
    return f"""
<header class="z-topbar">
  <div class="brand">
    <span class="brand-fig floaty">{EYE_LOGO}</span>
    <div>
      <div class="brand-name">Chronicle<span class="dotz">.</span></div>
      <div class="brand-sub">a head-cam that remembers</div>
    </div>
  </div>
  <div class="metric-row">
    <span class="metric" title="frames ingested"><span class="metric-val" id="metric-frames">—</span><span class="metric-label">frames</span></span>
    <span class="metric" title="pipeline fps"><span class="metric-val" id="metric-fps">—</span><span class="metric-label">fps</span></span>
    <span class="metric" title="objects in memory"><span class="metric-val accent" id="metric-entities">—</span><span class="metric-label">entities</span></span>
    <span class="metric" title="dual-graph edges"><span class="metric-val" id="metric-edges">—</span><span class="metric-label">edges</span></span>
    <span class="metric" title="VLM actions verified"><span class="metric-val" id="metric-actions">—</span><span class="metric-label">actions</span></span>
  </div>
  <div class="topbar-spacer"></div>
  <span class="chipz chipz-secondary" title="episode · query provider"><span id="chron-episode-title">{title}</span> · <span id="chron-provider-chip">{provider}</span></span>
  <span class="chipz chipz-lg status-pill {pill_cls}" id="chron-mode">
    <span class="pill-dot"></span><span id="chron-mode-badge">{_e(pill_label.upper())}</span>
  </span>
  <div class="clock">
    <div class="clock-time" id="chron-clock">--:--:--</div>
    <div class="clock-zone" id="chron-clock-zone">LOCAL</div>
  </div>
  <div class="chipz chipz-secondary" title="resolution" id="chron-meta">{_e(f"{duration:.1f}s · {fps} FPS · {width}×{height}")}</div>
</header>
""".strip()


# --------------------------------------------------------------------------- #
# Video stage
# --------------------------------------------------------------------------- #

VIDEO_HUD_HTML = """
<div class="hud">
  <div class="hud-row">
    <div class="hud-stack">
      <span class="hud-chip"><span class="rec-dot"></span>REC <strong id="hud-mode">DEMO REPLAY</strong></span>
      <span class="hud-chip"><span id="hud-camera">CAM · —</span></span>
    </div>
    <div class="hud-stack right">
      <span class="hud-chip timecode" id="hud-time">TC 00:00.00</span>
      <span class="hud-chip" id="hud-zone">ZONE · —</span>
    </div>
  </div>
  <div class="hud-row" style="align-items:flex-end">
    <div class="hud-stack">
      <span class="hud-chip"><span class="hud-key">EGO POSE</span><span id="hud-tracks">TRACKS · 0/0</span></span>
    </div>
    <div class="hud-stack right">
      <span class="hud-chip"><span id="hud-fps">— FPS</span><span class="hud-key">·</span><span id="hud-latency">PIPE · —</span></span>
    </div>
  </div>
  <div class="hud-caption hidden" id="hud-caption">
    <div class="cap-tag" id="hud-caption-tag">VLM · STANDBY</div>
    <div class="cap-text" id="hud-caption-text"></div>
  </div>
</div>
""".strip()


def clip_menu_html(options: list[dict[str, str]], active: str = "") -> str:
    """Clip switcher dropdown (recorded replay timelines)."""
    selected = active or (options[0].get("value") if options else "")
    items = "".join(
        f'<option value="{_e(option.get("value"))}"'
        f'{" selected" if option.get("value") == selected else ""}>'
        f'{_e(option.get("title"))}</option>'
        for option in options
    )
    return f"""
<label class="clip-switch" title="Switch the recorded clip replayed on the stage">
  <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <rect x="2" y="3.5" width="12" height="9" rx="1.8" stroke="currentColor" stroke-width="1.7"/>
    <path d="M6.6 6.3 L10.4 8 L6.6 9.7 Z" fill="currentColor"/>
  </svg>
  <span>clip</span>
  <select id="chron-clip" onchange="Chronicle.setClip(this.value)" aria-label="Choose clip">
    {items}
  </select>
</label>
""".strip()


def video_html(snapshot: dict[str, Any], replays: list[dict[str, str]] | None = None,
               active_replay: str = "") -> str:
    """The big stage card: video, boxes, HUD, transport, scrub and toolbar."""
    src = snapshot.get("video_url") or ""
    video = (
        f'<video class="chron-video" src="{_e(src)}" muted playsinline loop preload="auto"></video>'
        if src
        else '<div class="empty-note" style="color:#f7ead2;padding:24px">no feed attached yet</div>'
    )
    hint = (
        f"""
<div class="empty-stage" id="stage-hint">
  <div style="position:relative">
    {DOODLE_ARROW}
    <div class="font-hand-note">waiting for the first frame — POST /api/ingest or run the demo</div>
  </div>
</div>
"""
        if not src
        else ""
    )
    options = replays or []
    switcher = clip_menu_html(options, active_replay) if options else (
        '<span class="chipz chipz-mint"><span class="dot"></span>stream synced</span>'
    )
    return f"""
<div class="zcard card-pad video-card rot-l">
  <div class="tape tape-sky" style="width:110px"></div>
  <div class="card-head">
    <h2 class="card-title"><span id="chron-video-title">{_e(snapshot.get("title") or "Chronicle Episode")}</span>
      <span class="card-note">the head-cam feed</span></h2>
    <div class="head-right">{switcher}</div>
  </div>
  <div class="video-stage">
    <div class="video-shell" id="chron-shell" data-camera="STANDBY">
      {video}
    </div>
    {hint}
    {VIDEO_HUD_HTML}
  </div>
  <div class="transport">
    <button class="t-btn play" id="chron-play" onclick="Chronicle.togglePlay()" title="Play / pause (Space)"><span id="chron-play-icon">▶</span></button>
    <button class="t-btn restart" onclick="Chronicle.restart()" title="Replay from 0:00 (R)">↺</button>
    <button class="t-btn" id="chron-shot" onclick="Chronicle.screenshot()" title="Save the current frame as PNG (S)" aria-label="Screenshot current frame">⤓</button>
    <button class="t-btn" id="chron-full" onclick="Chronicle.fullscreen()" title="Fullscreen video (F)" aria-label="Toggle fullscreen video">⛶</button>
    <span class="t-time" id="chron-time">00:00.00 <span class="total">/ 00:00.00</span></span>
    <span class="t-sep"></span>
    <span class="transport-label" id="transport-label">SYNCED PLAYBACK · LOOP</span>
    <span class="t-loop" id="t-loop">{_e(snapshot.get("source_label") or "")}</span>
    <label class="btnz btnz-ghost clip-upload" title="Upload a clip to play in the stage">
      <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
        <path d="M8 11 V2.5 M4.5 6 L8 2.5 L11.5 6 M3 13.5 H13" stroke="currentColor"
              stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      upload
      <input type="file" id="chron-upload" accept="video/*" hidden
             onchange="Chronicle.uploadClip(this)">
    </label>
    <button class="btnz btnz-ghost" onclick="Chronicle.eventClips(this)"
            title="Cut the active clip into short event windows (3 s pre/post) and snapshot a graph for each"
            aria-label="Cut the active clip into event clips">✂ events</button>
    <button class="btnz btnz-ghost" onclick="Chronicle.exportGraphs()"
            title="Download every graph/timeline on this machine as one ZIP"
            aria-label="Export graphs as one ZIP">⤒ export</button>
    <label class="btnz btnz-ghost" title="Import a Chronicle graph ZIP (no re-ingest)">
      ⤓ import
      <input type="file" id="chron-import" accept=".zip,application/zip" hidden
             onchange="Chronicle.importGraphs(this)">
    </label>
  </div>
  <div class="chron-timeline">
    <div class="tl-track">
      <div class="tl-fill" id="chron-fill" style="width:0%"></div>
      <div class="tl-head" id="chron-head" style="left:0%"></div>
      <input class="chron-scrub" id="chron-scrub" type="range" min="0" max="0.01" step="0.01" value="0" aria-label="Scrub episode timeline">
    </div>
    <div class="tl-labels"><span>0s</span><span>—</span><span>—</span><span>—</span><span>—</span><span>—</span></div>
  </div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #

CHAT_GREETING = """
<div class="msg-row" id="chat-greet">
  <span class="ava">{eye}</span>
  <div>
    <div class="bub">Hey! I'm Chronicle — point a camera at life and I'll remember it.
Play the feed, watch me take notes, then <span class="hl">quiz me about anything in here</span>.</div>
    <div class="badges-row"><span class="chipz chipz-mint" style="font-size:.6rem;padding:.1rem .5rem">✓ grounded</span></div>
    <div class="msg-meta">chronicle · hello</div>
  </div>
</div>
""".strip().format(eye=EYE_LOGO)


def chat_head_html() -> str:
    return f"""
<div class="zcard card-pad chat-card rot-r">
  <div class="tape" style="width:84px"></div>
  <div class="card-head">
    <h2 class="card-title">ask my memory
      <span class="card-note">(grounded or I say so)</span></h2>
    <div class="head-right"><span class="chipz chipz-secondary" id="chat-status" style="font-size:.6rem">warming up…</span></div>
  </div>
  <div class="chat-log nicescroll" id="chron-chat-log">{CHAT_GREETING}</div>
  <div class="chat-sugs" id="chron-suggestions"></div>
  <div class="chat-input-row">
    <div class="chron-query-input">
      <input id="chron-chat-input" type="text" placeholder="e.g. where was the cup before the table?"
             aria-label="Ask Chronicle a question"
             onkeydown="if(event.key==='Enter'){{Chronicle.chatSendFromInput();}}">
    </div>
    <button class="btnz btnz-ink" onclick="Chronicle.chatSendFromInput()" aria-label="Send question">
      <svg viewBox="0 0 16 16" width="15" height="15" fill="none" aria-hidden="true">
        <path d="M2 8 L14 2 L9.5 14 L7.5 9 Z" stroke="currentColor" stroke-width="1.8"
              stroke-linejoin="round" fill="currentColor" fill-opacity="0.25"/>
      </svg>
      ask
    </button>
  </div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def graph_head_html() -> str:
    return """
<div class="card-head">
  <h2 class="card-title">memory graph
    <span class="card-note">zones hold the objects</span></h2>
  <div class="head-right">
    <span class="chipz chipz-secondary" id="graph-count" style="font-size:.6rem">—N / —E</span>
    <div class="mode-toggle" role="group" aria-label="Graph time mode">
      <button class="mode-btn active" data-mode="live" onclick="Chronicle.setGraphMode('live')">Live</button>
      <button class="mode-btn" data-mode="full" onclick="Chronicle.setGraphMode('full')">Episode</button>
    </div>
  </div>
</div>
""".strip()


GRAPH_LEGEND_HTML = """
<div class="graph-legend">
  <span class="legend-item sq"><span class="legend-dot" style="--legend:#26221a"></span>zone</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#37c585"></span>static</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#b08bd9"></span>held</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#56d0c0"></span>contained</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#ff9eb5"></span>out of view</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#ff6b4a"></span>lost</span>
  <span class="legend-item"><span class="legend-line" style="--legend:#ffd34d"></span>action</span>
  <span class="legend-item"><span class="legend-line" style="--legend:#56b8d8"></span>spatial</span>
</div>
""".strip()


def graph_body_html() -> str:
    return f"""
<div class="zcard card-pad graph-card rot-r">
  <div class="tape tape-mint" style="width:78px"></div>
  {graph_head_html()}
  <div class="graph-body"><div id="chron-graph"></div></div>
  {GRAPH_LEGEND_HTML}
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Inspector (rail dossier)
# --------------------------------------------------------------------------- #

EMPTY_INSPECTOR_HTML = """
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Entity dossier</div>
    <div class="inspector-title">no file open</div>
    <div class="inspector-sub">click any object in the list or the graph</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>
</div>
<div class="inspector-body">
  <div class="empty-strip">
    <span class="empty-note" style="text-align:center;padding:0 14px">
      pick an entity — its crop, state history, clips and relations open here.
    </span>
  </div>
</div>
""".strip()


def inspector_shell_html(content: str = "") -> str:
    inner = content.strip() or EMPTY_INSPECTOR_HTML
    return f"""
<div class="zcard card-pad inspector-card">
  <div class="tape" style="width:74px"></div>
  <div id="chron-inspector-content" style="margin:-2px">{inner}</div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Entity list
# --------------------------------------------------------------------------- #


def entity_card_html() -> str:
    return """
<div class="zcard card-pad rot-l">
  <div class="tape tape-pink" style="width:88px"></div>
  <div class="card-head">
    <h2 class="card-title">entity list
      <span class="card-note">click one · poke my memory</span></h2>
    <div class="head-right"><span class="chipz chipz-secondary" id="ent-count" style="font-size:.6rem">0 objects</span></div>
  </div>
  <input class="ent-filter" id="chron-ent-filter" type="text"
         placeholder="filter by label, zone…" aria-label="Filter entities"
         oninput="Chronicle.filterEntities(this.value)">
  <div class="ent-list nicescroll" id="chron-ent-list"></div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Bottom strip cards
# --------------------------------------------------------------------------- #


def events_card_html() -> str:
    return """
<div class="zcard card-pad rot-l">
  <div class="card-head">
    <h2 class="card-title">
      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M2 8 H14 M4 5 L2 8 L4 11 M12 5 L14 8 L12 11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      timeline of state changes
      <span class="card-note">what has happened up to the playhead</span>
    </h2>
    <div class="head-right"><span class="chipz chipz-secondary" id="ev-count" style="font-size:.6rem">0 events so far</span></div>
  </div>
  <div style="overflow-x:auto" class="nicescroll">
    <div class="events-strip" id="chron-events">
      <div class="empty-strip" style="width:100%"><span class="empty-note">events land here once I start noticing things…</span></div>
    </div>
  </div>
</div>
""".strip()


def actions_card_html() -> str:
    return """
<div class="zcard card-pad rot-r2" style="position:relative">
  <div class="tape tape-mint" style="width:80px"></div>
  <div class="card-head">
    <h2 class="card-title">actions & interactions
      <span class="card-note">hands were involved</span></h2>
  </div>
  <div class="actions-list" id="chron-actions">
    <div class="empty-strip"><span class="empty-note">no interactions at this point in the replay</span></div>
  </div>
</div>
""".strip()


def benchmark_card_html() -> str:
    """Static 15-minute benchmark card (Chronicle vs an assumed Gemini baseline).

    Gemini latency is deliberately labelled as assumed: it is a stand-in for a
    measured baseline, not a number produced on this machine.
    """
    return """
<div class="zcard card-pad rot-l" id="chron-benchmark">
  <div class="card-head">
    <h2 class="card-title">benchmark · 15-min video</h2>
    <div class="head-right">
      <span class="chipz chipz-mustard" style="font-size:.58rem">Gemini 2.8 s*</span>
    </div>
  </div>

  <div style="font-family:var(--hand);font-size:1rem;color:var(--inksoft);margin:-2px 0 8px">
    compile once vs re-watch per question
  </div>

  <div style="display:flex;align-items:baseline;gap:10px;margin:2px 0 10px">
    <div style="font-family:var(--display);font-size:2.15rem;font-weight:800;line-height:1">≈3,100×</div>
    <div style="font-size:.66rem;line-height:1.3;color:var(--inksoft)">
      faster per question<br>
      <b style="color:var(--ink)">0.89 ms</b> cold vs <b style="color:var(--ink)">2.8 s</b> assumed
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:9px">
    <div style="border:1.5px solid var(--ink);background:#eaf6ef;padding:7px 8px">
      <div style="font-size:.54rem;letter-spacing:.08em;font-weight:800;color:#1b7a4d">CHRONICLE VFM</div>
      <div style="font-size:1.05rem;font-weight:800;line-height:1.1;margin-top:2px">0.89 ms</div>
      <div style="font-size:.58rem;color:var(--inksoft)">cold · 3 ms cached</div>
    </div>
    <div style="border:1.5px solid var(--ink);background:#f2eee4;padding:7px 8px">
      <div style="font-size:.54rem;letter-spacing:.08em;font-weight:800;color:var(--inksoft)">GEMINI*</div>
      <div style="font-size:1.05rem;font-weight:800;line-height:1.1;margin-top:2px">2.8 s</div>
      <div style="font-size:.58rem;color:var(--inksoft)">per question</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1.15fr .9fr .9fr;gap:3px 7px;font-size:.6rem;line-height:1.35;color:var(--inksoft)">
    <span></span>
    <span style="font-weight:800;letter-spacing:.05em;color:var(--ink)">CHRONICLE</span>
    <span style="font-weight:800;letter-spacing:.05em;color:var(--ink)">GEMINI*</span>

    <span>video at query</span><b style="color:var(--ink)">no</b><b style="color:var(--ink)">yes</b>
    <span>memory / input</span><b style="color:var(--ink)">95 KB</b><b style="color:var(--ink)">249 MB</b>
    <span>answer path</span><b style="color:var(--ink)">deterministic</b><b style="color:var(--ink)">video model</b>
    <span>evidence</span><b style="color:var(--ink)">citation + frame</b><b style="color:var(--ink)">text</b>
    <span>100 questions</span><b style="color:var(--ink)">0.09 s</b><b style="color:var(--ink)">4.7 min</b>
  </div>

  <div style="margin-top:8px;padding-top:6px;border-top:1px dashed #d8cdb7;font-size:.55rem;line-height:1.35;color:var(--inksoft)">
    *Gemini 2.8 s is an illustrative baseline, not measured on this machine.
    Chronicle: 15-min walk, 0.89 ms cold / 3 ms cached; one-time compile ≈5 min.
  </div>
</div>
""".strip()


def vitals_card_html() -> str:
    rows = (
        "objects|stat-objects",
        "active|stat-active",
        "edges|stat-edges",
        "vlm calls|stat-vlm",
        "fps|stat-fps",
        "latency|stat-latency",
    )
    body = "".join(
        f'<div class="vital-row"><span class="vital-name">{esc_name}</span>'
        f'<span class="vital-val" id="{ident}">—</span></div>'
        for name_pair in rows
        for esc_name, ident in [name_pair.split("|", 1)]
    )
    return f"""
<div class="zcard card-pad rot-r" style="position:relative">
  <div class="tape" style="width:70px"></div>
  <div class="card-head">
    <h2 class="card-title">pipeline vitals</h2>
  </div>
  <div class="vitals-rows">{body}</div>
</div>
""".strip()


def telemetry_head_html() -> str:
    legend = "".join(
        f'<span><i style="--legend:{color}"></i>{name}</span>'
        for name, color in (
            ("GEOM", "#9bd9f2"),
            ("TRACK", "#cfc0f7"),
            ("STATE", "#ffe08a"),
            ("VLM", "#f7b8e8"),
            ("GRAPH", "#a8ecc9"),
            ("QUERY", "#ffc7d4"),
        )
    )
    return f"""
<div class="card-head">
  <h2 class="card-title">pipeline log
    <span class="card-note">raw thoughts, unfiltered</span></h2>
  <div class="head-right">
    <div class="telemetry-legend">{legend}</div>
    <span class="chipz chipz-secondary" style="font-size:.6rem">AUTO ↓</span>
  </div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Marquee + footer
# --------------------------------------------------------------------------- #


def marquee_html() -> str:
    tagline = (
        "✦ chronicle remembers objects ✦ colours ✦ places ✦ states ✦ actions "
        "✦ ask it anything — grounded answers only ✦ no hallucinating, pinky promise ✦ "
    )
    chunk = "".join(f"<span>{tagline}</span>" for _ in range(4))
    return f"""
<div class="zcard marquee-card rot-l" aria-hidden="true" style="transform:rotate(-0.5deg)">
  <div class="marquee-band">
    <div class="marquee-inner" id="chron-marquee">{chunk}{chunk}</div>
  </div>
</div>
""".strip()


def footer_html() -> str:
    return f"""
<footer class="z-footer">
  <p class="foot-note">built with too much coffee · everything runs local · <span class="hl hl-mint" id="foot-source">demo replay</span></p>
  <div class="foot-right">
    {STAR_DOODLE}
    <span class="chipz" style="font-size:.62rem">egocentric episodic memory · dual graph + graphrag</span>
  </div>
</footer>
""".strip()
