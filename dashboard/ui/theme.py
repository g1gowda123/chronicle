"""Static page chrome for the Chronicle dashboard."""

from __future__ import annotations

import html
from typing import Any

HEAD_HTML = """
<link rel="stylesheet" href="/chronicle/chronicle.css?v=2">
<script defer src="/chronicle/vis-network.min.js?v=1"></script>
<script defer src="/chronicle/chronicle.js?v=2"></script>
<meta name="theme-color" content="#05080d">
<meta name="color-scheme" content="dark">
<style>html,body{background:#05080d;color:#dce8f4}body{overflow:hidden}</style>
""".strip()


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def topbar_html(session: dict[str, Any], snapshot: dict[str, Any]) -> str:
    title = _e(session.get("title") or "Chronicle Episode")
    duration = float(snapshot.get("duration") or 0.0)
    fps = session.get("fps") or snapshot.get("fps") or "—"
    width = session.get("width") or snapshot.get("width") or 1280
    height = session.get("height") or snapshot.get("height") or 720
    mode = str(snapshot.get("mode") or "demo")
    mode_label = "Live ingest" if mode == "live" else "Demo replay"
    provider = _e(snapshot.get("query_provider") or "local-graphrag")
    return f"""
<header class="chron-topbar">
  <div class="brand">
    <span class="brand-mark" aria-hidden="true"></span>
    <div>
      <div class="brand-name">CHRONICLE</div>
      <div class="brand-sub">EPISODIC MEMORY · MISSION CONTROL</div>
    </div>
  </div>
  <div class="episode">
    <div class="episode-title" id="chron-episode">{title}</div>
    <div class="episode-meta" id="chron-meta">{duration:.1f}s · {_e(fps)} FPS · {_e(width)}×{_e(height)}</div>
  </div>
  <div class="topbar-spacer"></div>
  <div class="status-cluster">
    <span class="pill ghost" title="Dashboard query provider">RAG <span id="chron-provider">{provider}</span></span>
    <span class="pill ghost" title="Events per second">RATE <span id="header-rate">0.0</span>/s</span>
    <span class="pill {'live' if mode == 'live' else 'demo'}" id="chron-mode">
      <span class="dot"></span><span id="chron-mode-badge">{mode_label.upper()}</span>
    </span>
    <div class="clock">
      <div class="clock-time" id="chron-clock">--:--:--</div>
      <div class="clock-zone" id="chron-clock-zone">LOCAL</div>
    </div>
  </div>
</header>
""".strip()


def video_head_html() -> str:
    return """
<div class="panel-head">
  <span class="panel-title">Episodic Video</span>
  <span class="panel-kicker">VIEW 01 · EGOCENTRIC RGB</span>
  <div class="right">
    <span class="signal"><i></i>stream synced</span>
  </div>
</div>
""".strip()


VIDEO_HUD_HTML = """
<div class="video-vignette"></div>
<div class="video-hud">
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
  <div class="hud-row bottom">
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


def graph_head_html() -> str:
    return """
<div class="panel-head">
  <span class="panel-title">Dual Graph</span>
  <span class="panel-kicker">2-HOP MEMORY</span>
  <div class="right">
    <span class="signal" id="graph-count">—N / —E</span>
    <div class="mode-toggle" role="group" aria-label="Graph time mode">
      <button class="mode-btn active" data-mode="live" onclick="Chronicle.setGraphMode('live')">Live</button>
      <button class="mode-btn" data-mode="full" onclick="Chronicle.setGraphMode('full')">Episode</button>
    </div>
  </div>
</div>
""".strip()


GRAPH_LEGEND_HTML = """
<div class="graph-legend">
  <span class="legend-item"><span class="legend-dot" style="--legend:#35d07f"></span>static</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#c084fc"></span>held</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#2dd4bf"></span>contained</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#fb923c"></span>out of view</span>
  <span class="legend-item"><span class="legend-dot" style="--legend:#f26d6d"></span>lost</span>
  <span class="legend-item"><span class="legend-line" style="--legend:#f5b942"></span>action</span>
  <span class="legend-item"><span class="legend-line" style="--legend:#2dd4bf"></span>spatial</span>
</div>
""".strip()


GRAPH_FOOT_HTML = """
<div class="graph-foot">
  <div class="stat"><div class="stat-label">Objects</div><div class="stat-value" id="stat-objects">—</div></div>
  <div class="stat"><div class="stat-label">Active</div><div class="stat-value accent" id="stat-active">—</div></div>
  <div class="stat"><div class="stat-label">Edges</div><div class="stat-value" id="stat-edges">—</div></div>
  <div class="stat"><div class="stat-label">VLM calls</div><div class="stat-value" id="stat-vlm">—</div></div>
  <div class="stat"><div class="stat-label">FPS</div><div class="stat-value" id="stat-fps">—</div></div>
  <div class="stat"><div class="stat-label">Latency</div><div class="stat-value" id="stat-latency">—</div></div>
</div>
""".strip()


TELEMETRY_HEAD_HTML = """
<div class="panel-head">
  <span class="panel-title">Telemetry Stream</span>
  <span class="panel-kicker">TRACE BUS</span>
  <div class="right">
    <div class="telemetry-legend">
      <span><i style="--legend:#38bdf8"></i>GEOM</span>
      <span><i style="--legend:#a78bfa"></i>TRACK</span>
      <span><i style="--legend:#fbbf24"></i>STATE</span>
      <span><i style="--legend:#e879f9"></i>VLM</span>
      <span><i style="--legend:#34d399"></i>GRAPH</span>
    </div>
    <span class="telemetry-follow"><i></i>AUTO</span>
  </div>
</div>
""".strip()


QUERY_HEAD_HTML = """
<div class="panel-head">
  <span class="panel-title">Episodic Query</span>
  <span class="panel-kicker">GRAPHRAG · NATURAL LANGUAGE</span>
  <div class="right">
    <span class="signal" id="query-provider">local-graphrag</span>
  </div>
</div>
""".strip()


EMPTY_DRAWER_HTML = """
<div class="inspector-head">
  <div class="inspector-head-main">
    <div class="inspector-eyebrow">Node inspector</div>
    <div class="inspector-title">No node selected</div>
    <div class="inspector-sub">Click any entity, zone or anchor in the graph.</div>
  </div>
  <button class="inspector-close" onclick="Chronicle.closeDrawer()" aria-label="Close inspector">×</button>
</div>
<div class="inspector-body">
  <div class="empty-note">
    Select a memory node to inspect its object crop, state history, CLIP similarity,
    spatial anchors and VLM interaction strips. Evidence timestamps are clickable.
  </div>
</div>
""".strip()
