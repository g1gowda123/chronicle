"""Chronicle Role D app: NiceGUI dashboard + JSON ingest/query API.

Run with ``python run_dashboard.py`` and open http://localhost:8081.

Environment
-----------
CHRONICLE_SOURCE   ``demo`` (default) or ``live``
CHRONICLE_DATA_DIR directory holding demo assets (default ``./data``)
CHRONICLE_VIDEO    override the video URL (e.g. ``/media/session.webm``)
CHRONICLE_EVENTS_JSONL  tail a producer's JSONL event file
CHRONICLE_DB       poll a producer's SQLite database
CHRONICLE_QUERY_URL  optional external GraphRAG/LLM endpoint
CHRONICLE_LLM_BACKEND  auto | ollama | template | none (default auto)
CHRONICLE_LLM_MODEL    Ollama model (default qwen2.5:7b-instruct)
CHRONICLE_OLLAMA_URL   Ollama base URL (default http://127.0.0.1:11434)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from nicegui import app, ui

from . import memory
from .runtime import clip_title, default_data_dir, probe_duration, runtime
from .ui.page import build_dashboard

DATA_DIR = default_data_dir()
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.add_static_files("/chronicle", STATIC_DIR)
app.add_media_files("/media", DATA_DIR)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


@ui.page("/", title="Chronicle · Mission Control")
def index() -> None:
    build_dashboard()


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #


def _arg(request: Request, name: str, default: float | None = None) -> float | None:
    raw = request.query_params.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _t(request: Request) -> float:
    value = _arg(request, "t")
    return runtime.playhead if value is None else max(0.0, value)


def _mode(request: Request) -> str:
    return "full" if request.query_params.get("mode", "live").lower() == "full" else "live"


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


@app.post("/api/ingest")
async def ingest(request: Request) -> JSONResponse:
    """Accept one event, an array of events, ``{"events": [...]}`` or NDJSON."""
    content_type = (request.headers.get("content-type") or "").lower()
    payload: Any
    if "ndjson" in content_type:
        text = (await request.body()).decode("utf-8", "replace")
        payload = [line for line in text.splitlines() if line.strip()]
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        payload = payload["events"]
    if isinstance(payload, list):
        raws = payload
    elif isinstance(payload, dict) and payload:
        raws = [payload]
    else:
        return JSONResponse({"accepted": 0, "rejected": 1, "error": "empty payload"}, status_code=400)

    accepted = runtime.ingest_many(raws, origin="http")
    rejected = len(raws) - accepted
    return JSONResponse(
        {
            "accepted": accepted,
            "rejected": rejected,
            "received": len(raws),
            "mode": runtime.mode,
            "live_events": runtime.live_events,
            "playhead": round(runtime.playhead, 3),
        }
    )


@app.post("/api/upload")
async def upload(request: Request) -> JSONResponse:
    """Store an uploaded clip under ``data/uploads`` and return its URL."""
    raw_name = Path(str(request.query_params.get("name") or "clip.mp4")).name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".mkv"}:
        return JSONResponse({"error": "unsupported video type"}, status_code=400)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(raw_name).stem).strip("-") or "clip"
    uploads = DATA_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / f"{stem}{suffix}"
    counter = 1
    while target.exists():
        counter += 1
        target = uploads / f"{stem}-{counter}{suffix}"
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    if len(body) < 1024 or body[:64].startswith(b"version https://git-lfs"):
        return JSONResponse({"error": "not a video file (Git-LFS placeholder?)"}, status_code=400)
    if len(body) > 512 * 1024 * 1024:
        return JSONResponse({"error": "file too large"}, status_code=413)
    target.write_bytes(body)
    rel = target.relative_to(DATA_DIR).as_posix()
    return JSONResponse(
        {
            "path": rel,
            "video_url": "/media/" + rel,
            "title": clip_title(target),
            "duration": probe_duration(target),
            "bytes": len(body),
        }
    )


# --------------------------------------------------------------------------- #
# Video-free memory (VFM)
# --------------------------------------------------------------------------- #


def _vfm_clip(request: Request) -> str:
    return str(request.query_params.get("clip") or "")


def _vfm_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@app.get("/api/vfm/status")
async def vfm_status(request: Request) -> JSONResponse:
    try:
        return JSONResponse(memory.status(_vfm_clip(request), DATA_DIR))
    except Exception as error:  # keep the dashboard API alive
        return _vfm_error(f"status failed: {error}", 500)


@app.post("/api/vfm/compile")
async def vfm_compile(request: Request) -> JSONResponse:
    clip = _vfm_clip(request)
    source = memory.resolve(clip, DATA_DIR)
    if source is None:
        return _vfm_error("unknown clip")
    if source.db is None or not Path(source.db).is_file():
        return _vfm_error("this clip has no graph DB yet; ingest it first")
    manager = memory.manager(DATA_DIR)
    force = str(request.query_params.get("force") or "").lower() in {"1", "true", "yes"}
    manager.submit(source, source.video, force=force)
    return JSONResponse(memory.status(clip, DATA_DIR))


@app.get("/api/vfm/report")
async def vfm_report(request: Request) -> Any:
    source = memory.resolve(_vfm_clip(request), DATA_DIR)
    if source is None:
        return _vfm_error("unknown clip")
    artifact = memory.manager(DATA_DIR).artifact(source, "memory.html")
    if artifact is None:
        return _vfm_error("memory report is not compiled yet", 404)
    return FileResponse(artifact, media_type="text/html")


@app.get("/api/vfm/download")
async def vfm_download(request: Request) -> Any:
    source = memory.resolve(_vfm_clip(request), DATA_DIR)
    if source is None:
        return _vfm_error("unknown clip")
    name = str(request.query_params.get("name") or "")
    artifact = memory.manager(DATA_DIR).artifact(source, name)
    if artifact is None:
        return _vfm_error("artifact not found", 404)
    return FileResponse(artifact, filename=artifact.name)


@app.post("/api/vfm/ask")
async def vfm_ask(request: Request) -> JSONResponse:
    source = memory.resolve(_vfm_clip(request), DATA_DIR)
    if source is None:
        return _vfm_error("unknown clip")
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    question = str((payload or {}).get("question") or "").strip()
    if not question:
        return _vfm_error("question is required")
    use_qwen = bool((payload or {}).get("qwen"))
    try:
        answer = memory.manager(DATA_DIR).ask(source, question, use_qwen=use_qwen)
    except RuntimeError as error:
        return _vfm_error(str(error), 409)
    return JSONResponse(answer.to_dict())


# --------------------------------------------------------------------------- #
# Read models
# --------------------------------------------------------------------------- #


@app.get("/api/health")
async def health() -> JSONResponse:
    stats = runtime.stats()
    return JSONResponse(
        {
            "ok": True,
            "mode": runtime.mode,
            "demo_available": runtime.demo_available,
            "duration": round(runtime.duration, 3),
            "playhead": round(runtime.playhead, 3),
            "playing": runtime.playing,
            "entities": stats.get("entities"),
            "edges": stats.get("edges"),
            "rejected_events": runtime.rejected_events,
            "uptime_s": stats.get("uptime_s"),
        }
    )


@app.get("/api/state")
async def state(request: Request) -> JSONResponse:
    return JSONResponse(runtime.snapshot(mode=_mode(request)))


@app.get("/api/meta")
async def meta() -> JSONResponse:
    return JSONResponse(
        {
            "session": runtime.store.session,
            "mode": runtime.mode,
            "source_label": runtime.source_label,
            "video_url": runtime.video_url,
            "duration": runtime.duration,
            "fps": runtime.store.session.get("fps"),
            "width": runtime.store.session.get("width"),
            "height": runtime.store.session.get("height"),
        }
    )


@app.get("/api/graph")
async def graph(request: Request) -> JSONResponse:
    return JSONResponse(runtime.store.graph_snapshot(_t(request), mode=_mode(request)))


@app.get("/api/boxes")
async def boxes(request: Request) -> JSONResponse:
    return JSONResponse({"t": _t(request), "boxes": runtime.store.boxes_at(_t(request))})


@app.get("/api/entity/{entity_id}")
async def entity(entity_id: str, request: Request) -> JSONResponse:
    detail = runtime.store.entity_detail(entity_id, _t(request))
    if detail is None:
        return JSONResponse({"error": f"unknown entity {entity_id}"}, status_code=404)
    return JSONResponse(detail)


@app.get("/api/zone/{zone_id}")
async def zone(zone_id: str) -> JSONResponse:
    detail = runtime.store.zone_detail(zone_id)
    if detail is None:
        return JSONResponse({"error": f"unknown zone {zone_id}"}, status_code=404)
    return JSONResponse(detail)


@app.get("/api/timeline")
async def timeline(request: Request) -> JSONResponse:
    include_future = request.query_params.get("future", "1") not in {"0", "false", "no"}
    return JSONResponse({"markers": runtime.timeline(include_future=include_future)})


@app.get("/api/log")
async def log(request: Request) -> JSONResponse:
    since = int(_arg(request, "since", 0.0) or 0)
    limit_raw = _arg(request, "limit", 200.0) or 200.0
    limit = max(1, min(int(limit_raw), 1000))
    records = runtime.store.log_since(since, limit=limit)
    return JSONResponse(
        {
            "since": since,
            "next": records[-1].seq if records else since,
            "logs": [record.as_dict() for record in records],
        }
    )


@app.get("/api/stats")
async def stats() -> JSONResponse:
    return JSONResponse(runtime.stats())


# --------------------------------------------------------------------------- #
# Query / control
# --------------------------------------------------------------------------- #


@app.post("/api/query")
async def query(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    text = str(body.get("query") or body.get("q") or "").strip()
    if not text:
        return JSONResponse({"error": "missing query"}, status_code=400)
    t_raw = body.get("t")
    try:
        t = runtime.playhead if t_raw is None else float(t_raw)
    except (TypeError, ValueError):
        t = runtime.playhead
    answer = await runtime.query.answer_async(text, max(0.0, min(t, runtime.duration or t)))
    return JSONResponse(answer)


@app.post("/api/control")
async def control(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    action = str(body.get("action") or "").lower()
    if action == "seek":
        runtime.seek(float(body.get("t") or 0.0), playing=body.get("playing"))
    elif action == "play":
        runtime.set_playing(True)
    elif action == "pause":
        runtime.set_playing(False)
    elif action == "toggle":
        runtime.toggle_playing()
    elif action == "restart":
        runtime.restart_demo(float(body.get("t") or 0.0))
    elif action == "rate":
        runtime.set_rate(float(body.get("rate") or 1.0))
    elif action == "source":
        source = str(body.get("source") or "demo").lower()
        if source == "demo":
            runtime.restart_demo(float(body.get("t") or 0.0))
        elif source == "live":
            runtime._enter_live()  # noqa: SLF001 - explicit operator action
    else:
        return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "mode": runtime.mode,
            "playhead": round(runtime.playhead, 3),
            "playing": runtime.playing,
        }
    )


# --------------------------------------------------------------------------- #
# Lifecycle / runner
# --------------------------------------------------------------------------- #


app.on_startup(runtime.start)
app.on_shutdown(runtime.stop)


def run(host: str = "0.0.0.0", port: int = 8081, *, show: bool = False) -> None:
    """Start the NiceGUI server (blocking)."""
    ui.run(
        host=host,
        port=port,
        title="Chronicle · Mission Control",
        favicon="🌌",
        reload=False,
        show=show,
        dark=True,
        fastapi_docs=True,
        storage_secret="chronicle-role-d",
    )


if __name__ == "__main__":
    run()
