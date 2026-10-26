"""NiceGUI dashboard for the integrated A -> B -> C runtime.

This is the team-repo port of the prototype dashboard.  It is intentionally
decoupled from the prototype's ``ChronicleEngine``: all data comes from
:class:`chronicle.integration.ChronicleRuntime`, which uses the team Role A
perception pipeline, Role B dual graph and Role C semantic engine.
"""

from __future__ import annotations

import copy
import queue
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from nicegui import app, run, ui

from chronicle.integration import ChronicleRuntime
from chronicle.perception.config import BROAD_VOCABULARY, PerceptionConfig
from chronicle.semantic.config import SemanticConfig
from chronicle.sources import (
    DEFAULT_DATA_DIR,
    SOURCES,
    ClipSource,
    default_source,
    source_for_key,
    source_for_video,
)
from chronicle.ui.graph_view import graph_options as _graph_options
from chronicle.ui.graph_view import timeline_options as _timeline_options
from chronicle.ui.snapshot_io import SnapshotImportError, parse_snapshot, snapshot_summary
from chronicle.ui.uploads import (
    MAX_UPLOAD_BYTES,
    UploadError,
    load_upload_sources,
    media_type,
    new_upload_target,
    probe_video,
    register_upload,
)
from chronicle.vfm.service import DOWNLOAD_NAMES as VFM_DOWNLOAD_NAMES
from chronicle.vfm.service import VFMJobManager


@dataclass
class RunRequest:
    source: ClipSource
    video: Path
    max_frames: int | None = None


class DashboardState:
    """Own the active runtime and serialise source switches."""

    def __init__(
        self,
        data_dir: Path,
        source: ClipSource,
        video: Path,
        *,
        base_config: PerceptionConfig | None = None,
        semantic_config: SemanticConfig | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.base_config = base_config or PerceptionConfig()
        self.semantic_config = semantic_config or SemanticConfig.from_env()
        self.source = source
        self.source_key = source.key
        self.video = Path(video)
        self.max_frames = max_frames
        self.lock = threading.RLock()
        self.running = False
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None
        self.runtime = self._make_runtime(source, reset=False)
        self.vfm: Any = None
        self.vfm_error: str | None = None
        self.on_ingest_done: Callable[[ClipSource, Path], None] | None = None
        self._requests: queue.Queue[RunRequest | None] = queue.Queue()
        self._active: RunRequest | None = None
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _make_runtime(self, source: ClipSource, *, reset: bool) -> ChronicleRuntime:
        config = copy.deepcopy(self.base_config)
        config.vocabulary = tuple(source.vocabulary)
        return ChronicleRuntime(
            source.db_path(self.data_dir),
            perception_config=config,
            semantic_config=self.semantic_config,
            zone_label=source.zone_label,
            reset=reset,
        )

    def request_run(
        self,
        source: ClipSource,
        video: Path | None = None,
        *,
        max_frames: int | None = None,
    ) -> None:
        request = RunRequest(source, Path(video or source.video_path(self.data_dir)), max_frames)
        with self.lock:
            self._active = request
            self.running = True
            self.error = None
            self.summary = None
            self.source = source
            self.source_key = source.key
            self.video = request.video
        self._requests.put(request)
        current = self.runtime
        if current is not None:
            current.stop()

    def _worker_loop(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            while True:
                try:
                    newer = self._requests.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    return
                request = newer
            old: ChronicleRuntime | None = None
            runtime: ChronicleRuntime | None = None
            try:
                with self.lock:
                    old = self.runtime
                runtime = self._make_runtime(request.source, reset=True)
                # Swap and retire the previous runtime while holding the lock:
                # ``refresh`` snapshots under the same lock, so it can never see
                # a runtime whose SQLite connection has already been closed.
                with self.lock:
                    if self._active is not request:
                        runtime.close()
                        continue
                    self.runtime = runtime
                    if old is not None:
                        old.stop()
                        old.close()
                summary = runtime.ingest_video(
                    request.video,
                    max_frames=request.max_frames or self.max_frames,
                )
                with self.lock:
                    if self._active is request:
                        self.summary = summary.as_dict()
                        active = True
                    else:
                        active = False
                callback = self.on_ingest_done
                if active and callback is not None:
                    # Compile the memory server-side so it still happens when
                    # the browser tab that started the upload is closed.
                    try:
                        callback(request.source, request.video)
                    except Exception as exc:  # keep the ingest worker alive
                        with self.lock:
                            self.vfm_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # surfaced in the UI
                with self.lock:
                    if self._active is request:
                        self.error = f"{type(exc).__name__}: {exc}"
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception:
                        pass
            finally:
                with self.lock:
                    if self._active is request:
                        self.running = False

    def stop(self) -> None:
        with self.lock:
            current = self.runtime
            self._active = None
            self.running = False
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        if current is not None:
            current.stop()

    def close(self) -> None:
        self.stop()
        self._requests.put(None)
        self._thread.join(timeout=5.0)
        with self.lock:
            try:
                self.runtime.close()
            except Exception:
                pass


def _video_url(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"/source-video?v={stamp}"


def _custom_source(video: Path, data_dir: Path) -> ClipSource:
    return ClipSource(
        key="custom",
        label=video.stem,
        badge="CUSTOM",
        caption=video.name,
        description="Explicit video passed on the command line.",
        relative_path=str(video),
        relative_db=str(Path(data_dir) / "custom" / "chronicle.db"),
        zone_label="Custom Scene",
        vocabulary=PerceptionConfig().vocabulary,
        questions=("what did I see?",),
    )


def _upload_vocabulary(raw: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma/semicolon separated target list for a video upload."""
    labels = tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in raw.replace("\n", ",").replace(";", ",").split(",")
            if part.strip()
        )
    )
    return labels or tuple(fallback)


def create_app(
    *,
    data_dir: str | Path | None = None,
    video: str | Path | None = None,
    source_key: str | None = None,
    auto_run: bool = False,
    port: int = 8080,
    host: str = "127.0.0.1",
    show: bool = False,
    max_frames: int | None = None,
    detector_backend: str = "auto",
    hands_backend: str = "auto",
    embedder_backend: str = "auto",
    semantic_config: SemanticConfig | None = None,
    qwen: bool = False,
    qwen_model: str | None = None,
    vfm_enabled: bool = True,
    vfm_window_s: float = 30.0,
    vfm_frames_per_window: int = 3,
    vfm_max_snapshots: int = 40,
    vfm_delete_video: bool = False,
    start_server: bool = True,
) -> DashboardState:
    data_root = DEFAULT_DATA_DIR if data_dir is None else Path(data_dir).expanduser().resolve()
    try:
        from nicegui import core

        if not hasattr(core.app.config, "prod_js"):
            core.app.config.add_run_config(
                reload=False,
                title="Chronicle — episodic memory",
                viewport="width=device-width, initial-scale=1",
                favicon="🕸️",
                dark=True,
                language=None,
                binding_refresh_interval=0.1,
                reconnect_timeout=3.0,
                message_history_length=1000,
                cache_control_directives="public, max-age=31536000",
                tailwind=True,
                unocss=None,
                prod_js=True,
                show_welcome_message=False,
                markdown=False,
            )
    except Exception:
        pass
    custom: ClipSource | None = None
    upload_sources = load_upload_sources(data_root)
    if video is not None:
        video_path = Path(video).expanduser().resolve()
        matched = source_for_video(video_path, data_root)
        if matched is None:
            matched = next(
                (
                    source
                    for source in upload_sources
                    if source.video_path(data_root) == video_path
                ),
                None,
            )
        if matched is None and video_path.is_file():
            custom = _custom_source(video_path, data_root)
            source_key = "custom"
        elif matched is not None:
            source_key = matched.key
    if source_key is None:
        source_key = (custom or default_source(data_root)).key
    sources = ([custom] if custom is not None else []) + upload_sources + list(SOURCES)
    source_map = {source.key: source for source in sources}
    try:
        initial = source_for_key(source_key, data_root)
    except KeyError:
        initial = custom or source_map.get(source_key) or default_source(data_root)
    if custom is not None:
        initial = custom
    video_path = (
        Path(video).expanduser().resolve()
        if video is not None
        else initial.video_path(data_root)
    )

    base_config = PerceptionConfig()
    base_config.detector.backend = detector_backend
    base_config.hands.backend = hands_backend
    base_config.embedder.backend = embedder_backend
    state = DashboardState(
        data_root,
        initial,
        video_path,
        base_config=base_config,
        semantic_config=semantic_config,
        max_frames=max_frames,
    )
    vfm = (
        VFMJobManager(
            data_root,
            qwen_model=qwen_model or state.semantic_config.llm_model,
            ollama_url=state.semantic_config.ollama_url,
            qwen_timeout=state.semantic_config.llm_timeout_s,
            window_s=vfm_window_s,
            frames_per_window=vfm_frames_per_window,
            max_snapshots=vfm_max_snapshots,
            delete_video=vfm_delete_video,
        )
        if vfm_enabled
        else None
    )
    if vfm is not None:
        app.on_shutdown(vfm.close)
        state.on_ingest_done = lambda source, video: vfm.submit(source, video)
    state.vfm = vfm

    @app.get("/source-video")
    def _source_video() -> FileResponse:
        path = Path(state.video).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"video not found: {path}")
        return FileResponse(
            path,
            media_type=media_type(path),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/export/snapshot.json")
    def _export_snapshot_json() -> Response:
        with state.lock:
            runtime = state.runtime
        content = runtime.export_snapshot_json()
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="chronicle_snapshot.json"'},
        )

    @app.get("/export/entities.csv")
    def _export_entities_csv() -> Response:
        with state.lock:
            runtime = state.runtime
        content = runtime.export_entities_csv()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="chronicle_entities.csv"'},
        )

    @app.get("/export/actions.csv")
    def _export_actions_csv() -> Response:
        with state.lock:
            runtime = state.runtime
        content = runtime.export_actions_csv()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="chronicle_actions.csv"'},
        )

    def _vfm_source(source_key: str) -> ClipSource:
        source = source_map.get(source_key)
        if source is None:
            raise HTTPException(status_code=404, detail=f"unknown source: {source_key}")
        return source

    @app.get("/vfm/status/{source_key}")
    def _vfm_status_route(source_key: str) -> dict[str, Any]:
        if vfm is None:
            return {"enabled": False, "status": None}
        source = _vfm_source(source_key)
        return {
            "enabled": True,
            "status": vfm.status(source),
            "files": vfm.files(source),
        }

    @app.get("/vfm/report/{source_key}")
    def _vfm_report_route(source_key: str) -> FileResponse:
        if vfm is None:
            raise HTTPException(status_code=404, detail="VFM is disabled")
        source = _vfm_source(source_key)
        path = vfm.artifact(source, "memory.html")
        if path is None:
            raise HTTPException(status_code=404, detail="memory report is not compiled yet")
        return FileResponse(
            path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/vfm/download/{source_key}/{name}")
    def _vfm_download_route(source_key: str, name: str) -> FileResponse:
        if vfm is None:
            raise HTTPException(status_code=404, detail="VFM is disabled")
        source = _vfm_source(source_key)
        path = vfm.artifact(source, name)
        if path is None:
            raise HTTPException(status_code=404, detail=f"VFM artifact not available: {name}")
        return FileResponse(path, filename=path.name)

    @app.get("/vfm/evidence/{source_key}/{name}")
    def _vfm_evidence_route(source_key: str, name: str) -> FileResponse:
        if vfm is None:
            raise HTTPException(status_code=404, detail="VFM is disabled")
        source = _vfm_source(source_key)
        path = vfm.evidence_file(source, name)
        if path is None:
            raise HTTPException(status_code=404, detail=f"evidence not found: {name}")
        return FileResponse(path)

    @app.post("/vfm/ask/{source_key}")
    async def _vfm_ask_route(source_key: str, request: Request) -> dict[str, Any]:
        if vfm is None:
            raise HTTPException(status_code=404, detail="VFM is disabled")
        source = _vfm_source(source_key)
        payload = await request.json()
        question = str((payload or {}).get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        try:
            result = await run.io_bound(
                vfm.ask,
                source,
                question,
                use_qwen=bool((payload or {}).get("qwen")),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return result.to_dict()

    @app.post("/import/snapshot")
    async def _import_snapshot_route(request: Request) -> dict[str, Any]:
        payload = await request.json()
        with state.lock:
            state.runtime.load_snapshot_json(payload)
        return {
            "status": "imported",
            "entities": len(payload.get("entities", [])),
            "furniture": len(payload.get("furniture", [])),
        }

    def _switch_source(source: ClipSource) -> bool:
        target = source.video_path(data_root)
        if source.key == "custom":
            target = Path(state.video)
        if not target.is_file():
            ui.notify(
                f"missing {source.label}: add {source.relative_path} under {data_root}.",
                type="negative",
            )
            return False
        state.request_run(source, target)
        ui.notify(f"switching to {source.label}", type="info")
        return True

    def _run_current() -> None:
        source = source_map.get(state.source_key)
        if source is not None:
            _switch_source(source)

    def _vfm_download(name: str) -> None:
        source = source_map.get(state.source_key)
        if vfm is None or source is None:
            ui.notify("VFM memory is disabled.", type="warning")
            return
        path = vfm.artifact(source, name)
        if path is None:
            ui.notify("Compile the memory first (Memory tab).", type="warning")
            return
        ui.download(f"/vfm/download/{source.key}/{name}", path.name)

    def _vfm_open_report() -> None:
        source = source_map.get(state.source_key)
        if vfm is None or source is None:
            ui.notify("VFM memory is disabled.", type="warning")
            return
        if vfm.artifact(source, "memory.html") is None:
            ui.notify("Compile the memory first (Memory tab).", type="warning")
            return
        ui.navigate.to(f"/vfm/report/{source.key}", new_tab=True)

    def _vfm_compile(force: bool = False) -> None:
        source = source_map.get(state.source_key)
        if vfm is None or source is None:
            ui.notify("VFM memory is disabled.", type="warning")
            return
        job = vfm.submit(source, state.video, force=force)
        if job is None:
            ui.notify("VFM memory is disabled.", type="warning")
            return
        ui.notify(
            f"compiling {source.label} memory…" if force else f"queued {source.label} memory",
            type="info",
        )

    if auto_run and video_path.is_file():
        state.request_run(initial, video_path)

    suggestion_state: dict[str, Any] = {"row": None, "signature": None}

    def _tracked_labels() -> set[str]:
        try:
            with state.lock:
                snapshot = state.runtime.snapshot()
            return {str(row.get("label", "")).lower() for row in snapshot.get("entities", [])}
        except Exception:
            return set()

    def _suggestion_prompts(source_key: str) -> list[str]:
        source = source_map.get(source_key, initial)
        prompts = list(source.questions)
        tracked = _tracked_labels()
        for label in ("cup", "joy-con", "backpack", "keys", "sofa", "mug", "phone"):
            if label not in tracked:
                continue
            for prompt in (f"what colour is the {label}?", f"where is the {label}?"):
                if prompt not in prompts:
                    prompts.append(prompt)
        if "what did I see?" not in prompts:
            prompts.append("what did I see?")
        return prompts[:5]

    def _render_suggestions(source_key: str, ask: Any) -> None:
        row = suggestion_state["row"]
        if row is None:
            return
        prompts = _suggestion_prompts(source_key)
        signature = (source_key, tuple(prompts))
        if signature == suggestion_state["signature"]:
            return
        suggestion_state["signature"] = signature
        row.clear()
        with row:
            for prompt in prompts:
                ui.button(
                    prompt,
                    on_click=lambda p=prompt: ask(p),
                ).props("flat dense size=sm no-caps").classes(
                    "text-teal-300 text-[10px]"
                )

    if not hasattr(app.config, "markdown"):
        app.config.markdown = False

    @ui.page("/", markdown=False)
    def index() -> None:
        ui.dark_mode().enable()
        ui.colors(primary="#2dd4bf", secondary="#38bdf8", accent="#f59e0b")
        ui.add_head_html(
            "<style>"
            "body{background-color:#070b11;color:#dbe4ee;}"
            "body::before{content:'';position:fixed;inset:0;z-index:-1;"
            "pointer-events:none;background:radial-gradient(900px 480px at 18% -12%,"
            "rgba(45,212,191,.08),transparent 62%);}"
            ".chronicle-card{background:rgba(13,20,30,.84);"
            "border:1px solid rgba(148,163,184,.14);border-radius:14px;overflow:hidden;}"
            ".chronicle-mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
            ".chronicle-title{font-size:11px;text-transform:uppercase;"
            "letter-spacing:.18em;color:#64748b;}"
            ".q-field__control{background:rgba(2,6,13,.55)!important;}"
            ".q-tab{padding:0 14px;min-height:34px;}"
            ".chronicle-dropzone{flex-shrink:0;}"
            ".chronicle-dropzone .q-uploader{width:100%;max-height:118px;"
            "background:rgba(2,6,13,.35);border:1px dashed rgba(45,212,191,.42);"
            "border-radius:10px;box-shadow:none;}"
            ".chronicle-dropzone .q-uploader__header{background:transparent!important;"
            "color:#94a3b8!important;}"
            ".chronicle-dropzone .q-uploader__header-content{padding:5px 10px;"
            "min-height:34px;}"
            ".chronicle-dropzone .q-uploader__list{background:transparent!important;"
            "color:#cbd5e1!important;padding:0 10px 6px;}"
            ".chronicle-dropzone .q-uploader__file{background:transparent!important;}"
            ".chronicle-dropzone .q-uploader__header .q-btn{color:#2dd4bf!important;}"
            "</style>"
        )

        async def _import_snapshot(event: Any) -> None:
            file = getattr(event, "file", None)
            if file is None:
                ui.notify("No file received", type="negative")
                return
            try:
                payload = parse_snapshot(await file.text())
            except SnapshotImportError as exc:
                ui.notify(f"Import failed: {exc}", type="negative")
                return
            except Exception as exc:  # defensive: upload/stream failure
                ui.notify(f"Could not read file: {exc}", type="negative")
                return
            state.stop()
            with state.lock:
                runtime = state.runtime
            try:
                await run.io_bound(runtime.load_snapshot_json, payload)
            except Exception as exc:
                ui.notify(f"Import failed: {type(exc).__name__}: {exc}", type="negative")
                return
            import_upload.reset()
            import_dialog.close()
            ui.notify(snapshot_summary(payload), type="positive")

        with ui.dialog() as import_dialog, ui.card().classes("dark chronicle-card").style(
            "min-width:390px;max-width:480px"
        ):
            ui.label("Import snapshot").classes("text-sm font-semibold")
            ui.label(
                "Upload a chronicle_snapshot.json exported from this dashboard. "
                "The graph is reloaded from the file without re-ingesting video."
            ).classes("text-xs text-slate-400")
            import_upload = ui.upload(
                label="chronicle_snapshot.json",
                on_upload=_import_snapshot,
                auto_upload=True,
                max_file_size=8 * 1024 * 1024,
            ).props("accept=.json flat dense").classes("w-full")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=import_dialog.close).props(
                    "flat dense no-caps"
                ).classes("text-xs")

        with ui.header().classes("items-center px-6 py-2").style(
            "background:rgba(7,11,17,.92);border-bottom:1px solid rgba(148,163,184,.12)"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("hub", size="22px").classes("text-teal-300")
                ui.label("Chronicle").classes("text-base font-semibold tracking-wide")
                ui.label("egocentric memory").classes(
                    "text-[10px] uppercase tracking-[0.18em] text-slate-500"
                )
            with ui.row().classes("items-center gap-2").style("margin-left:auto"):
                status_chip = ui.badge("IDLE").props("outline").classes(
                    "text-xs chronicle-mono"
                )
                fps_chip = ui.badge("0.0 fps").props("outline").classes(
                    "text-xs chronicle-mono"
                )
                camera_chip = ui.badge("CAM · —").props("outline").classes(
                    "text-xs chronicle-mono"
                )
                queue_chip = ui.badge("QUEUE · 0").props("outline").classes(
                    "text-xs chronicle-mono"
                )
                with queue_chip:
                    queue_tip = ui.tooltip("Role C verification queue: depth 0")

                llm_chip = ui.badge("LLM · CHECKING").props("outline").classes(
                    "text-xs chronicle-mono"
                )
                with llm_chip:
                    llm_tip = ui.tooltip("Local model status")

                ui.button("Import", icon="upload", on_click=import_dialog.open).props(
                    "outline dense size=sm no-caps"
                ).classes("text-xs chronicle-mono text-slate-300").tooltip(
                    "Reload an exported snapshot without re-ingesting video"
                )

                with ui.button("Export", icon="download").props(
                    "outline dense size=sm no-caps"
                ).classes("text-xs chronicle-mono text-teal-300"):
                    with ui.menu().classes("dark chronicle-card"):
                        ui.menu_item(
                            "Snapshot (JSON)",
                            on_click=lambda: ui.download(
                                "/export/snapshot.json", "chronicle_snapshot.json"
                            ),
                        )
                        ui.menu_item(
                            "Entities (CSV)",
                            on_click=lambda: ui.download(
                                "/export/entities.csv", "chronicle_entities.csv"
                            ),
                        )
                        ui.menu_item(
                            "Actions (CSV)",
                            on_click=lambda: ui.download(
                                "/export/actions.csv", "chronicle_actions.csv"
                            ),
                        )
                        ui.separator()
                        ui.menu_item(
                            "VFM memory (ZIP)",
                            on_click=lambda: _vfm_download("memory.zip"),
                        )
                        ui.menu_item(
                            "Memory pack (MD)",
                            on_click=lambda: _vfm_download("memory_pack.md"),
                        )
                        ui.menu_item(
                            "Open memory report",
                            on_click=_vfm_open_report,
                        )

        with ui.column().classes("w-full max-w-[1440px] mx-auto px-5 pt-3 gap-3"):
            with ui.card().props("flat").classes("chronicle-card w-full"):
                with ui.row().classes("w-full items-center justify-between px-3 py-1"):
                    stat_cards: dict[str, Any] = {}
                    for i, (key, label) in enumerate(
                        (
                            ("entities", "ENTITIES"),
                            ("actions", "ACTIONS"),
                            ("transitions", "TRANSITIONS"),
                            ("zones", "ZONES"),
                            ("furniture", "FURNITURE"),
                        )
                    ):
                        if i:
                            ui.separator().props("vertical").classes("h-6 self-center")
                        with ui.row().classes("items-baseline gap-1"):
                            stat_cards[key] = ui.label("0").classes(
                                "text-lg font-semibold chronicle-mono text-slate-200"
                            )
                            ui.label(label).classes(
                                "text-[9px] tracking-[0.16em] text-slate-500"
                            )

            with ui.row().classes("w-full items-stretch gap-3 flex-nowrap"):
                with ui.card().props("flat").classes("chronicle-card p-4").style(
                    "flex:9 1 0;min-width:0;height:500px;display:flex;flex-direction:column"
                ):
                    source_sync = {"active": False}
                    upload_busy = {"active": False, "name": ""}
                    upload_vocab = {
                        "value": ", ".join(base_config.vocabulary),
                        "broad": False,
                    }

                    def _on_source_change(event: Any) -> None:
                        if source_sync["active"]:
                            return
                        source = source_map.get(event.value)
                        if source is None:
                            return
                        if not _switch_source(source):
                            source_sync["active"] = True
                            source_select.value = state.source_key
                            source_sync["active"] = False

                    async def _on_video_upload(event: Any) -> None:
                        """Save the selected clip and start processing immediately."""

                        if upload_busy["active"]:
                            ui.notify("A video is already uploading.", type="warning")
                            video_upload.reset()
                            return
                        upload_busy["active"] = True
                        name = Path(str(event.file.name)).name
                        upload_busy["name"] = name
                        upload_button.disable()
                        target: Path | None = None
                        try:
                            target = new_upload_target(data_root, name)
                            await event.file.save(target)
                            probe = await run.io_bound(probe_video, target)
                            if probe is None:
                                shutil.rmtree(target.parent, ignore_errors=True)
                                target = None
                                ui.notify(
                                    "The selected file could not be decoded as a video.",
                                    type="negative",
                                )
                                return
                            source = register_upload(
                                data_root,
                                target,
                                original_name=name,
                                vocabulary=(
                                    BROAD_VOCABULARY
                                    if upload_vocab["broad"]
                                    else _upload_vocabulary(
                                        upload_vocab["value"], base_config.vocabulary
                                    )
                                ),
                            )
                            source_map[source.key] = source
                            source_select.options[source.key] = source.label
                            source_sync["active"] = True
                            source_select.value = source.key
                            source_sync["active"] = False
                            source_select.update()
                            state.request_run(source, target)
                            width, height, fps = probe
                            video_badge.text = f"PRE-RECORDED · {source.badge}"
                            video_name_label.text = target.name
                            video_caption_label.text = f"{source.caption} · processing live"
                            ui.notify(
                                f"Processing {name} · {width}×{height} · {fps:.1f} fps",
                                type="positive",
                            )
                            log_view.push(
                                f"[SYSTEM] uploaded {name}; live processing started"
                            )
                        except UploadError as exc:
                            ui.notify(f"Upload failed: {exc}", type="negative")
                        except Exception as exc:
                            if target is not None and not (
                                target.parent / "source.json"
                            ).exists():
                                shutil.rmtree(target.parent, ignore_errors=True)
                            ui.notify(
                                f"Upload failed: {type(exc).__name__}: {exc}",
                                type="negative",
                            )
                            log_view.push(f"upload error: {type(exc).__name__}: {exc}")
                        finally:
                            upload_busy["active"] = False
                            upload_busy["name"] = ""
                            upload_button.enable()
                            video_upload.reset()

                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2 min-w-0"):
                            video_badge = ui.badge("PRE-RECORDED").props("outline").classes(
                                "text-[10px] chronicle-mono text-teal-300"
                            )
                            video_name_label = ui.label(Path(state.video).name).classes(
                                "text-[11px] chronicle-mono text-slate-500 truncate"
                            )
                        with ui.row().classes("items-center gap-1"):
                            source_select = ui.select(
                                {s.key: s.label for s in sources},
                                value=initial.key,
                                on_change=_on_source_change,
                            ).props("outlined dense dark options-dense").style(
                                "min-width:180px;max-width:230px"
                            )
                            upload_button = ui.button(
                                icon="video_file",
                                on_click=lambda: video_upload.run_method("pickFiles"),
                            ).props("flat round dense color=teal").tooltip(
                                "Change demo video · upload and process it live"
                            )
                            ui.button(icon="play_arrow", on_click=_run_current).props(
                                "flat round dense"
                            ).tooltip("Run the selected source")
                            ui.button(icon="stop", on_click=state.stop).props(
                                "flat round dense color=negative"
                            ).tooltip("Stop the current run")
                    with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                        ui.input(
                            "Objects to track (comma separated)",
                            value=upload_vocab["value"],
                            on_change=lambda event: upload_vocab.__setitem__(
                                "value", event.value
                            ),
                        ).props("outlined dense dark").classes("grow").tooltip(
                            "YOLO-World prompts for uploaded clips. Add monitor, mouse, "
                            "keyboard or any object you want the memory to track."
                        )
                        ui.checkbox(
                            "Broad scene",
                            value=False,
                            on_change=lambda event: upload_vocab.__setitem__(
                                "broad", bool(event.value)
                            ),
                        ).props("dense").classes("text-xs").tooltip(
                            "Track ~100 common household and egocentric objects. "
                            "Better coverage, more false positives."
                        )
                    video_upload = ui.upload(
                        label="Drag & drop a video to scan live · or add a file",
                        on_upload=_on_video_upload,
                        on_rejected=lambda: ui.notify(
                            "Only video files up to 4 GB are accepted.",
                            type="negative",
                        ),
                        auto_upload=True,
                        max_file_size=MAX_UPLOAD_BYTES,
                        max_files=1,
                    ).props(
                        'accept="video/*" flat bordered dark color=teal'
                    ).classes("w-full chronicle-dropzone")
                    video_player = ui.video(
                        _video_url(Path(state.video)),
                        controls=True,
                        autoplay=True,
                        muted=True,
                        loop=True,
                    ).style(
                        "flex:1;min-height:0;width:100%;background:#000;"
                        "object-fit:contain;border-radius:10px"
                    )
                    with ui.row().classes("w-full items-center justify-between").style(
                        "flex-shrink:0"
                    ):
                        video_caption_label = ui.label(initial.caption).classes(
                            "text-[11px] text-teal-300"
                        )
                        video_time_label = ui.label("processed t=0.0s").classes(
                            "text-[10px] chronicle-mono text-slate-500"
                        )

                with ui.card().props("flat").classes("chronicle-card p-4").style(
                    "flex:11 1 0;min-width:0;height:500px;display:flex;flex-direction:column"
                ):
                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label("Memory graph").classes("chronicle-title")
                            ui.button(
                                "JSON",
                                icon="file_download",
                                on_click=lambda: ui.download(
                                    "/export/snapshot.json", "chronicle_snapshot.json"
                                ),
                            ).props("flat dense size=xs no-caps").classes(
                                "text-slate-400 text-[10px]"
                            ).tooltip("Download graph snapshot JSON")
                            ui.button(
                                "CSV",
                                icon="table_view",
                                on_click=lambda: ui.download(
                                    "/export/entities.csv", "chronicle_entities.csv"
                                ),
                            ).props("flat dense size=xs no-caps").classes(
                                "text-slate-400 text-[10px]"
                            ).tooltip("Download entities CSV")
                            ui.button(
                                "Import",
                                icon="file_upload",
                                on_click=import_dialog.open,
                            ).props("flat dense size=xs no-caps").classes(
                                "text-slate-400 text-[10px]"
                            ).tooltip("Import snapshot JSON")
                        pipeline_label = ui.label("").classes(
                            "text-[10px] chronicle-mono text-slate-500 truncate"
                        )
                    with ui.tabs().props("dense no-caps").classes("w-full").style(
                        "background:transparent;min-height:34px"
                    ) as tabs:
                        graph_tab = ui.tab("Graph", icon="hub")
                        timeline_tab = ui.tab("Timeline", icon="timeline")
                        entities_tab = ui.tab("Entities", icon="inventory_2")
                        actions_tab = ui.tab("Actions", icon="bolt")
                        memory_tab = ui.tab("Memory", icon="psychology")
                    with ui.tab_panels(tabs, value=graph_tab).classes("w-full").style(
                        "background:transparent;flex:1;min-height:0"
                    ):
                        with ui.tab_panel(graph_tab).style("padding:4px 0 0 0"):
                            graph_chart = ui.echart(
                                _graph_options({"nodes": [], "edges": []})
                            ).classes("w-full").style("height:396px")
                        with ui.tab_panel(timeline_tab).style("padding:4px 0 0 0"):
                            timeline_chart = ui.echart(_timeline_options([])).classes(
                                "w-full"
                            ).style("height:396px")
                        with ui.tab_panel(entities_tab).style("padding:4px 0 0 0"):
                            with ui.row().classes(
                                "w-full items-center gap-2 pb-1"
                            ).style("flex-wrap:nowrap"):
                                entity_filter = ui.input(
                                    placeholder="Filter by label, colour, zone…"
                                ).props("outlined dense dark clearable").classes(
                                    "flex-1"
                                ).style("min-width:0")
                                ui.label("click a row to ask · colour ring = detected colour").classes(
                                    "text-[9px] text-slate-600 whitespace-nowrap"
                                )
                            entity_table = ui.table(
                                columns=[
                                    {"name": "label", "label": "Label", "field": "label", "align": "left"},
                                    {"name": "colour", "label": "Colour", "field": "colour", "align": "left"},
                                    {"name": "appearance", "label": "Properties", "field": "appearance", "align": "left"},
                                    {"name": "state", "label": "State", "field": "state", "align": "left"},
                                    {"name": "zone", "label": "Zone", "field": "zone", "align": "left"},
                                    {"name": "anchor", "label": "Anchor", "field": "anchor", "align": "left"},
                                    {"name": "container", "label": "Container", "field": "container", "align": "left"},
                                    {"name": "span", "label": "Visible window", "field": "span", "align": "left"},
                                ],
                                rows=[],
                                row_key="id",
                                pagination=8,
                            ).props("dark flat dense virtual-scroll").classes(
                                "w-full"
                            ).style("height:346px")
                            entity_table.add_slot(
                                "body-cell-colour",
                                """
                                <q-td :props="props">
                                  <div style="display:flex;align-items:center;gap:6px">
                                    <span :style="{width:'11px',height:'11px',borderRadius:'50%',
                                      background:props.row.colour_hex || 'transparent',
                                      border:'1px solid rgba(148,163,184,.45)',
                                      display:'inline-block'}"></span>
                                    <span>{{ props.value }}</span>
                                  </div>
                                </q-td>
                                """,
                            )
                        with ui.tab_panel(actions_tab).style("padding:4px 0 0 0"):
                            action_table = ui.table(
                                columns=[
                                    {"name": "time", "label": "Time", "field": "time", "align": "left"},
                                    {"name": "action", "label": "Action", "field": "action", "align": "left"},
                                    {"name": "entity", "label": "Entity", "field": "entity", "align": "left"},
                                    {"name": "caption", "label": "Caption", "field": "caption", "align": "left"},
                                    {"name": "source", "label": "Source", "field": "source", "align": "left"},
                                ],
                                rows=[],
                                row_key="id",
                                pagination=8,
                            ).props("dark flat dense virtual-scroll").classes(
                                "w-full"
                            ).style("height:396px")
                        with ui.tab_panel(memory_tab).style(
                            "padding:4px 0 0 0;display:flex;flex-direction:column;gap:5px"
                        ):
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-nowrap"
                            ):
                                ui.label("VFM").classes("chronicle-title")
                                vfm_status_badge = ui.badge("NO MEMORY").props(
                                    "outline"
                                ).classes("text-[10px] chronicle-mono")
                                vfm_stage_label = ui.label("not compiled").classes(
                                    "text-[10px] chronicle-mono text-slate-500 truncate"
                                )
                                vfm_qwen_switch = ui.checkbox(
                                    "Qwen", value=qwen
                                ).props("dense").classes("text-[10px]").tooltip(
                                    "Answer from the memory pack with Qwen. Off = "
                                    "deterministic vault query."
                                )
                            vfm_progress = ui.linear_progress(
                                value=0, show_value=False
                            ).classes("w-full").style("height:4px")
                            vfm_stats_label = ui.label(
                                "Compile the memory from the current run, or upload a video "
                                "to compile it automatically."
                            ).classes("text-[10px] chronicle-mono text-slate-500")
                            with ui.row().classes("w-full items-center gap-1 flex-wrap"):
                                ui.button(
                                    "Compile",
                                    icon="build",
                                    on_click=lambda: _vfm_compile(True),
                                ).props("flat dense size=xs no-caps").classes(
                                    "text-[10px] text-teal-300"
                                ).tooltip("Rebuild vault, evidence, report and memory pack")
                                vfm_report_button = ui.button(
                                    "Open report",
                                    icon="open_in_new",
                                    on_click=_vfm_open_report,
                                ).props("flat dense size=xs no-caps").classes(
                                    "text-[10px]"
                                )
                                for vfm_name in VFM_DOWNLOAD_NAMES:
                                    ui.button(
                                        vfm_name,
                                        icon="download",
                                        on_click=lambda n=vfm_name: _vfm_download(n),
                                    ).props("flat dense size=xs no-caps").classes(
                                        "text-[10px] chronicle-mono"
                                    )
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-nowrap"
                            ):
                                vfm_ask_input = ui.input(
                                    placeholder="where is the cup? / what happened around 20s?"
                                ).props("outlined dense dark").classes("flex-1")
                                vfm_ask_button = ui.button(
                                    "Ask memory", icon="search"
                                ).props("unelevated dense no-caps").classes("text-xs")
                            vfm_answer_label = ui.label(
                                "The memory pack is text-only: captions, events, verified "
                                "answers and evidence refs. Qwen never receives frames."
                            ).classes(
                                "text-xs text-slate-300 whitespace-pre-line w-full"
                            ).style("flex:1;min-height:0;overflow-y:auto")
                            vfm_citations_row = ui.row().classes(
                                "w-full gap-1 items-center flex-wrap"
                            )
                            vfm_answer_meta = ui.label("").classes(
                                "text-[9px] chronicle-mono text-slate-500"
                            )

            with ui.card().props("flat").classes("chronicle-card p-4 w-full").style(
                "height:264px;display:flex;flex-direction:column"
            ):
                with ui.grid().classes("w-full").style(
                    "grid-template-columns:minmax(0,3fr) minmax(0,2fr);"
                    "gap:0;flex:1;min-height:0"
                ):
                    with ui.column().classes("gap-2").style("min-width:0;height:100%"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Ask the memory").classes("chronicle-title")
                            ui.label("grounded in SQLite · local model").classes(
                                "text-[10px] text-slate-600"
                            )
                        with ui.row().classes("w-full items-center gap-2"):
                            query_input = ui.input(
                                placeholder="what was my friend doing with the cup?"
                            ).classes("flex-1").props("outlined dense dark")
                            ask_button = ui.button("Ask", icon="search").props(
                                "unelevated dense no-caps"
                            )
                        answer_label = ui.label(
                            "Answers are grounded in the graph: location, state, "
                            "actions, co-presence and last-seen context."
                        ).classes(
                            "text-xs text-slate-300 whitespace-pre-line w-full"
                        ).style("flex:1;min-height:0;overflow-y:auto")
                        answer_meta = ui.label("").classes(
                            "text-[9px] chronicle-mono text-slate-500"
                        )
                        suggestion_state["row"] = ui.row().classes(
                            "gap-1 items-center flex-wrap"
                        ).style("flex-shrink:0")
                    with ui.column().classes("gap-2 pl-4").style(
                        "min-width:0;height:100%;border-left:1px solid rgba(148,163,184,.14)"
                    ):
                        ui.label("Pipeline log").classes("chronicle-title")
                        log_view = ui.log(max_lines=400).classes(
                            "w-full chronicle-mono"
                        ).style(
                            "flex:1;min-height:0;background:rgba(2,6,13,.45);"
                            "border-radius:10px;padding:8px"
                        )

        async def _ask(query: str | None = None) -> None:
            text = (query or query_input.value or "").strip()
            if not text:
                return
            answer_label.text = "retrieving…"
            answer_meta.text = ""
            try:
                with state.lock:
                    runtime = state.runtime
                result = await run.io_bound(runtime.ask, text)
                answer_label.text = result.answer
                citations = " · ".join(
                    f"graph@{float(cite.get('timestamp') or 0):.1f}s"
                    for cite in result.citations[:4]
                )
                answer_meta.text = (
                    f"method={result.method} · {result.elapsed * 1000:.0f} ms · "
                    f"{len(result.citations)} citations"
                    + (f" · {citations}" if citations else "")
                )
            except Exception as exc:
                answer_label.text = f"{type(exc).__name__}: {exc}"
                answer_meta.text = "query failed"

        ask_button.on("click", lambda: _ask())
        query_input.on("keydown.enter", lambda: _ask())

        async def _ask_vfm() -> None:
            source = source_map.get(state.source_key)
            text = (vfm_ask_input.value or "").strip()
            if vfm is None or source is None:
                ui.notify("VFM memory is disabled.", type="warning")
                return
            if not text:
                return
            vfm_ask_button.disable()
            vfm_answer_label.text = "retrieving from the memory pack…"
            vfm_answer_meta.text = ""
            vfm_citations_row.clear()
            try:
                result = await run.io_bound(
                    vfm.ask,
                    source,
                    text,
                    use_qwen=bool(vfm_qwen_switch.value),
                )
                vfm_answer_label.text = result.answer
                citations = " · ".join(
                    f"t={float(cite.get('t') or 0):.1f}s"
                    for cite in result.citations[:4]
                )
                vfm_answer_meta.text = (
                    f"method={result.method} · {result.elapsed_ms:.0f} ms · "
                    f"confidence {result.confidence:.2f}"
                    + (f" · {citations}" if citations else "")
                )
                with vfm_citations_row:
                    for cite in result.citations[:6]:
                        citation_t = float(cite.get("t") or 0.0)
                        ui.button(
                            f"t={citation_t:.1f}s",
                            icon="play_arrow",
                            on_click=lambda t=citation_t: _seek_video(t),
                        ).props("flat dense size=xs no-caps").classes(
                            "text-[10px] chronicle-mono text-teal-300"
                        ).tooltip(str(cite.get("caption") or "")[:220])
                    for evidence in result.evidence[:1]:
                        rel = Path(str(evidence.get("rel_path") or ""))
                        if rel.name:
                            ui.image(
                                f"/vfm/evidence/{source.key}/{rel.name}"
                            ).classes("w-24 rounded").style(
                                "border:1px solid rgba(148,163,184,.25)"
                            )
            except Exception as exc:
                vfm_answer_label.text = f"{type(exc).__name__}: {exc}"
                vfm_answer_meta.text = (
                    "compile the memory first" if vfm.artifact(source, "vault.db") is None
                    else "query failed"
                )
            finally:
                vfm_ask_button.enable()

        def _seek_video(seconds: float) -> None:
            video_player.seek(max(0.0, float(seconds)))
            video_player.play()

        vfm_ask_button.on("click", lambda: _ask_vfm())
        vfm_ask_input.on("keydown.enter", lambda: _ask_vfm())

        async def _ask_from_entity_row(event: Any) -> None:
            """Clicking an Entities row asks where that object was last seen."""

            args = getattr(event, "args", None)
            row: Any = None
            if isinstance(args, (list, tuple)):
                row = next((item for item in args if isinstance(item, dict)), None)
            elif isinstance(args, dict):
                row = args
            label = str((row or {}).get("label") or "").strip()
            if not label:
                return
            query_input.value = f"where is the {label}?"
            await _ask()

        entity_table.on("rowClick", _ask_from_entity_row)
        _render_suggestions(initial.key, _ask)

        with state.lock:
            initial_runtime = state.runtime
        for event in initial_runtime.bus.history(limit=12):
            log_view.push(event.format())

        cursor = len(initial_runtime.bus.history())
        last_bus = id(initial_runtime.bus)
        last_graph_version: tuple[int, int, int, float] | None = None
        last_timeline_version: int | None = None
        last_video_key: tuple[str, int] | None = None
        last_llm_check = 0.0
        llm_status: dict[str, Any] = {}
        last_vfm_log: str | None = None
        last_ingest_signature: tuple[Any, ...] | None = None

        def refresh() -> None:
            nonlocal cursor, last_bus, last_graph_version, last_timeline_version
            nonlocal last_video_key, last_llm_check, llm_status
            nonlocal last_vfm_log, last_ingest_signature
            try:
                with state.lock:
                    runtime = state.runtime
                    running = state.running
                    error = state.error
                    summary = state.summary
                    source_key = state.source_key
                    snapshot = runtime.snapshot()
                stats = snapshot.get("stats", {})
                nodes = snapshot.get("nodes", [])
                edges = snapshot.get("edges", [])
                entities = snapshot.get("entities", [])
                actions = snapshot.get("actions", [])
                transitions = snapshot.get("transitions", [])

                status_chip.text = (
                    "RUNNING"
                    if running
                    else ("ERROR" if error else ("READY" if nodes else "IDLE"))
                )
                if error:
                    status_chip.props("color=negative")
                fps_chip.text = f"{float(stats.get('fps', 0) or 0):.1f} fps"
                camera_chip.text = f"CAM · {stats.get('camera_state', 'STATIONARY')}"

                stat_cards["entities"].text = str(len(entities))
                stat_cards["actions"].text = str(len(actions))
                stat_cards["transitions"].text = str(len(transitions))
                stat_cards["zones"].text = str(stats.get("zones", 0))
                stat_cards["furniture"].text = str(len(snapshot.get("furniture", [])))

                verification = snapshot.get("verification", {})
                q_depth = int(verification.get("depth", 0) or 0)
                q_dropped = int(verification.get("dropped", 0) or 0)
                q_verified = int(verification.get("verified", 0) or 0)
                if q_dropped > 0:
                    queue_chip.text = f"QUEUE · {q_depth} ({q_dropped} drop)"
                    queue_chip.props("color=warning")
                else:
                    queue_chip.text = f"QUEUE · {q_depth}"
                    queue_chip.props("color=positive" if q_depth == 0 else "color=info")
                queue_tip.text = (
                    f"Verification queue: {q_depth} pending · "
                    f"{q_verified} verified · {q_dropped} dropped"
                )

                now = __import__("time").monotonic()
                if now - last_llm_check > 5.0:
                    last_llm_check = now
                    llm_status = runtime.llm_status()
                if llm_status.get("available"):
                    loaded = llm_status.get("loaded") or []
                    if loaded:
                        gpu = int(loaded[0].get("gpu_percent") or 0)
                        llm_chip.text = f"LLM · {gpu}% GPU"
                        vram = float(loaded[0].get("size_vram") or 0) / 1e9
                        llm_tip.text = f"{llm_status.get('model')} · {vram:.1f} GB VRAM"
                    else:
                        llm_chip.text = "LLM · READY"
                        llm_tip.text = f"{llm_status.get('model')} · loads on next question"
                else:
                    llm_chip.text = "LLM · FALLBACK"
                    llm_tip.text = "Ollama offline · template GraphRAG · graph-grounded"

                video_path = Path(state.video)
                try:
                    video_stamp = video_path.stat().st_mtime_ns
                except OSError:
                    video_stamp = 0
                video_key = (str(video_path), video_stamp)
                if video_key != last_video_key:
                    last_video_key = video_key
                    video_player.set_source(_video_url(video_path))
                    video_player.update()
                    video_name_label.text = video_path.name
                source = source_map.get(source_key)
                if upload_busy["active"]:
                    video_badge.text = "UPLOADING"
                    video_caption_label.text = f"{upload_busy['name']} · uploading…"
                elif source is not None:
                    if source.key not in source_select.options:
                        source_select.options[source.key] = source.label
                        source_select.update()
                    suffix = "processing live" if running else "processed locally"
                    video_caption_label.text = f"{source.caption} · {suffix}"
                    video_badge.text = f"PRE-RECORDED · {source.badge}"
                if source_select.value != source_key:
                    source_sync["active"] = True
                    source_select.value = source_key
                    source_sync["active"] = False
                _render_suggestions(source_key, _ask)

                if vfm is not None and source is not None:
                    # Compile once per finished ingest.  The manager dedupes by
                    # DB/video signature, so several open tabs cannot fan out.
                    if summary and not running and not error:
                        ingest_signature = (
                            source_key,
                            summary.get("db_path"),
                            summary.get("frames"),
                            summary.get("actions"),
                        )
                        if ingest_signature != last_ingest_signature:
                            last_ingest_signature = ingest_signature
                            queued = vfm.submit(source, video_path)
                            if queued is not None and queued.status == "queued":
                                log_view.push("[VFM] memory compile queued after ingest")
                    vfm_snapshot = vfm.status(source)
                    if vfm_snapshot is None:
                        if vfm.has_memory(source):
                            vfm_status_badge.text = "COMPILED"
                            vfm_status_badge.props("color=positive")
                            vfm_stage_label.text = "ready (no job manifest)"
                            vfm_progress.set_value(1.0)
                            vault_path = vfm.artifact(source, "vault.db")
                            report_path = vfm.artifact(source, "memory.html")
                            vfm_stats_label.text = (
                                f"{vault_path.stat().st_size / 1024:.1f} KB vault · "
                                f"{(report_path.stat().st_size if report_path else 0) / 1024:.1f} KB report"
                            )
                            vfm_report_button.enable()
                        else:
                            vfm_status_badge.text = "NO MEMORY"
                            vfm_status_badge.props("color=grey")
                            vfm_stage_label.text = "not compiled"
                            vfm_progress.set_value(0.0)
                            vfm_stats_label.text = (
                                "Compile the memory from the current run, or upload a video "
                                "to compile it automatically."
                            )
                            vfm_report_button.disable()
                    else:
                        job_status = str(vfm_snapshot.get("status") or "idle")
                        job_stage = str(vfm_snapshot.get("stage") or "")
                        job_message = str(vfm_snapshot.get("message") or "")
                        job_progress = float(vfm_snapshot.get("progress") or 0) / 100.0
                        vfm_status_badge.text = job_status.upper()
                        colour = {
                            "done": "positive",
                            "error": "negative",
                            "running": "info",
                            "queued": "warning",
                        }.get(job_status, "grey")
                        vfm_status_badge.props(f"color={colour}")
                        vfm_stage_label.text = (
                            f"{job_stage} · {job_message}"[:96]
                            if job_message
                            else job_stage
                        )
                        vfm_progress.set_value(max(0.0, min(1.0, job_progress)))
                        job_summary = vfm_snapshot.get("summary") or {}
                        pack_stats = job_summary.get("pack") or {}
                        files = vfm_snapshot.get("files") or {}
                        if job_status == "done":
                            vfm_stats_label.text = (
                                f"{files.get('vault.db', 0) / 1024:.1f} KB vault · "
                                f"{files.get('memory.html', 0) / 1024:.1f} KB report · "
                                f"{pack_stats.get('approx_tokens', 0)} pack tokens · "
                                f"{pack_stats.get('windows', 0)} caption windows · "
                                f"{files.get('evidence', 0) / 1024:.1f} KB evidence"
                            )
                            vfm_report_button.enable()
                        elif job_status == "error":
                            vfm_stats_label.text = str(vfm_snapshot.get("error") or "compile failed")
                            vfm_report_button.disable()
                        else:
                            vfm_stats_label.text = f"{job_message or 'working'}…"
                            vfm_report_button.disable()
                        logs = vfm_snapshot.get("logs") or []
                        if logs:
                            latest = str(logs[-1])
                            if latest != last_vfm_log:
                                last_vfm_log = latest
                                log_view.push(f"[VFM] {latest}")

                pipeline_label.text = (
                    f"frames {stats.get('frames', 0)} · "
                    f"stage {stats.get('stage', 'idle')} · "
                    f"zone {stats.get('current_zone') or '-'} · "
                    f"elapsed {float(stats.get('elapsed', 0) or 0):.0f}s"
                )
                if summary:
                    pipeline_label.text += f" · db {Path(str(summary.get('db_path', ''))).name}"
                if error:
                    pipeline_label.text += f" · {error}"

                version = (
                    len(nodes),
                    len(edges),
                    len(actions),
                    round(float(stats.get("timestamp", 0) or 0), 1),
                )
                if version != last_graph_version:
                    last_graph_version = version
                    graph_chart.options.clear()
                    graph_chart.options.update(_graph_options(snapshot))
                    graph_chart.update()

                if len(transitions) != last_timeline_version:
                    last_timeline_version = len(transitions)
                    timeline_chart.options.clear()
                    timeline_chart.options.update(_timeline_options(transitions))
                    timeline_chart.update()

                entity_rows = [
                    {
                        "id": row.get("id"),
                        "label": row.get("label"),
                        "colour": row.get("colour") or "—",
                        "colour_hex": row.get("colour_hex") or "",
                        "appearance": row.get("appearance") or "—",
                        "state": row.get("state"),
                        "zone": row.get("zone"),
                        "anchor": row.get("anchor"),
                        "container": row.get("container"),
                        "span": row.get("span"),
                        "_search": " ".join(
                            str(row.get(key) or "")
                            for key in (
                                "label",
                                "colour",
                                "appearance",
                                "state",
                                "zone",
                                "anchor",
                                "container",
                            )
                        ).lower(),
                    }
                    for row in entities
                ]
                needle = str(entity_filter.value or "").strip().lower()
                if needle:
                    entity_rows = [
                        row for row in entity_rows if needle in row["_search"]
                    ]
                entity_table.rows = entity_rows
                entity_table.update()

                action_table.rows = [
                    {
                        "id": row.get("id"),
                        "time": f"{float(row.get('timestamp') or 0):.2f}s",
                        "action": row.get("action"),
                        "entity": row.get("entity"),
                        "caption": row.get("caption") or "",
                        "source": row.get("source") or "",
                    }
                    for row in actions
                ]
                action_table.update()

                if id(runtime.bus) != last_bus:
                    last_bus = id(runtime.bus)
                    cursor = 0
                history = runtime.bus.history()
                if cursor > len(history):
                    cursor = 0
                for event in history[cursor:]:
                    log_view.push(event.format())
                cursor = len(history)

                last_t = float(stats.get("timestamp", 0) or 0)
                video_time_label.text = f"last observation t={last_t:5.1f}s"
            except Exception as exc:  # pragma: no cover - defensive UI loop
                log_view.push(f"dashboard error: {type(exc).__name__}: {exc}")

        ui.timer(1.0, refresh)
        refresh()

    if start_server:
        ui.run(
            host=host,
            port=port,
            show=show,
            reload=False,
            title="Chronicle — episodic memory",
            favicon="🕸️",
        )
    return state


__all__ = ["create_app"]
