#!/usr/bin/env python3
"""Chronicle VFM Studio: upload a video and download a compiled memory.

Local NiceGUI app:
  * upload/drop a video,
  * compile dense VLM captions + a SQLite vault + evidence WebP + HTML report,
  * download vault.db, memory.html, ledger.json, CSVs and a zip,
  * optionally answer natural-language questions with Qwen.

Run:
    ./run_studio.sh --qwen --port 8093
    ./run_studio.sh --no-qwen --port 8093

The deterministic query path is stdlib-only; this UI is the optional app layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from nicegui import app, run, ui

import caption_ledger
import memory_pack
import nlq
import query
import video_vault
from report import build_report

ROOT = Path(__file__).resolve().parent.parent
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
ARGS: argparse.Namespace


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(job: dict[str, Any], message: str) -> None:
    job["logs"].append(f"[{_now()}] {message}")
    job["logs"] = job["logs"][-80:]


def _stage(job: dict[str, Any], stage: str, progress: int, message: str | None = None) -> None:
    job["stage"] = stage
    job["progress"] = max(0, min(100, progress))
    if message:
        _log(job, message)


def _fallback_ledger(video: Path, out: Path) -> dict[str, Any]:
    duration = caption_ledger.probe_duration(video)
    windows = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + ARGS.window_s, duration)
        windows.append(
            {
                "t_start": round(start, 3),
                "t_end": round(end, 3),
                "caption": f"Video segment {index + 1}: {start:.1f}s to {end:.1f}s.",
                "objects": [],
                "actions": [],
                "zone": "unknown",
                "tags": [],
                "frames": [],
            }
        )
        start = end
        index += 1
    ledger = {
        "version": 1,
        "model": "fallback (VLM unavailable)",
        "mode": "fallback",
        "window_s": ARGS.window_s,
        "video": video.name,
        "duration_s": round(duration, 3),
        "generated_at": time.time(),
        "windows": windows,
    }
    out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def _write_csv(vault: Path, out_dir: Path) -> None:
    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "kind", "subject", "relation", "object", "zone", "caption"])
            for event in sorted(store.events, key=lambda row: float(row["t"])):
                writer.writerow(
                    [
                        f"{float(event['t']):.3f}",
                        event["kind"],
                        event["subject"],
                        event["relation"],
                        event["object"],
                        event["zone"],
                        query.caption_for(event),
                    ]
                )
        with (out_dir / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_start", "t_end", "caption", "objects", "actions", "zone", "tags"])
            for window in store.ledger_windows:
                writer.writerow(
                    [
                        window.get("t_start"),
                        window.get("t_end"),
                        window.get("caption") or "",
                        "|".join(str(value) for value in window.get("objects") or []),
                        "|".join(str(value) for value in window.get("actions") or []),
                        window.get("zone") or "",
                        "|".join(str(value) for value in window.get("tags") or []),
                    ]
                )
    finally:
        con.close()


def _zip_job(job: dict[str, Any], vault: Path, evidence_dir: Path | None) -> Path:
    zip_path = job["dir"] / "memory.zip"
    files = [
        (vault, "vault.db"),
        (job["dir"] / "memory.html", "memory.html"),
        (job["dir"] / "ledger.json", "ledger.json"),
        (job["dir"] / "memory_pack.md", "memory_pack.md"),
        (job["dir"] / "memory_pack.json", "memory_pack.json"),
        (job["dir"] / "events.csv", "events.csv"),
        (job["dir"] / "windows.csv", "windows.csv"),
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in files:
            if source.is_file():
                archive.write(source, name)
        if evidence_dir is not None and evidence_dir.is_dir():
            for image in sorted(evidence_dir.glob("*")):
                if image.is_file():
                    archive.write(image, f"evidence/{image.name}")
        archive.writestr(
            "README.txt",
            "Chronicle VFM memory\n"
            "====================\n"
            "vault.db      SQLite queryable vault (no video)\n"
            "memory.html   single-file offline memory browser + query box\n"
            "ledger.json   dense timestamped VLM captions\n"
            "memory_pack.md   LLM-ready memory pack (captions, events, evidence refs)\n"
            "memory_pack.json same pack as structured JSON\n"
            "events.csv    packed event ledger export\n"
            "windows.csv   caption windows export\n"
            "evidence/     event WebP snapshots\n\n"
            "The source video is not included and is not required to query this memory.\n",
        )
    return zip_path


def process_job(job: dict[str, Any]) -> None:
    try:
        job_dir = job["dir"]
        video = Path(job["video"])
        _stage(job, "probe", 5, "probing video")
        duration = caption_ledger.probe_duration(video)
        job["duration_s"] = round(duration, 3)
        _stage(job, "captions", 12, f"{duration:.1f}s video · captioning 30s windows")
        ledger_path = job_dir / "ledger.json"
        try:
            summary = caption_ledger.build_ledger(
                video,
                ledger_path,
                model=ARGS.vlm_model,
                host=ARGS.ollama_url,
                window_s=ARGS.window_s,
                frames_per_window=ARGS.frames_per_window,
                width=ARGS.frame_width,
                timeout=ARGS.vlm_timeout,
            )
            job["ledger"] = summary
            _log(
                job,
                f"captions ready: {summary['windows']} windows, "
                f"{summary['bytes']} B ledger, {summary['empty_windows']} empty",
            )
        except Exception as exc:
            _log(job, f"VLM captioning failed ({type(exc).__name__}: {exc}); using placeholders")
            _fallback_ledger(video, ledger_path)
        _stage(job, "vault", 55, "building SQLite vault")
        vault = job_dir / "vault.db"
        summary = video_vault.compile_caption_vault(
            video,
            ledger_path,
            vault,
            title=video.stem,
        )
        job["vault"] = summary
        _log(
            job,
            f"vault ready: {summary['events']} events, {summary['entities']} caption entities, "
            f"{summary['vault_bytes']} B",
        )

        evidence_dir = job_dir / "evidence"
        _stage(job, "evidence", 70, "extracting 480px WebP snapshots")
        try:
            from extract_evidence import extract

            evidence = extract(
                vault,
                video,
                evidence_dir,
                data_root=job_dir,
                delete_video=False,
                max_items=ARGS.max_snapshots,
                width=480,
                quality=75,
                image_format="webp",
                min_gap=2.0,
            )
            job["evidence"] = evidence
            _log(job, f"evidence: {evidence['snapshots']} snapshots, {evidence['snapshot_bytes']} B")
        except Exception as exc:
            job["evidence"] = {"snapshots": 0, "snapshot_bytes": 0, "error": str(exc)}
            _log(job, f"evidence skipped: {type(exc).__name__}: {exc}")

        _stage(job, "report", 85, "rendering offline HTML report")
        report_path = job_dir / "memory.html"
        report = build_report(vault, report_path, data_root=job_dir, title=video.stem)
        job["report"] = report
        _write_csv(vault, job_dir)
        con = sqlite3.connect(str(vault))
        try:
            store = query.load_store(con)
            job["timeline"] = [
                {
                    "t": round(float(event["t"]), 3),
                    "kind": str(event["kind"]),
                    "label": str(event["subject"] or event["kind"]),
                    "caption": query.caption_for(event),
                }
                for event in sorted(store.events, key=lambda row: float(row["t"]))[:120]
            ]
            job["pack_chars"] = len(memory_pack.build_pack(store, max_chars=14000))
            pack_md = memory_pack.build_pack(store)
            pack_json = memory_pack.build_pack_data(store)
            (job_dir / "memory_pack.md").write_text(pack_md, encoding="utf-8")
            (job_dir / "memory_pack.json").write_text(
                json.dumps(pack_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            job["pack"] = {
                "chars": len(pack_md),
                "approx_tokens": round(len(pack_md) / 4.0),
                "windows": len(store.ledger_windows),
                "events": len(store.events),
            }
            _log(
                job,
                f"LLM memory pack: {job['pack']['chars']} chars, "
                f"~{job['pack']['approx_tokens']} tokens",
            )
        finally:
            con.close()
        _stage(job, "zip", 93, "packing downloads")
        zip_path = _zip_job(job, vault, evidence_dir if evidence_dir.is_dir() else None)
        job["files"] = {
            "vault.db": vault,
            "memory.html": report_path,
            "ledger.json": ledger_path,
            "memory_pack.md": job_dir / "memory_pack.md",
            "memory_pack.json": job_dir / "memory_pack.json",
            "events.csv": job_dir / "events.csv",
            "windows.csv": job_dir / "windows.csv",
            "memory.zip": zip_path,
        }
        if ARGS.delete_video:
            try:
                video.unlink()
                _log(job, "source video deleted after compilation")
            except OSError:
                pass
        else:
            _log(job, "source video retained for playback; downloads stay video-free")
        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = 100
        _log(
            job,
            f"done · vault {summary['vault_bytes']/1024:.1f} KB · "
            f"report {report['bytes']/1024:.1f} KB · "
            f"evidence {job['evidence'].get('snapshot_bytes', 0)/1024:.1f} KB",
        )
    except Exception as exc:
        job["status"] = "error"
        job["stage"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        _log(job, f"ERROR: {job['error']}")
    finally:
        job["finished_at"] = time.time()
        try:
            (job["dir"] / "job.json").write_text(
                json.dumps(
                    {
                        "id": job["id"],
                        "filename": job["filename"],
                        "status": job["status"],
                        "stage": job["stage"],
                        "duration_s": job.get("duration_s"),
                        "logs": job["logs"],
                        "error": job.get("error"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name, path in (job.get("files") or {}).items():
        try:
            files[name] = path.stat().st_size
        except OSError:
            continue
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "duration_s": job.get("duration_s"),
        "logs": job["logs"][-30:],
        "error": job.get("error"),
        "files": files,
    }


def _safe_child(base: Path, name: str) -> Path | None:
    candidate = (base / Path(name).name).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_ui() -> None:
    ui.dark_mode().enable()
    ui.colors(primary="#2dd4bf", secondary="#38bdf8", accent="#f59e0b")
    ui.add_head_html(
        "<style>"
        "body{background-color:#070b11;color:#dbe4ee;}"
        "body::before{content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;"
        "background:radial-gradient(900px 480px at 18% -12%,rgba(45,212,191,.08),transparent 62%);}"
        ".chronicle-card{background:rgba(13,20,30,.84);border:1px solid rgba(148,163,184,.14);"
        "border-radius:14px;overflow:hidden;}"
        ".chronicle-mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
        ".chronicle-title{font-size:11px;text-transform:uppercase;letter-spacing:.18em;color:#64748b;}"
        ".chronicle-dropzone .q-uploader{width:100%;background:rgba(2,6,13,.35);"
        "border:1px dashed rgba(45,212,191,.42);border-radius:10px;box-shadow:none;}"
        ".chronicle-dropzone .q-uploader__header{background:transparent!important;"
        "color:#94a3b8!important;}"
        ".chronicle-dropzone .q-uploader__list{background:transparent!important;color:#cbd5e1!important;}"
        ".chronicle-dropzone .q-uploader__header .q-btn{color:#2dd4bf!important;}"
        ".studio-progress .q-linear-progress__track{background:rgba(148,163,184,.14);}"
        ".studio-answer{white-space:pre-wrap;line-height:1.5;}"
        ".studio-cite{color:#94a3b8;font-size:12px;}"
        "</style>"
    )

    with ui.header().classes("items-center px-6 py-2").style(
        "background:rgba(7,11,17,.92);border-bottom:1px solid rgba(148,163,184,.12)"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("hub", size="22px").classes("text-teal-300")
            ui.label("Chronicle Studio").classes("text-base font-semibold tracking-wide")
            ui.label("video → memory").classes(
                "text-[10px] uppercase tracking-[0.18em] text-slate-500"
            )
        with ui.row().classes("items-center gap-2").style("margin-left:auto"):
            ui.badge("VFM", color="teal").props("outline").classes("text-xs chronicle-mono")
            ui.badge(
                "QWEN · MEMORY ONLY" if ARGS.qwen else "QWEN · OFF",
                color="teal" if ARGS.qwen else "grey",
            ).props("outline").classes("text-xs chronicle-mono")
            ui.badge(ARGS.vlm_model, color="grey").props("outline").classes("text-xs chronicle-mono")
            ui.label(f":{ARGS.port}").classes("text-[10px] chronicle-mono text-slate-500")

    state: dict[str, Any] = {"job": None}

    def refresh() -> None:
        job = state["job"]
        if job is None:
            return
        status_label.text = job["status"].upper()
        stage_label.text = job["stage"]
        progress.set_value(job["progress"] / 100.0)
        progress_label.text = f"{job['progress']}%"
        log_box.clear()
        with log_box:
            for line in job["logs"][-16:]:
                ui.label(line).classes("text-[11px] chronicle-mono text-slate-400 whitespace-pre-wrap")
        if job["status"] == "done" and not state.get("preview_loaded"):
            preview.clear()
            with preview:
                ui.html(
                    f'<iframe src="/jobs/{job["id"]}/memory.html" '
                    'style="width:100%;height:520px;border:1px solid rgba(148,163,184,.18);'
                    'border-radius:12px;background:#0b1020"></iframe>'
                )
            state["preview_loaded"] = True
            downloads.clear()
            with downloads:
                for name in (
                    "memory.zip",
                    "vault.db",
                    "memory.html",
                    "memory_pack.md",
                    "memory_pack.json",
                    "ledger.json",
                    "events.csv",
                    "windows.csv",
                ):
                    if name in (job.get("files") or {}):
                        ui.button(
                            name,
                            icon="download",
                            on_click=lambda name=name, job=job: ui.download(
                                f"/jobs/{job['id']}/download/{name}"
                            ),
                        ).props("outline dense no-caps").classes("text-xs chronicle-mono")
                ui.label(
                    f"{job.get('duration_s', 0):.1f}s · source kept for playback · "
                    f"LLM pack {job.get('pack_chars', 0)/1024:.1f} KB text · 0 frames"
                ).classes("text-[11px] chronicle-mono text-slate-500")
            if not state.get("timeline_loaded"):
                render_timeline(job)
                state["timeline_loaded"] = True

    def render_video(job: dict[str, Any]) -> None:
        video_box.clear()
        with video_box:
            ui.html(
                f'<video id="studio-video" controls playsinline preload="metadata" '
                f'src="/jobs/{job["id"]}/video" '
                'style="width:100%;max-height:540px;background:#000;border-radius:12px;'
                'border:1px solid rgba(148,163,184,.18)"></video>'
                f'<div class="text-[10px] chronicle-mono text-slate-500 mt-1">'
                f'{job["filename"]} · human playback only; Qwen receives ledger text, not frames.</div>'
            )
        timeline_box.clear()
        state["timeline_loaded"] = False

    def render_timeline(job: dict[str, Any]) -> None:
        timeline_box.clear()
        with timeline_box:
            ui.label("EVENT TIMELINE").classes("chronicle-title")
            with ui.row().classes("w-full gap-1 flex-wrap"):
                for event in (job.get("timeline") or [])[:70]:
                    t = float(event.get("t") or 0.0)
                    label = str(event.get("label") or event.get("kind") or "")
                    ui.button(
                        f"[{t:.1f}s] {label}",
                        on_click=lambda t=t: ui.run_javascript(
                            f'document.getElementById("studio-video").currentTime={t};'
                            f'document.getElementById("studio-video").play();'
                        ),
                    ).props("flat dense no-caps").classes(
                        "text-[10px] chronicle-mono text-slate-400"
                    ).tooltip(str(event.get("caption") or ""))

    async def on_upload(event: Any) -> None:
        if state.get("busy"):
            ui.notify("A conversion is already running.", type="warning")
            upload.reset()
            return
        name = Path(str(event.file.name)).name
        suffix = Path(name).suffix.lower() or ".mp4"
        job_id = uuid.uuid4().hex[:12]
        job_dir = ARGS.data_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / f"source{suffix}"
        await event.file.save(target)
        job = {
            "id": job_id,
            "filename": name,
            "video": str(target),
            "dir": job_dir,
            "status": "processing",
            "stage": "queued",
            "progress": 2,
            "logs": [f"[{_now()}] uploaded {name} ({target.stat().st_size/1048576:.1f} MB)"],
            "files": {},
            "created_at": time.time(),
        }
        with LOCK:
            JOBS[job_id] = job
        state["job"] = job
        state["busy"] = True
        state["preview_loaded"] = False
        state["timeline_loaded"] = False
        preview.clear()
        downloads.clear()
        render_video(job)
        upload.reset()
        status_label.text = "PROCESSING"
        stage_label.text = "queued"
        ui.notify(f"Processing {name}", type="positive")
        refresh()
        threading.Thread(target=process_job, args=(job,), daemon=True).start()

    def _set_busy_done() -> None:
        job = state["job"]
        if job and job["status"] in {"done", "error"}:
            state["busy"] = False

    async def ask_question() -> None:
        job = state["job"]
        if not job or job["status"] != "done":
            ui.notify("Convert a video first.", type="warning")
            return
        text = (ask_input.value or "").strip()
        if not text:
            return
        ask_button.disable()
        answer_box.clear()
        with answer_box:
            ui.label("thinking…").classes("text-xs chronicle-mono text-slate-500")
        try:
            vault = job["dir"] / "vault.db"
            con = sqlite3.connect(str(vault))
            try:
                store = query.load_store(con)
                if ARGS.qwen:
                    result = await run.io_bound(
                        nlq.answer_nl,
                        store,
                        text,
                        model=ARGS.qwen_model,
                        host=ARGS.ollama_url,
                        timeout=ARGS.qwen_timeout,
                    )
                else:
                    result = await run.io_bound(query.answer, store, text, use_cache=True)
            finally:
                con.close()
            answer_box.clear()
            with answer_box:
                ui.label(result.answer).classes("text-sm studio-answer text-slate-200")
                ui.label(
                    f"{result.method} · {result.elapsed_ms:.1f} ms · "
                    f"confidence {result.confidence:.2f}"
                ).classes("text-[10px] chronicle-mono text-slate-500")
                for cite in result.citations[:6]:
                    ui.label(
                        f"[t={cite.get('t')}s] {str(cite.get('caption') or '')[:180]}"
                    ).classes("studio-cite chronicle-mono")
                for evidence in result.evidence[:1]:
                    rel = Path(str(evidence.get("rel_path") or ""))
                    if rel.name:
                        ui.html(
                            f'<img src="/jobs/{job["id"]}/evidence/{rel.name}" '
                            'style="max-width:220px;border:1px solid rgba(148,163,184,.2);'
                            'border-radius:8px;margin-top:8px">'
                        )
        except Exception as exc:
            answer_box.clear()
            with answer_box:
                ui.label(f"Query failed: {type(exc).__name__}: {exc}").classes(
                    "text-sm text-rose-300"
                )
        finally:
            ask_button.enable()

    with ui.column().classes("w-full max-w-[1440px] mx-auto px-5 pt-3 gap-3"):
        with ui.card().props("flat").classes("chronicle-card p-4 w-full"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("PLAYBACK").classes("chronicle-title")
                ui.label("human playback · LLM context is text-only").classes(
                    "text-[10px] chronicle-mono text-slate-500"
                )
            video_box = ui.column().classes("w-full gap-1")
            with video_box:
                ui.label("Upload a video to preview it here.").classes("text-xs text-slate-500")
            timeline_box = ui.column().classes("w-full gap-1 mt-2")
        with ui.row().classes("w-full items-stretch gap-3 flex-nowrap"):
            with ui.card().props("flat").classes("chronicle-card p-4").style(
                "flex:5 1 0;min-width:0"
            ):
                ui.label("UPLOAD").classes("chronicle-title")
                ui.label("Drop a video. The compiler reads frames once, captions 30s windows, "
                         "builds a SQLite vault, extracts WebP evidence, renders an offline report "
                         "and writes an LLM-ready memory pack. Qwen later answers from that "
                         "compiled text, never from the video.").classes(
                    "text-xs text-slate-400"
                )
                with ui.card().props("flat").classes("chronicle-dropzone w-full q-pa-none").style(
                    "background:transparent;border:0"
                ):
                    upload = ui.upload(
                        label="Choose or drop a video",
                        on_upload=on_upload,
                        auto_upload=True,
                        max_file_size=ARGS.max_upload_gb * 1024**3,
                    ).props("accept=video/* flat dense").classes("w-full")
                ui.label(
                    f"MP4 / MOV / WebM / MKV · up to {ARGS.max_upload_gb:g} GB · "
                    f"VLM {ARGS.vlm_model}"
                ).classes("text-[10px] chronicle-mono text-slate-600")
            with ui.card().props("flat").classes("chronicle-card p-4").style(
                "flex:7 1 0;min-width:0"
            ):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("CONVERSION").classes("chronicle-title")
                    status_label = ui.badge("IDLE", color="grey").props("outline").classes(
                        "text-[10px] chronicle-mono"
                    )
                    stage_label = ui.label("—").classes("text-[10px] chronicle-mono text-slate-500")
                    progress_label = ui.label("0%").classes("text-[10px] chronicle-mono text-slate-500")
                with ui.element("div").classes("w-full studio-progress"):
                    progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
                log_box = ui.column().classes("w-full gap-0 mt-2").style(
                    "max-height:150px;overflow:auto"
                )
                downloads = ui.row().classes("w-full gap-2 flex-wrap mt-2")
                preview = ui.element("div").classes("w-full")
        with ui.card().props("flat").classes("chronicle-card p-4 w-full"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("ASK THE MEMORY").classes("chronicle-title")
                ui.label(
                    f"natural language · {ARGS.qwen_model} · reads captions + events only"
                    if ARGS.qwen
                    else "deterministic · qwen off"
                ).classes("text-[10px] chronicle-mono text-slate-500")
            with ui.row().classes("w-full items-center gap-2"):
                ask_input = ui.input(
                    placeholder="where is the cup? / what happened around 20s? / what did I do with the tomato?"
                ).props("outlined dense dark").classes("flex-1")
                ask_button = ui.button("Ask", icon="search", on_click=ask_question).props(
                    "outline dense no-caps"
                ).classes("text-xs chronicle-mono")
            answer_box = ui.column().classes("w-full gap-1 mt-2")
            with answer_box:
                ui.label(
                    "Convert a video, then ask. " + ("Qwen is enabled." if ARGS.qwen else "Qwen is off.")
                ).classes("text-xs text-slate-500")

    ui.timer(0.7, refresh)
    ui.timer(0.7, _set_busy_done)


@app.get("/jobs/{job_id}/memory.html")
def job_memory(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        return FileResponse("/nonexistent", status_code=404)
    path = job["dir"] / "memory.html"
    return FileResponse(path) if path.is_file() else FileResponse("/nonexistent", status_code=404)


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        return FileResponse("/nonexistent", status_code=404)
    path = Path(job["video"])
    return FileResponse(path) if path.is_file() else FileResponse("/nonexistent", status_code=404)


@app.get("/jobs/{job_id}/download/{name}")
def job_download(job_id: str, name: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        return FileResponse("/nonexistent", status_code=404)
    path = _safe_child(job["dir"], name)
    if path is None:
        return FileResponse("/nonexistent", status_code=404)
    return FileResponse(path, filename=path.name)


@app.get("/jobs/{job_id}/evidence/{name}")
def job_evidence(job_id: str, name: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        return FileResponse("/nonexistent", status_code=404)
    path = _safe_child(job["dir"] / "evidence", name)
    if path is None:
        return FileResponse("/nonexistent", status_code=404)
    return FileResponse(path)


def main(argv: list[str] | None = None) -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--qwen", action="store_true", help="enable Qwen natural-language answers")
    parser.add_argument("--qwen-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--vlm-model", default="qwen2.5vl:3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--frames-per-window", type=int, default=3)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--max-snapshots", type=int, default=40)
    parser.add_argument("--max-upload-gb", type=float, default=2.0)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/vfm/studio"))
    parser.add_argument(
        "--delete-video",
        action="store_true",
        help="delete the uploaded video after compilation (no playback)",
    )
    parser.add_argument("--vlm-timeout", type=float, default=300.0)
    parser.add_argument("--qwen-timeout", type=float, default=180.0)
    parser.add_argument("--show", action="store_true", help="open a browser tab")
    args = parser.parse_args(argv)
    ARGS = args
    ARGS.data_dir = args.data_dir.expanduser().resolve()
    ARGS.data_dir.mkdir(parents=True, exist_ok=True)
    ui.run(
        build_ui,
        host=args.host,
        port=args.port,
        title="Chronicle Studio",
        dark=True,
        reload=False,
        show=args.show,
        favicon="🧠",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
