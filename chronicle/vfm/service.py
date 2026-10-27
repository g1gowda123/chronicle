"""Service layer for the video-free memory (VFM) integration.

The dashboard already owns the upload button, player and ingest runtime.  This
module turns a finished ingest into a small, downloadable memory:

    source DB snapshot -> SQLite vault -> dense caption ledger -> WebP evidence
    -> single-file HTML report -> LLM memory pack -> zip

It is deliberately UI-free so it can be tested headlessly and reused by the
NiceGUI dashboard, the CLI or a future worker process.  The source video is
never required for queries; it is only used once for captions and evidence.
"""

from __future__ import annotations

import csv
import json
import queue
import re
import shutil
import sqlite3
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import caption_ledger, memory_pack, nlq, query, video_vault
from .attach_ledger import attach as attach_ledger_file
from .compile import compile_vault
from .extract_evidence import extract as extract_evidence
from .report import build_report

DEFAULT_WINDOW_S = 30.0
DEFAULT_FRAMES_PER_WINDOW = 3
DEFAULT_MAX_SNAPSHOTS = 40
EVIDENCE_WIDTH = 480
EVIDENCE_QUALITY = 75
EVIDENCE_FORMAT = "webp"

DOWNLOAD_NAMES = (
    "memory.zip",
    "vault.db",
    "memory.html",
    "memory_pack.md",
    "memory_pack.json",
    "ledger.json",
    "events.csv",
    "windows.csv",
)

_LOGS_KEPT = 80


# --------------------------------------------------------------------------- paths
def memory_job_dir(data_dir: Path | str, source: Any) -> Path:
    """Return the stable VFM job directory for a source.

    Uploads keep their memory beside the source (``uploads/<run>/vfm``); bundled
    sources use ``<data-dir>/vfm/<source-key>`` so the repository data tree is
    never polluted.
    """

    data_root = Path(data_dir).expanduser().resolve()
    try:
        db_path = source.db_path(data_root)
    except AttributeError:
        db_path = Path(str(source)).expanduser().resolve()
    uploads = (data_root / "uploads").resolve()
    try:
        db_path.resolve().relative_to(uploads)
    except ValueError:
        pass
    else:
        return db_path.parent / "vfm"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(getattr(source, "key", "memory")))
    return data_root / "vfm" / (slug.strip(".-") or "memory")


def _safe_child(base: Path, name: str) -> Path | None:
    candidate = (base / Path(name).name).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _artifact_map(job_dir: Path) -> dict[str, Path]:
    evidence = job_dir / "evidence"
    return {
        "memory.zip": job_dir / "memory.zip",
        "vault.db": job_dir / "vault.db",
        "memory.html": job_dir / "memory.html",
        "memory_pack.md": job_dir / "memory_pack.md",
        "memory_pack.json": job_dir / "memory_pack.json",
        "ledger.json": job_dir / "ledger.json",
        "events.csv": job_dir / "events.csv",
        "windows.csv": job_dir / "windows.csv",
        "evidence": evidence,
    }


def artifacts(job_dir: Path | str) -> dict[str, Path]:
    """Return every existing artifact for a job directory."""

    return {name: path for name, path in _artifact_map(Path(job_dir)).items() if path.exists()}


def artifact(job_dir: Path | str, name: str) -> Path | None:
    allowed = _artifact_map(Path(job_dir))
    path = allowed.get(Path(name).name)
    return path if path is not None and path.is_file() else None


# --------------------------------------------------------------------------- compile
def _snapshot_db(source_db: Path, destination: Path) -> None:
    """Make a consistent copy of a live SQLite DB (WAL included)."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    uri = f"{source_db.resolve().as_uri()}?mode=ro"
    source = sqlite3.connect(uri, uri=True, timeout=30.0)
    target = sqlite3.connect(str(destination), timeout=30.0)
    try:
        source.backup(target)
        target.commit()
    finally:
        target.close()
        source.close()


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


def _zip_job(job_dir: Path, evidence_dir: Path | None) -> Path:
    zip_path = job_dir / "memory.zip"
    sources = _artifact_map(job_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("vault.db", "memory.html", "ledger.json", "memory_pack.md",
                     "memory_pack.json", "events.csv", "windows.csv"):
            path = sources[name]
            if path.is_file():
                archive.write(path, name)
        if evidence_dir is not None and evidence_dir.is_dir():
            for image in sorted(evidence_dir.glob("*")):
                if image.is_file():
                    archive.write(image, f"evidence/{image.name}")
        archive.writestr(
            "README.txt",
            "Chronicle VFM memory\n"
            "====================\n"
            "vault.db         SQLite queryable vault (no video)\n"
            "memory.html      single-file offline memory browser + query box\n"
            "ledger.json      dense timestamped VLM captions\n"
            "memory_pack.md   LLM-ready memory pack (captions, events, evidence refs)\n"
            "memory_pack.json same pack as structured JSON\n"
            "events.csv       packed event ledger export\n"
            "windows.csv      caption windows export\n"
            "evidence/        event WebP snapshots\n\n"
            "The source video is not included and is not required to query this memory.\n",
        )
    return zip_path


def _update_vault_flags(vault: Path, *, video_retained: bool) -> None:
    con = sqlite3.connect(str(vault))
    try:
        value = 1 if video_retained else 0
        con.execute("update episodes set video_retained=?", (value,))
        con.execute(
            "insert or replace into meta(key, value) values (?, ?)",
            ("video_retained", str(value)),
        )
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def compile_memory(
    video: Path | None,
    source_db: Path | str | None,
    job_dir: Path | str,
    *,
    data_root: Path | str | None = None,
    title: str | None = None,
    episode_id: str | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    frames_per_window: int = DEFAULT_FRAMES_PER_WINDOW,
    max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
    delete_video: bool = False,
    ledger_path: Path | str | None = None,
    captioner: Callable[..., dict[str, Any]] | None = None,
    vlm_model: str = caption_ledger.DEFAULT_MODEL,
    vlm_host: str = caption_ledger.DEFAULT_HOST,
    vlm_timeout: float = 300.0,
    synth_model: str = caption_ledger.DEFAULT_SYNTH_MODEL,
    frame_width: int = caption_ledger.FRAME_WIDTH,
    max_ledger_windows: int = 0,
    progress: Callable[[str, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile one video/DB pair into a downloadable memory.

    ``captioner`` is injectable for tests; it receives the same keyword
    arguments as :func:`caption_ledger.build_ledger`.
    """

    source_db = Path(source_db).expanduser().resolve() if source_db is not None else None
    job_dir = Path(job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video).expanduser().resolve() if video is not None else None
    if video_path is not None and not video_path.is_file():
        if log:
            log(f"video not found: {video_path}")
        video_path = None
    if source_db is not None and not source_db.is_file():
        if log:
            log(f"source database not found, using caption-first path: {source_db}")
        source_db = None
    if source_db is None and video_path is None:
        raise FileNotFoundError("a source database or a video is required")

    def _stage(stage: str, percent: int, message: str = "") -> None:
        if progress:
            progress(stage, percent, message)
        if message and log:
            log(message)

    source_copy: Path | None = None
    if source_db is not None:
        _stage("snapshot", 5, "snapshotting the run database")
        source_copy = job_dir / "source.db"
        _snapshot_db(source_db, source_copy)
    else:
        _stage("snapshot", 5, "no graph DB; caption-first memory")

    ledger_file: Path | None = None
    caption_summary: dict[str, Any] | None = None
    duration = 0.0
    if ledger_path is not None:
        candidate = Path(ledger_path).expanduser().resolve()
        if candidate.is_file():
            ledger_file = job_dir / "ledger.json"
            if candidate != ledger_file:
                shutil.copyfile(candidate, ledger_file)
            caption_summary = {"mode": "provided", "ledger": str(ledger_file)}
            _stage("captions", 20, f"using supplied ledger {candidate.name}")
        else:
            _stage("captions", 20, f"supplied ledger missing: {candidate}")

    if ledger_file is None and video_path is not None:
        _stage("captions", 15, f"captioning {window_s:g}s windows from the source video")
        builder = captioner or caption_ledger.build_ledger
        try:
            caption_summary = builder(
                video_path,
                job_dir / "ledger.json",
                model=vlm_model,
                host=vlm_host,
                window_s=window_s,
                frames_per_window=frames_per_window,
                width=frame_width,
                timeout=vlm_timeout,
                synth_model=synth_model,
                max_windows=max_ledger_windows,
            )
            if int(caption_summary.get("windows") or 0) > 0:
                ledger_file = job_dir / "ledger.json"
                _stage(
                    "captions",
                    28,
                    f"captions ready: {caption_summary['windows']} windows, "
                    f"{caption_summary.get('bytes', 0)} B",
                )
            else:
                _stage("captions", 28, "captioner returned no windows; using graph captions")
        except (Exception, SystemExit) as exc:
            _stage(
                "captions",
                28,
                f"captioning unavailable ({type(exc).__name__}: {exc}); using graph captions",
            )
    elif video_path is None:
        _stage("captions", 20, "no source video; compiling the verified graph memory only")

    _stage("vault", 40, "building the packed SQLite vault")
    vault = job_dir / "vault.db"
    if source_copy is not None:
        # ``compile_vault`` is a synced copy of the experiment module; keep our
        # call surface to its public ``(source_db, out_db, title=...)`` signature.
        vault_summary = compile_vault(source_copy, vault, title=title)
    elif ledger_file is not None and video_path is not None:
        vault_summary = video_vault.compile_caption_vault(
            video_path, ledger_file, vault, title=title or "Caption memory"
        )
    else:
        raise FileNotFoundError("no graph DB and no caption ledger")
    if episode_id:
        con = sqlite3.connect(str(vault))
        try:
            con.execute("update episodes set id=?", (str(episode_id),))
            con.commit()
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()
    duration = float(vault_summary.get("duration_s") or 0.0)
    _stage(
        "vault",
        52,
        f"vault ready: {vault_summary['events']} events, "
        f"{vault_summary['entities']} entities, {vault_summary['vault_bytes']} B",
    )

    if ledger_file is not None:
        attached = attach_ledger_file(vault, ledger_file)
        _stage(
            "ledger",
            58,
            f"attached {attached['windows']} caption windows "
            f"({attached['stored_bytes']} B compressed)",
        )

    evidence_dir = job_dir / "evidence"
    evidence_summary: dict[str, Any] = {"snapshots": 0, "snapshot_bytes": 0}
    if video_path is not None:
        _stage("evidence", 68, "extracting 480 px WebP snapshots")
        try:
            evidence_summary = extract_evidence(
                vault,
                video_path,
                evidence_dir,
                data_root=job_dir,
                delete_video=False,
                max_items=max_snapshots,
                width=EVIDENCE_WIDTH,
                quality=EVIDENCE_QUALITY,
                image_format=EVIDENCE_FORMAT,
                min_gap=2.0,
            )
            _stage(
                "evidence",
                76,
                f"evidence: {evidence_summary['snapshots']} snapshots, "
                f"{evidence_summary['snapshot_bytes']} B",
            )
        except Exception as exc:
            evidence_summary = {"snapshots": 0, "snapshot_bytes": 0, "error": str(exc)}
            _stage("evidence", 76, f"evidence skipped ({type(exc).__name__}: {exc})")

    _stage("report", 84, "rendering the offline HTML report")
    report_summary = build_report(vault, job_dir / "memory.html", data_root=job_dir, title=title)
    _write_csv(vault, job_dir)

    _stage("pack", 90, "writing the LLM memory pack")
    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        pack_md = memory_pack.build_pack(store)
        pack_json = memory_pack.build_pack_data(store)
        (job_dir / "memory_pack.md").write_text(pack_md, encoding="utf-8")
        (job_dir / "memory_pack.json").write_text(
            json.dumps(pack_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pack_stats = {
            "chars": len(pack_md),
            "approx_tokens": round(len(pack_md) / 4.0),
            "windows": len(store.ledger_windows),
            "events": len(store.events),
            "evidence": len(store.evidence),
        }
        _stage(
            "pack",
            94,
            f"memory pack: {pack_stats['chars']} chars "
            f"(~{pack_stats['approx_tokens']} tokens)",
        )
    finally:
        con.close()

    _stage("zip", 96, "packing downloads")
    _update_vault_flags(vault, video_retained=video_path is not None)
    archive = _zip_job(job_dir, evidence_dir if evidence_dir.is_dir() else None)

    deleted = False
    if delete_video and video_path is not None and video_path.is_file():
        if data_root is not None:
            uploads = (Path(data_root).expanduser().resolve() / "uploads").resolve()
        else:
            uploads = None
        # Only delete inside the uploads tree so bundled repository clips are
        # never removed.  Playback stays the default.
        allowed = False
        if uploads is not None:
            try:
                video_path.resolve().relative_to(uploads)
                allowed = True
            except ValueError:
                allowed = False
        if allowed:
            try:
                video_path.unlink()
                deleted = True
                _stage("done", 99, "source video deleted (opt-in)")
            except OSError as exc:
                _stage("done", 99, f"could not delete source video: {exc}")
        elif log:
            log("refusing to delete a non-upload video; playback is preserved")

    summary = {
        "status": "done",
        "source_db": str(source_db),
        "video": str(video_path) if video_path is not None else None,
        "job_dir": str(job_dir),
        "duration_s": round(duration, 3),
        "vault": str(vault),
        "report": str(job_dir / "memory.html"),
        "ledger": str(ledger_file) if ledger_file is not None else None,
        "pack_md": str(job_dir / "memory_pack.md"),
        "pack_json": str(job_dir / "memory_pack.json"),
        "events_csv": str(job_dir / "events.csv"),
        "windows_csv": str(job_dir / "windows.csv"),
        "archive": str(archive),
        "evidence": evidence_summary,
        "captions": caption_summary,
        "vault_stats": vault_summary,
        "report_stats": report_summary,
        "pack": pack_stats,
        "video_deleted": deleted,
        "video_retained": video_path is not None and not deleted,
    }
    _stage(
        "done",
        100,
        f"memory ready · vault {vault_summary['vault_bytes']/1024:.1f} KB · "
        f"report {int(report_summary.get('bytes') or 0)/1024:.1f} KB",
    )
    return summary


# --------------------------------------------------------------------------- jobs
@dataclass
class VFMJob:
    source_key: str
    source_db: Path | None
    video: Path | None
    job_dir: Path
    title: str
    signature: tuple[Any, ...]
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "queued"
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def _log(self, message: str) -> None:
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            self.logs = self.logs[-_LOGS_KEPT:]

    def _stage(self, stage: str, percent: int, message: str = "") -> None:
        with self.lock:
            self.stage = stage
            self.progress = max(0, min(100, int(percent)))
            if message:
                self.message = message
                self._log(message)
        self.write_manifest()

    def write_manifest(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        try:
            (self.job_dir / "job.json").write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def file_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for name, path in artifacts(self.job_dir).items():
            if path.is_dir():
                total = sum(child.stat().st_size for child in path.glob("*") if child.is_file())
                if total:
                    sizes[name] = total
            else:
                try:
                    sizes[name] = path.stat().st_size
                except OSError:
                    pass
        return sizes

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "source_key": self.source_key,
                "title": self.title,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "logs": list(self.logs[-40:]),
                "summary": self.summary,
                "files": self.file_sizes(),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "job_dir": str(self.job_dir),
            }


def load_status(job_dir: Path | str) -> dict[str, Any] | None:
    path = Path(job_dir) / "job.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("files", {})
    return payload


class VFMJobManager:
    """Run one VFM compile at a time and expose its status/artifacts."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        qwen_model: str = "qwen2.5:7b-instruct",
        ollama_url: str = caption_ledger.DEFAULT_HOST,
        qwen_timeout: float = 180.0,
        window_s: float = DEFAULT_WINDOW_S,
        frames_per_window: int = DEFAULT_FRAMES_PER_WINDOW,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        vlm_model: str = caption_ledger.DEFAULT_MODEL,
        synth_model: str = caption_ledger.DEFAULT_SYNTH_MODEL,
        vlm_timeout: float = 300.0,
        delete_video: bool = False,
        captioner: Callable[..., dict[str, Any]] | None = None,
        enabled: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.qwen_model = qwen_model
        self.ollama_url = ollama_url
        self.qwen_timeout = qwen_timeout
        self.window_s = float(window_s)
        self.frames_per_window = int(frames_per_window)
        self.max_snapshots = int(max_snapshots)
        self.vlm_model = vlm_model
        self.synth_model = synth_model
        self.vlm_timeout = float(vlm_timeout)
        self.delete_video = bool(delete_video)
        self.captioner = captioner
        self.enabled = bool(enabled)
        self._jobs: dict[str, VFMJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[VFMJob | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------
    def job_dir(self, source: Any) -> Path:
        return memory_job_dir(self.data_dir, source)

    def status(self, source: Any) -> dict[str, Any] | None:
        job_dir = self.job_dir(source)
        with self._lock:
            job = self._jobs.get(str(source.key))
        if job is not None:
            return job.to_dict()
        status = load_status(job_dir)
        if status is not None:
            status.setdefault("source_key", str(source.key))
        return status

    def files(self, source: Any) -> dict[str, int]:
        status = self.status(source) or {}
        files = status.get("files")
        return files if isinstance(files, dict) else {}

    def artifact(self, source: Any, name: str) -> Path | None:
        return artifact(self.job_dir(source), name)

    def evidence_file(self, source: Any, name: str) -> Path | None:
        return _safe_child(self.job_dir(source) / "evidence", name)

    def has_memory(self, source: Any) -> bool:
        return artifact(self.job_dir(source), "vault.db") is not None

    def submit(
        self,
        source: Any,
        video: Path | str | None = None,
        *,
        force: bool = False,
    ) -> VFMJob | None:
        if not self.enabled:
            return None
        raw_db = getattr(source, "db", None)
        source_db = Path(raw_db).expanduser().resolve() if raw_db is not None else None
        if source_db is not None and not source_db.is_file():
            source_db = None
        video_path = Path(video).expanduser().resolve() if video is not None else None
        if video_path is not None and not video_path.is_file():
            video_path = None
        if source_db is None and video_path is None:
            raise FileNotFoundError("source has neither a graph DB nor a video")
        signature = self._signature(source_db, video_path)
        job_dir = self.job_dir(source)
        with self._lock:
            current = self._jobs.get(str(source.key))
            if current is not None and not force:
                same = current.signature == signature
                if same and current.status in {"queued", "running"}:
                    return current
                if same and current.status == "done" and artifact(job_dir, "vault.db"):
                    return current
            if current is not None and current.status in {"queued", "running"}:
                current.cancel.set()
            job = VFMJob(
                source_key=str(source.key),
                source_db=source_db,
                video=video_path,
                job_dir=job_dir,
                title=str(getattr(source, "label", source.key)),
                signature=signature,
            )
            self._jobs[str(source.key)] = job
            job.write_manifest()
            self._queue.put(job)
        return job

    def wait(self, source: Any, timeout: float = 120.0) -> dict[str, Any] | None:
        """Block until the current job for ``source`` finishes (tests/demo)."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(source)
            if status is not None and status.get("status") in {"done", "error", "cancelled"}:
                return status
            time.sleep(0.05)
        return self.status(source)

    def ask(
        self,
        source: Any,
        question: str,
        *,
        use_qwen: bool = False,
        now: float | None = None,
    ) -> query.Answer:
        vault = artifact(self.job_dir(source), "vault.db")
        if vault is None:
            raise RuntimeError("memory is not compiled yet")
        con = sqlite3.connect(str(vault))
        con.row_factory = sqlite3.Row
        try:
            store = query.load_store(con)
            if use_qwen:
                return nlq.answer_nl(
                    store,
                    question,
                    model=self.qwen_model,
                    host=self.ollama_url,
                    timeout=self.qwen_timeout,
                )
            return query.answer(store, question, use_cache=True, now=now)
        finally:
            con.close()

    def close(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job.status in {"queued", "running"}:
                    job.cancel.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    # -- internals ------------------------------------------------------
    def _signature(self, source_db: Path | None, video: Path | None) -> tuple[Any, ...]:
        def _stamp(path: Path | None) -> tuple[int, int]:
            if path is None:
                return (0, 0)
            try:
                stat = path.stat()
            except OSError:
                return (0, 0)
            return (int(stat.st_size), int(stat.st_mtime_ns))

        return (
            str(source_db),
            *_stamp(source_db),
            str(video) if video is not None else "",
            *_stamp(video),
            self.window_s,
            self.frames_per_window,
            self.max_snapshots,
        )

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if job.cancel.is_set():
                with job.lock:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.finished_at = time.time()
                job.write_manifest()
                continue
            self._run(job)

    def _run(self, job: VFMJob) -> None:
        with job.lock:
            job.status = "running"
            job.stage = "snapshot"
            job.progress = 2
            job.started_at = time.time()
            job.message = "starting compile"
        job.write_manifest()

        def _progress(stage: str, percent: int, message: str) -> None:
            if job.cancel.is_set():
                raise RuntimeError("cancelled")
            job._stage(stage, percent, message)

        try:
            summary = compile_memory(
                job.video,
                job.source_db,
                job.job_dir,
                data_root=self.data_dir,
                title=job.title,
                episode_id=job.source_key,
                window_s=self.window_s,
                frames_per_window=self.frames_per_window,
                max_snapshots=self.max_snapshots,
                delete_video=self.delete_video,
                captioner=self.captioner,
                vlm_model=self.vlm_model,
                vlm_host=self.ollama_url,
                vlm_timeout=self.vlm_timeout,
                synth_model=self.synth_model,
                progress=_progress,
                log=job._log,
            )
            with job.lock:
                job.status = "done"
                job.stage = "done"
                job.progress = 100
                job.summary = {
                    "duration_s": summary.get("duration_s"),
                    "vault_bytes": (summary.get("vault_stats") or {}).get("vault_bytes"),
                    "pack": summary.get("pack"),
                    "evidence": summary.get("evidence"),
                    "video_retained": summary.get("video_retained"),
                }
                job.message = "memory ready"
                job.finished_at = time.time()
            job._log("memory ready")
        except Exception as exc:
            with job.lock:
                job.status = "error"
                job.stage = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                job.finished_at = time.time()
            job._log(f"ERROR: {job.error}")
        finally:
            job.write_manifest()


__all__ = [
    "DOWNLOAD_NAMES",
    "VFMJob",
    "VFMJobManager",
    "artifact",
    "artifacts",
    "compile_memory",
    "load_status",
    "memory_job_dir",
]
