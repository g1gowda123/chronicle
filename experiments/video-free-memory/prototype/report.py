#!/usr/bin/env python3
"""Render a single-file, offline HTML memory browser from a VFM vault.

The output has no video, no external JavaScript and no network dependencies.
It is the "memory as a product" artifact: entities, timeline, captions and
optional evidence thumbnails embedded as base64 JPEGs.

Usage:
    python prototype/report.py --db vault.db --out memory.html --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

from query import caption_for, load_store, stats

CSS = """
:root{--bg:#0b1020;--panel:#121a2e;--ink:#e2e8f0;--muted:#94a3b8;--accent:#2dd4bf;--line:#243044}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
header{padding:24px 28px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#0b1020,#101a33)}
h1{margin:0 0 4px;font-size:26px;letter-spacing:.02em}
.sub{color:var(--muted);font-size:13px}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin-top:16px}
.stat b{display:block;font-size:20px;color:var(--accent)}
.stat span{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
main{display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px;padding:20px 28px;max-width:1280px;margin:auto}
aside{position:sticky;top:20px;align-self:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}
.entity{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);cursor:pointer}
.entity:last-child{border:0}
.entity:hover{color:var(--accent)}
.pill{font-size:10px;padding:2px 6px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:#0a1020;color:var(--ink);margin-bottom:14px}
ol{list-style:none;margin:0;padding:0}
li{display:grid;grid-template-columns:78px minmax(0,1fr);gap:12px;padding:10px 0;border-bottom:1px solid var(--line);align-items:start}
li:hover{background:#0f1729}
time{color:var(--accent);font-variant-numeric:tabular-nums;font-size:13px}
.cap{font-size:14px}
.kind{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:10px}
.shot{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.shot img{width:100%;display:block;aspect-ratio:16/10;object-fit:cover}
.shot div{padding:6px 8px;font-size:11px;color:var(--muted)}
.note{color:var(--muted);font-size:12px;margin:10px 0}
@media(max-width:900px){main{grid-template-columns:1fr}aside{position:static}li{grid-template-columns:64px minmax(0,1fr)}}
"""

JS = """
const search=document.getElementById('search');
const rows=[...document.querySelectorAll('#timeline li')];
const entities=[...document.querySelectorAll('.entity')];
let active=null;
function apply(){
  const q=(search.value||'').toLowerCase();
  rows.forEach(li=>{
    const text=(li.dataset.text||'').toLowerCase();
    const okQ=!q||text.includes(q);
    const okE=!active||text.includes(active);
    li.style.display=(okQ&&okE)?'grid':'none';
  });
}
search.addEventListener('input',apply);
entities.forEach(el=>el.addEventListener('click',()=>{
  active=(active===el.dataset.label)?null:el.dataset.label;
  entities.forEach(x=>x.style.color=(x.dataset.label===active)?'var(--accent)':'');
  apply();
}));
apply();
"""


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _image_data_url(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > 400_000:
            return None
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except OSError:
        return None


def build_report(db: Path, out: Path, data_root: Path | None = None, *, title: str | None = None) -> dict[str, Any]:
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        store = load_store(con)
        info = stats(store, db)
        evidence: list[dict[str, Any]] = []
        try:
            evidence = [dict(row) for row in con.execute("select * from evidence order by t")]
        except sqlite3.Error:
            evidence = []
        episode_title = title
        try:
            row = con.execute("select title from episodes limit 1").fetchone()
            if row and not episode_title:
                episode_title = str(row["title"])
        except sqlite3.Error:
            pass
    finally:
        con.close()

    counts: dict[str, int] = {}
    for entity in store.entities:
        label = str(entity["label"])
        counts[label] = counts.get(label, 0) + 1

    cards = "".join(
        f'<div class="entity" data-label="{_esc(label)}"><span>{_esc(label)}</span>'
        f'<span class="pill">{count}</span></div>'
        for label, count in sorted(counts.items())
    )

    rows: list[str] = []
    for event in sorted(store.events, key=lambda row: float(row["t"]), reverse=True):
        caption = caption_for(event)
        rows.append(
            f'<li data-text="{_esc(caption + " " + str(event["subject"]))}">'
            f'<time>[t={float(event["t"]):.1f}s]</time>'
            f'<div><div class="cap">{_esc(caption)}</div>'
            f'<div class="kind">{_esc(event["kind"])}</div></div></li>'
        )

    gallery: list[str] = []
    for shot in sorted(evidence, key=lambda row: float(row["t"])):
        path = None
        if data_root is not None:
            path = (data_root / str(shot["rel_path"])).resolve()
        elif Path(str(shot["rel_path"])).is_file():
            path = Path(str(shot["rel_path"]))
        data_url = _image_data_url(path) if path is not None else None
        if not data_url:
            continue
        label = str(shot.get("label") or shot.get("kind") or "event")
        gallery.append(
            f'<figure class="shot"><img src="{data_url}" alt="{_esc(label)} at {shot["t"]}s" loading="lazy">'
            f'<div>[t={float(shot["t"]):.1f}s] {_esc(label)}</div></figure>'
        )

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(episode_title or "Chronicle memory")}</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>{_esc(episode_title or "Chronicle memory")}</h1>
  <div class="sub">Compiled memory · no video retained · {info.get("duration_s", 0):.1f}s episode</div>
  <div class="stats">
    <div class="stat"><b>{info.get("entities", 0)}</b><span>entities</span></div>
    <div class="stat"><b>{info.get("events", 0)}</b><span>events</span></div>
    <div class="stat"><b>{info.get("captions", 0)}</b><span>caption windows</span></div>
    <div class="stat"><b>{info.get("bytes", 0)/1024:.1f} KB</b><span>vault size</span></div>
    <div class="stat"><b>{len(gallery)}</b><span>snapshots</span></div>
    <div class="stat"><b>0</b><span>video retained</span></div>
  </div>
</header>
<main>
  <aside><div class="card"><h2>Entities</h2>{cards}</div>
  <div class="card" style="margin-top:14px"><h2>Notes</h2>
  <div class="note">This page is generated from the vault alone. Click an entity to filter.
  Evidence thumbnails are embedded JPEGs captured at event time.</div></div></aside>
  <section>
    <input id="search" placeholder="Search the memory…" autocomplete="off">
    <ol id="timeline">{"".join(rows)}</ol>
    <h2 style="font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 0">
      Evidence snapshots ({len(gallery)})</h2>
    <div class="gallery">{"".join(gallery) if gallery else '<div class="note">No snapshots were extracted for this vault.</div>'}</div>
  </section>
</main>
<script>{JS}</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return {
        "html": str(out),
        "bytes": out.stat().st_size,
        "events": len(store.events),
        "entities": len(store.entities),
        "evidence": len(evidence),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    summary = build_report(
        args.db.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.data_root.expanduser().resolve() if args.data_root else None,
        title=args.title,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
