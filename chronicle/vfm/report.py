#!/usr/bin/env python3
"""Render a single-file, offline HTML memory browser from a VFM vault.

The output has no video, no external JavaScript and no network dependencies.
It includes a client-side query box that implements the same intents as
`query.py` (where / when / around / before / after / count / inventory /
keyword), plus the dense caption ledger and embedded event snapshots.

Usage:
    python prototype/report.py --db vault.db --out memory.html --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sqlite3
from pathlib import Path
from typing import Any

from .query import caption_for, load_store, stats

CSS = """
:root{--bg:#0b1020;--panel:#121a2e;--ink:#e2e8f0;--muted:#94a3b8;--accent:#2dd4bf;--line:#243044;--warn:#fbbf24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
header{padding:24px 28px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#0b1020,#101a33)}
h1{margin:0 0 4px;font-size:26px;letter-spacing:.02em}
.sub{color:var(--muted);font-size:13px}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin-top:16px}
.stat b{display:block;font-size:20px;color:var(--accent)}
.stat span{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
main{display:grid;grid-template-columns:300px minmax(0,1fr);gap:20px;padding:20px 28px;max-width:1280px;margin:auto}
aside{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}
.entity{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);cursor:pointer}
.entity:last-child{border:0}
.entity:hover{color:var(--accent)}
.pill{font-size:10px;padding:2px 6px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:#0a1020;color:var(--ink);margin-bottom:14px}
button{padding:10px 14px;border-radius:8px;border:1px solid var(--line);background:#17243d;color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
.answer{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:14px;margin-bottom:18px;min-height:64px}
.answer .a-text{white-space:pre-line;font-size:15px}
.answer .a-meta{color:var(--muted);font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.06em}
.answer .a-cites{margin-top:10px;display:flex;flex-direction:column;gap:4px}
.answer .a-cites span{color:var(--muted);font-size:12px}
.answer .a-evidence{margin-top:12px;display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
.answer .a-evidence img{width:180px;border-radius:6px;border:1px solid var(--line);display:block}
.answer .a-evidence div{color:var(--muted);font-size:12px;max-width:340px}
.window{border-bottom:1px solid var(--line);padding:7px 0}
.window:last-child{border:0}
.window b{color:var(--accent);font-size:12px;font-variant-numeric:tabular-nums}
.window p{margin:3px 0 0;font-size:13px}
.window .tags{color:var(--muted);font-size:11px;margin-top:3px}
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
@media(max-width:900px){main{grid-template-columns:1fr}aside{position:static;max-height:none}li{grid-template-columns:64px minmax(0,1fr)}}
"""

JS = """
const DATA=JSON.parse(document.getElementById('memory-data').textContent);
const search=document.getElementById('search');
const ask=document.getElementById('ask');
const askBtn=document.getElementById('ask-btn');
const answerBox=document.getElementById('answer');
const rows=[...document.querySelectorAll('#timeline li')];
const entities=[...document.querySelectorAll('.entity')];
let active=null;
const cache=new Map();
const STOP=new Set(('the a an is was were are am did do does what where when which who how many much my i me in on at to of for with and or this that it there be been being have has had can could would should about around last first before after then tell show memory').split(' '));
const SYN={mug:'cup',glass:'cup',glasses:'cup',dish:'plate',dishes:'plate',utensil:'knife',utensils:'knife',knives:'knife',knifes:'knife',people:'person',persons:'person',friend:'person',user:'person',rucksack:'backpack',bag:'backpack'};
const ACTIONS={touch:['touch','touched','touching','pick','picked','grab','grabbed','hold','held','handle','handled','use','used'],place:['place','placed','put','putting','set','drop','dropped','insert','inserted'],see:['see','saw','seen','appear','appeared'],lost:['lose','lost','disappeared']};
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function toks(q){const out=new Set(); for(const t of (q.toLowerCase().match(/[a-z0-9]+/g)||[])){const v=SYN[t]||t; out.add(v); if(v.endsWith('s')&&v.length>3)out.add(v.slice(0,-1));} return out;}
function singular(t){return t.endsWith('ies')&&t.length>4?t.slice(0,-3)+'y':(t.endsWith('s')&&t.length>3?t.slice(0,-1):t);}
function matchEntity(q){const ts=toks(q), low=q.toLowerCase(); let best=null; for(const e of DATA.entities){const label=e.label.toLowerCase(); let score=0; if(label&&new RegExp('\\\\b'+label.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&')+'\\\\b').test(low))score+=3; for(const lt of toks(label))if(ts.has(lt))score+=2; if(!score&&(ts.has(singular(label))||[...ts].some(x=>singular(x)===singular(label))))score+=1.5; if(score<=0)continue; if(['VISIBLE_STATIC','HELD','CONTAINED'].includes(e.state))score+=0.1; if(!best||score>best.score||(score===best.score&&(+e.last_seen||0)>(+best.entity.last_seen||0)))best={score,entity:e};} return best?best.entity:null;}
function action(q){const ts=toks(q); for(const [k,words] of Object.entries(ACTIONS))if(words.some(w=>ts.has(w)))return k; return '';}
function timeHint(q){const m=q.toLowerCase().match(/(?:at|around|near|after|before|t\\s*=)?\\s*(\\d+(?:\\.\\d+)?)\\s*(s|sec|secs|seconds?|m|min|mins|minutes?)\\b/); if(!m)return null; let v=parseFloat(m[1]); if(/^(m|min|mins|minutes)$/.test(m[2]))v*=60; return v;}
function timeHintLoose(q){const m=q.toLowerCase().match(/(?:at|around|near)\\s+(\\d+(?:\\.\\d+)?)\\b/); return m?parseFloat(m[1]):null;}
function eventsFor(label){return DATA.events.filter(e=>e.subject.toLowerCase()===label.toLowerCase());}
function matchesAction(e,a){if(!a)return true; const r=e.relation.toUpperCase(), k=e.kind; if(a==='touch')return k==='interaction'||r==='TOUCHED'||r==='HELD'; if(a==='place')return k==='anchor'&&['PLACED_ON','PLACED_IN','RESTING_ON','INSIDE'].includes(r); if(a==='see')return r==='APPEARED'||r==='VISIBLE_STATIC'; if(a==='lost')return r==='LOST'; return true;}
function latest(rows){return rows.length?rows.reduce((a,b)=>(+a.t>=+b.t?a:b)):null;}
function nearestEvidence(t){if(!DATA.evidence.length)return null; let best=null; for(const e of DATA.evidence){const d=Math.abs(+e.t-t); if(!best||d<best.d)best={...e,d};} return best;}
function evidenceHtml(t){const e=nearestEvidence(t); if(!e)return ''; const src=DATA.srcByRel[e.rel_path]; if(!src)return ''; return `<div class="a-evidence"><img src="${src}" alt="evidence at ${e.t}s"><div><b>[t=${(+e.t).toFixed(1)}s]</b> ${esc(e.label||e.kind)}<br>Nearest stored snapshot (${e.d.toFixed(1)}s away). No video was used.</div></div>`;}
function windowAt(t){let best=null; for(const w of DATA.windows){if(+w.t_start<=t&&t<=+w.t_end)return w; const d=Math.abs((+w.t_start + +w.t_end)/2-t); if(!best||d<best.d)best={w,d};} return best?best.w:null;}
function cite(t,caption){return `<span>[t=${(+t).toFixed(1)}s] ${esc(caption)}</span>`;}
function canonical(intent,e,t,a){return [intent,e?e.id:'-',t==null?'-':t.toFixed(1),a||'any'].join('|');}
function render(text,cites,meta,evidenceT,method){cached=false; answerBox.innerHTML=`<div class="a-text">${esc(text)}</div><div class="a-meta">${esc(method||'vault')} · ${esc(meta||'')}</div>${cites&&cites.length?`<div class="a-cites">${cites.join('')}</div>`:''}${evidenceT!=null?evidenceHtml(evidenceT):''}`;}
function respond(){
  const q=(ask.value||'').trim(); if(!q)return;
  const started=performance.now();
  const low=q.toLowerCase(), e=matchEntity(q), a=action(q), t=timeHint(q)??timeHintLoose(q);
  let intent='keyword';
  if(/what did i see|what did i observe|inventory|tracked|overview|what objects/.test(low))intent='inventory';
  else if(/how many/.test(low))intent='count';
  else if(/not in (this )?memory|unknown|no such/.test(low))intent='unknown';
  else if(/\\bbefore\\b/.test(low))intent='before';
  else if(/\\bafter\\b/.test(low))intent='after';
  else if(/what happened|what was happening|what occurred|what was going on/.test(low))intent=t!=null?'window':'history';
  else if(/\\b(where|which room|what room|what zone)\\b/.test(low))intent='where';
  else if(/\\b(when|what time|at what (second|time))\\b/.test(low))intent='when';
  else if(/^(did|do|does|was|were|is|are|have|has)\\b/.test(low))intent='yesno';
  else if(/doing with|do with|did i do/.test(low))intent='history';
  const key=canonical(intent,e,t,a);
  if(cache.has(key)){const hit=cache.get(key); render(hit.text,hit.cites,`${(performance.now()-started).toFixed(1)} ms · cached`,hit.evidenceT,`cache:${hit.method}`); return;}
  let text='', cites=[], evidenceT=null, method='vault';
  if(intent==='inventory'){
    const counts={}; for(const x of DATA.entities)counts[x.label]=(counts[x.label]||0)+1;
    text=`The memory tracked ${DATA.entities.length} object tracks across ${Object.keys(counts).length} labels: ${Object.entries(counts).sort().map(([k,v])=>`${v} ${k} track${v>1?'s':''}`).join(', ')}.`;
    if(DATA.windows.length)text+=` The dense caption ledger covers ${DATA.windows.length} window(s).`;
    const last=latest(DATA.events); if(last){cites.push(cite(last.t,caption(last)));evidenceT=+last.t;} method='vault-inventory';
  } else if(intent==='unknown'){
    text="I don't have that in my memory."; method='no-match';
  } else if(intent==='count'){
    if(!e){text="I don't have that in my memory.";method='no-match';}
    else{const n=DATA.entities.filter(x=>x.label.toLowerCase()===e.label.toLowerCase()).length; const ev=latest(eventsFor(e.label)); const bound=DATA.counts&&DATA.counts[e.label.toLowerCase()]; const peak=bound&&+bound.peak||0; const phys=bound&&bound.physical!=null?+bound.physical:null; const dups=bound&&+bound.duplicates||0; const trans=bound&&+bound.transient||0; const merged=(bound&&bound.merged_into)||{}; if(phys!=null&&phys<n){const mergedText=Object.entries(merged).map(([k,v])=>`${v} ${k} track${+v===1?'':'s'}`).join(', '); let detail=dups?`${dups} of them are near-identical duplicate detections of ${mergedText||'another label'}`:''; if(trans)detail+=`${detail?' and ':''}${trans} one-frame track${trans===1?'':'s'} ${trans===1?'is':'are'} transient`; text=`The memory tracked ${n} ${e.label.toLowerCase()} track${n===1?'':'s'}. ${detail}, so the physical ${e.label.toLowerCase()} count is ${phys}.`; method='vault-count-dedup';} else if(peak&&peak<n){text=`The memory tracked ${n} ${e.label.toLowerCase()} track${n===1?'':'s'}. At most ${peak} ${e.label.toLowerCase()} track${peak===1?'':'s'} ${peak===1?'was':'were'} visible at the same instant, so the physical count is between ${peak} and ${n}: fragmentation or occlusion may inflate the track count.`; method='vault-count-visibility';} else if(peak===1&&n===1){text=`The memory tracked 1 ${e.label.toLowerCase()} track. It was visible in the observation window, but this is still a track count, not a verified physical count.`; method='vault-count-visibility';} else if(peak){text=`The memory tracked ${n} ${e.label.toLowerCase()} track${n===1?'':'s'}. Up to ${peak} were visible at the same instant, which is still a track count, not a verified physical count.`; method='vault-count-visibility';} else {text=`The memory tracked ${n} ${e.label.toLowerCase()} track${n===1?'':'s'}. The tracker can split one physical object into more than one track, so this is a track count, not a verified object count.`; method='vault-count';} if(ev){cites.push(cite(ev.t,caption(ev)));evidenceT=+ev.t;}}
  } else if(intent==='where'){
    if(!e){const hit=keyword(q); text=hit.text; cites=hit.cites; evidenceT=hit.evidenceT; method=hit.method;}
    else{const rows=eventsFor(e.label); const last=latest(rows); const places=rows.filter(x=>x.kind==='anchor'&&['PLACED_ON','PLACED_IN','RESTING_ON','INSIDE'].includes(x.relation.toUpperCase())); const place=latest(places); if(!last){text=`I don't have a location for the ${e.label.toLowerCase()}.`;method='no-match';} else {const held=last.kind==='interaction'&&last.relation.toUpperCase()==='HELD'; const seen=Math.max(+e.last_seen||0,+last.t); text=`The ${e.label.toLowerCase()} was last ${held?'held':'seen'} at t=${seen.toFixed(1)}s${e.zone?' in the '+e.zone:''}.`; if(place)text+=` Last recorded placement: ${caption(place).replace(/^At t=[\\d.]+s, /,'').replace(/\\.$/,'')}.`; else text+=` Last recorded event: ${caption(last).replace(/^At t=[\\d.]+s, /,'').replace(/\\.$/,'')}.`; cites.push(cite(last.t,caption(last))); evidenceT=+last.t; method='vault-where';}}
  } else if(intent==='when'){
    if(!e){const hit=keyword(q); text=hit.text; cites=hit.cites; evidenceT=hit.evidenceT; method=hit.method;}
    else{const rows=eventsFor(e.label); const filtered=rows.filter(x=>matchesAction(x,a)); if(!filtered.length){text=`No ${a||'matching'} events are recorded for the ${e.label.toLowerCase()}.`;method='no-match';} else {const first=/first/.test(low); const target=first?filtered[0]:filtered[filtered.length-1]; const what={touch:'touched or held',place:'placed',see:'seen',lost:'lost from view'}[a]||'recorded'; text=`The ${e.label.toLowerCase()} was ${first?'first':'last'} ${what} at t=${(+target.t).toFixed(1)}s${target.zone?' in the '+target.zone:''}. ${caption(target)}`; cites.push(cite(target.t,caption(target))); evidenceT=+target.t; method='vault-when';}}
  } else if(intent==='before'||intent==='after'){
    let anchor=null; if(t!=null)anchor=t; else if(e){const rows=eventsFor(e.label); const filtered=rows.filter(x=>matchesAction(x,a)); anchor=(latest(filtered.length?filtered:rows)||{}).t;}
    if(anchor==null){text='I need a time or an object to answer that.';method='no-match';}
    else{const radius=15; const rows=DATA.events.filter(x=>intent==='before'?(+x.t>=anchor-radius&&+x.t<anchor):(+x.t>anchor&&+x.t<=anchor+radius)); if(!rows.length){text=`No events recorded ${intent} t=${anchor.toFixed(1)}s.`;method='no-match';} else {text=`${intent==='before'?'Before':'After'} t=${anchor.toFixed(1)}s:\\n`+rows.map(x=>`- ${caption(x)}`).join('\\n'); cites=rows.slice(0,8).map(x=>cite(x.t,caption(x))); evidenceT=intent==='before'?+rows[rows.length-1].t:+rows[0].t; method='vault-'+intent;}}
  } else if(intent==='window'){
    const tt=t!=null?t:DATA.duration*0.5; let rows=DATA.events.filter(x=>Math.abs(+x.t-tt)<=8); if(rows.length<2)rows=DATA.events.filter(x=>Math.abs(+x.t-tt)<=20); const w=windowAt(tt); text=`Around t=${tt.toFixed(1)}s:`; if(w)text+=`\\nMemory caption: ${w.caption||''}`; text+='\\n'+rows.slice(0,12).map(x=>`- ${caption(x)}`).join('\\n'); cites=rows.slice(0,8).map(x=>cite(x.t,caption(x))); if(w)cites.unshift(cite(w.t_start,w.caption||'')); evidenceT=tt; method='vault-window';
  } else if(intent==='yesno'){
    if(!e){text="No, I don't have that in my memory.";method='no-match';}
    else{const rows=eventsFor(e.label).filter(x=>matchesAction(x,a)); if(!rows.length){text=`No record of the ${e.label.toLowerCase()} in that action.`;method='no-match';} else {const target=rows[rows.length-1]; const what={touch:'touched',place:'placed',see:'seen',lost:'lost'}[a]||'recorded'; text=`Yes. The ${e.label.toLowerCase()} was ${what} at t=${(+target.t).toFixed(1)}s. ${caption(target)}`; cites.push(cite(target.t,caption(target))); evidenceT=+target.t; method='vault-yesno';}}
  } else if(intent==='history'){
    if(!e){const hit=keyword(q); text=hit.text; cites=hit.cites; evidenceT=hit.evidenceT; method=hit.method;}
    else{const rows=eventsFor(e.label).slice(-6); if(!rows.length){text=`No events recorded for the ${e.label.toLowerCase()}.`;method='no-match';} else {text=`Here is the recorded history for the ${e.label.toLowerCase()}:\\n`+rows.map(x=>`- ${caption(x)}`).join('\\n'); cites=rows.map(x=>cite(x.t,caption(x))); evidenceT=+rows[rows.length-1].t; method='vault-history';}}
  } else {const hit=keyword(q); text=hit.text; cites=hit.cites; evidenceT=hit.evidenceT; method=hit.method;}
  cache.set(key,{text,cites,evidenceT,method});
  render(text,cites,`${(performance.now()-started).toFixed(1)} ms`,evidenceT,method);
}
function keyword(q){
  const ts=[...toks(q)].filter(x=>!STOP.has(x)); if(!ts.length)return {text:"I don't have that in my memory.",cites:[],evidenceT:null,method:'no-match'};
  const scored=[];
  for(const w of DATA.windows){const text=((w.caption||'')+' '+(w.objects||[]).join(' ')+' '+(w.actions||[]).join(' ')+' '+(w.tags||[]).join(' ')).toLowerCase(); const hits=ts.filter(x=>text.includes(x)).length; if(hits)scored.push({hits:hits*2,t:+w.t_start,text:w.caption||'',kind:'caption'});}
  if(!scored.length){for(const x of DATA.events){const text=caption(x).toLowerCase(); const hits=ts.filter(y=>text.includes(y)).length; if(hits)scored.push({hits,t:+x.t,text:caption(x),kind:x.kind});}}
  if(!scored.length)return {text:"I don't have that in my memory.",cites:[],evidenceT:null,method:'no-match'};
  scored.sort((a,b)=>b.hits-a.hits||b.t-a.t); const top=scored.slice(0,5);
  return {text:'Here is what the memory contains:\\n'+top.map(x=>`- [t=${x.t.toFixed(1)}s] ${x.text.split('\\n')[0]}`).join('\\n'),cites:top.map(x=>cite(x.t,x.text)),evidenceT:top[0].t,method:top[0].kind==='caption'?'vault-keyword-caption':'vault-keyword-event'};
}
function caption(e){const t=+e.t,s=(e.subject||'').toLowerCase(),r=(e.relation||'').replace(/_/g,' ').toLowerCase(),o=(e.object||'').toLowerCase(),z=e.zone||''; let text=''; if(e.kind==='interaction')text=`At t=${t.toFixed(1)}s, the ${s} was ${r}`; else if(e.kind==='anchor'){if(r==='placed in')text=`At t=${t.toFixed(1)}s, the ${s} was placed in the ${o}`; else if(r==='placed on')text=`At t=${t.toFixed(1)}s, the ${s} was placed on the ${o}`; else text=`At t=${t.toFixed(1)}s, the ${s} was ${r} the ${o}`;} else if(e.kind==='state'){if(r==='appeared')text=`At t=${t.toFixed(1)}s, the ${s} appeared`; else if(r==='lost')text=`At t=${t.toFixed(1)}s, the ${s} was lost from view`; else text=`At t=${t.toFixed(1)}s, the ${s} was ${r}`;} else if(e.kind==='keyframe')text=`Scene anchor at t=${t.toFixed(1)}s`; else text=`At t=${t.toFixed(1)}s, ${s} ${o}`; if(z&&z.toLowerCase()!==s&&z.toLowerCase()!==o)text+=` in the ${z}`; return text.replace(/\\.$/,'')+'.';}
function apply(){
  const q=(search.value||'').toLowerCase();
  rows.forEach(li=>{const text=(li.dataset.text||'').toLowerCase(); const okQ=!q||text.includes(q); const okE=!active||text.includes(active); li.style.display=(okQ&&okE)?'grid':'none';});
}
search.addEventListener('input',apply);
ask.addEventListener('keydown',ev=>{if(ev.key==='Enter')respond();});
askBtn.addEventListener('click',respond);
entities.forEach(el=>el.addEventListener('click',()=>{active=(active===el.dataset.label)?null:el.dataset.label; entities.forEach(x=>x.style.color=(x.dataset.label===active)?'var(--accent)':''); apply();}));
apply();
"""


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _image_data_url(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > 400_000:
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"
    except OSError:
        return None


def build_report(
    db: Path,
    out: Path,
    data_root: Path | None = None,
    *,
    title: str | None = None,
    max_images: int = 40,
) -> dict[str, Any]:
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

    # Embed each unique image once; evidence rows can repeat across a synthetic
    # long-horizon vault, and duplicating base64 per row balloons the HTML.
    unique: dict[str, dict[str, Any]] = {}
    for shot in sorted(evidence, key=lambda row: float(row["t"])):
        rel_path = str(shot["rel_path"])
        if rel_path in unique:
            continue
        path = None
        if data_root is not None:
            path = (data_root / rel_path).resolve()
        elif Path(rel_path).is_file():
            path = Path(rel_path)
        data_url = _image_data_url(path) if path is not None else None
        if not data_url:
            continue
        unique[rel_path] = {**shot, "data_url": data_url}
        if len(unique) >= max_images:
            break

    gallery: list[str] = []
    src_by_rel: dict[str, str] = {}
    for index, (rel_path, shot) in enumerate(unique.items()):
        label = str(shot.get("label") or shot.get("kind") or "event")
        figure_id = f"shot-{index}"
        src_by_rel[rel_path] = str(shot["data_url"])
        gallery.append(
            f'<figure class="shot" id="{figure_id}"><img src="{shot["data_url"]}" '
            f'alt="{_esc(label)} at {shot["t"]}s" loading="lazy">'
            f'<div>[t={float(shot["t"]):.1f}s] {_esc(label)}</div></figure>'
        )

    windows = "".join(
        f'<div class="window"><b>[{float(window.get("t_start", 0)):.1f}–'
        f'{float(window.get("t_end", 0)):.1f}s]</b><p>{_esc(window.get("caption") or window.get("text") or "")}</p>'
        f'<div class="tags">{_esc(", ".join(str(tag) for tag in window.get("tags") or []))}</div></div>'
        for window in store.ledger_windows
    )

    memory_data = {
        "duration": round(store.duration, 3),
        # Per-label {"tracks", "peak"}: track count is the upper bound, peak
        # simultaneous visibility the lower bound. Empty on caption-first vaults.
        "counts": store.counts,
        "entities": [
            {
                "id": entity["id"],
                "label": entity["label"],
                "state": entity["state"],
                "zone": entity["zone"],
                "last_seen": entity.get("last_seen"),
            }
            for entity in store.entities
        ],
        "events": [
            {
                "t": round(float(event["t"]), 3),
                "kind": event["kind"],
                "subject": event["subject"],
                "relation": event["relation"],
                "object": event["object"],
                "zone": event["zone"],
            }
            for event in store.events
        ],
        "windows": [
            {
                "t_start": window.get("t_start"),
                "t_end": window.get("t_end"),
                "caption": window.get("caption") or window.get("text") or "",
                "objects": window.get("objects") or [],
                "actions": window.get("actions") or [],
                "tags": window.get("tags") or [],
            }
            for window in store.ledger_windows
        ],
        "evidence": [
            {
                "t": round(float(shot["t"]), 3),
                "kind": shot["kind"],
                "label": shot["label"],
                "rel_path": shot["rel_path"],
            }
            for shot in evidence
        ],
        "srcByRel": src_by_rel,
    }
    data_json = json.dumps(memory_data, ensure_ascii=False).replace("</", "<\\/")

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
    <div class="stat"><b>{info.get("ledger_windows", 0)}</b><span>dense windows</span></div>
    <div class="stat"><b>{info.get("bytes", 0)/1024:.1f} KB</b><span>vault size</span></div>
    <div class="stat"><b>{len(unique)}</b><span>unique snapshots</span></div>
    <div class="stat"><b>0</b><span>video retained</span></div>
  </div>
</header>
<main>
  <aside><div class="card"><h2>Entities</h2>{cards}</div>
  <div class="card" style="margin-top:14px"><h2>Dense caption ledger</h2>
  {windows if windows else '<div class="note">No VLM ledger attached.</div>'}</div>
  <div class="card" style="margin-top:14px"><h2>Notes</h2>
  <div class="note">This page is generated from the vault alone. Click an entity to filter.
  The query box runs the same intents as <code>query.py</code>, entirely in the browser.
  Evidence images are stored snapshots, not video frames.</div></div></aside>
  <section>
    <input id="ask" placeholder="Ask: where is the cup? / what happened around 20s? / how many plates?" autocomplete="off">
    <button id="ask-btn" type="button">Ask the vault</button>
    <div class="answer" id="answer"><div class="a-text">Ask a question above. No video, no model, no network.</div></div>
    <input id="search" placeholder="Search the timeline…" autocomplete="off">
    <ol id="timeline">{"".join(rows)}</ol>
    <h2 style="font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 0">
      Evidence snapshots ({len(unique)} unique, {len(evidence)} rows)</h2>
    <div class="gallery">{"".join(gallery) if gallery else '<div class="note">No snapshots were extracted for this vault.</div>'}</div>
  </section>
</main>
<script id="memory-data" type="application/json">{data_json}</script>
<script>{JS}</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return {
        "html": str(out),
        "bytes": out.stat().st_size,
        "events": len(store.events),
        "entities": len(store.entities),
        "ledger_windows": len(store.ledger_windows),
        "evidence_rows": len(evidence),
        "unique_images": len(unique),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-images", type=int, default=40)
    args = parser.parse_args(argv)
    summary = build_report(
        args.db.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.data_root.expanduser().resolve() if args.data_root else None,
        title=args.title,
        max_images=max(1, args.max_images),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
