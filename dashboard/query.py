"""Natural-language query console backend (two-stage GraphRAG).

Matches section 6 of the technical spec:

1. resolve the entity + active spatial anchor,
2. serialise the 2-hop relational subgraph,
3. synthesise an answer.  The deterministic answerer keeps the dashboard
   useful without a GPU, while a local Ollama model (Qwen 2.5 by default)
   turns the evidence table into natural prose.  ``CHRONICLE_QUERY_URL`` can
   still attach an external answer endpoint when one is available.

Every answer is grounded in graph records and carries timestamped citations
which the UI can seek to.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .contracts import clean_text, pretty_ref
from .frames import AnswerFrameSelector
from .store import ACTION_RELATIONS, SPATIAL_RELATIONS, Edge, Entity, MemoryStore, Zone

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HTML_RE = re.compile(r"<[^>]+>")

#: Everyday words -> canonical labels.  Keep this small and predictable.
ALIASES: dict[str, set[str]] = {
    "key": {"keys"},
    "keys": {"keys", "key"},
    "phone": {"phone", "smartphone", "mobile", "cellphone"},
    "smartphone": {"phone"},
    "mobile": {"phone"},
    "cup": {"mug", "cup"},
    "mug": {"mug", "cup", "coffee"},
    "coffee": {"mug"},
    "bag": {"tote", "bag", "backpack"},
    "tote": {"tote", "bag"},
    "backpack": {"backpack", "bag"},
    "rucksack": {"backpack"},
    "book": {"notebook", "book"},
    "notebook": {"notebook", "book"},
    "laptop": {"notebook", "laptop"},
    "container": {"tote", "bag"},
}

RELATION_VERBS: dict[str, str] = {
    "PICKED_UP": "picked up",
    "PICK_UP": "picked up",
    "PUT_DOWN": "put down",
    "PLACED_IN": "placed",
    "INSERT_INTO": "placed",
    "PLACED_ON": "placed",
    "TOUCHED": "touched",
    "OPENED": "opened",
    "CLOSED": "closed",
    "USED": "used",
    "MOVED": "moved",
}

SYSTEM_PROMPT = (
    "You are Chronicle, an egocentric episodic-memory query engine. "
    "Answer ONLY from the supplied spatio-temporal subgraph; never invent objects, "
    "places or times. Cite every fact with the presentation timestamp in square "
    "brackets, e.g. [t=16.9s]. Be concise (1-3 sentences). If the graph does not "
    "contain the answer, say so. Return JSON only: "
    '{"answer": "...", "citations": [{"t": 16.9, "label": "CONTAINED", "entity_id": "keys_1"}]}'
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "qwen2.5:7b-instruct"
NO_MEMORY_PHRASE = "i don't have that in my memory"


class QueryEngine:
    def __init__(
        self,
        store: MemoryStore,
        endpoint: str | None = None,
        token: str | None = None,
        *,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        ollama_url: str | None = None,
        data_dir: str | Path | None = None,
        video_path: str | Path | Callable[[], str | Path | None] | None = None,
    ) -> None:
        self.store = store
        self.endpoint = endpoint if endpoint is not None else os.getenv("CHRONICLE_QUERY_URL")
        self.token = token if token is not None else os.getenv("CHRONICLE_QUERY_TOKEN")
        self.frames = AnswerFrameSelector(
            store,
            data_dir=data_dir,
            video_path=video_path,
        )
        self.llm_backend = (
            llm_backend
            if llm_backend is not None
            else os.getenv("CHRONICLE_LLM_BACKEND", "auto")
        ).strip().lower()
        self.llm_model = (
            llm_model if llm_model is not None else os.getenv("CHRONICLE_LLM_MODEL", DEFAULT_LLM_MODEL)
        ).strip()
        self.ollama_url = (
            ollama_url if ollama_url is not None else os.getenv("CHRONICLE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self._last_entity: str | None = None
        self.last_provider = "local-graphrag"
        self._chat_client: Any = None
        self._llm_checked = False
        self._llm_status: dict[str, Any] = {}
        self._llm_checked_at = 0.0
        self._llm_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def answer(self, query: str, t: float) -> dict[str, Any]:
        started = time.perf_counter()
        query = clean_text(query, 300)
        if not query:
            result = self._empty("Ask a question about an object in the episode.", t)
            result["frame"] = None
            return result

        match = self._match_entities(query)
        zone = self._match_zone(query) if not match else None

        if match:
            entity = match[0]
            self._last_entity = entity.id
            result = self._answer_entity(entity, query, t)
        elif zone is not None:
            result = self._answer_zone(zone, query, t)
        elif self._wants_timeline(query):
            result = self._answer_timeline(query, t)
        elif self._last_entity and self.store.entities.get(self._last_entity):
            result = self._answer_entity(self.store.entities[self._last_entity], query, t)
            result["answer"] = "(Resolved 'it' to the last inspected object.) " + result["answer"]
        else:
            result = self._no_match(query, t)

        result["query"] = query
        result["playhead"] = round(t, 3)
        result["matched_entity"] = match[0].id if match else None
        result["provider"] = "local-graphrag"
        result["model"] = "chronicle-local"
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)

        route = result.get("route")
        if self.endpoint and route in {"entity", "zone", "timeline"}:
            external = self._ask_external(query, t, result)
            if external:
                if external.get("answer"):
                    result.update(external)
                    result["provider"] = "role-c-llm"
                    result["model"] = external.get("model") or "role-c"
                else:
                    result["llm_error"] = external.get("llm_error")
        elif route in {"entity", "zone", "timeline"} and result.get("rows") and self._llm_enabled():
            local = self._ask_local_llm(query, t, result)
            if local:
                if local.get("answer"):
                    result["answer"] = local["answer"]
                    result["citations"] = local.get("citations") or result.get("citations") or []
                    result["provider"] = local.get("provider") or f"ollama:{self.llm_model}"
                    result["model"] = local.get("model") or self.llm_model
                    result["llm_error"] = None
                else:
                    if self.llm_backend == "ollama":
                        result["llm_error"] = local.get("llm_error")
        self.last_provider = str(result.get("provider") or "local-graphrag")
        result["frame"] = self.frames.for_answer(result, t)
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return the configured answer provider without blocking the UI."""
        if self.endpoint:
            return {
                "provider": "role-c-llm",
                "model": "role-c",
                "backend": "external",
                "server": True,
                "available": True,
            }
        configured = self._llm_enabled()
        if not configured:
            return {
                "provider": "local-graphrag",
                "model": "chronicle-local",
                "backend": self.llm_backend,
                "server": False,
                "available": False,
            }
        now = time.monotonic()
        with self._llm_lock:
            if refresh or not self._llm_status or now - self._llm_checked_at >= 15.0:
                self._llm_status = self._probe_llm()
                self._llm_checked_at = now
            status = dict(self._llm_status)
        return {
            "provider": f"ollama:{self.llm_model}" if status.get("available") else "local-graphrag",
            "model": self.llm_model if status.get("available") else "chronicle-local",
            "backend": "ollama",
            "server": bool(status.get("server")),
            "available": bool(status.get("available")),
            "loaded": status.get("loaded") or [],
            "version": status.get("version"),
        }

    def provider_label(self) -> str:
        if self.endpoint:
            return "role-c-llm"
        if self._llm_enabled():
            return f"ollama:{self.llm_model}"
        return "local-graphrag"

    async def answer_async(self, query: str, t: float) -> dict[str, Any]:
        return await asyncio.to_thread(self.answer, query, t)

    def suggestions(self, t: float | None = None) -> list[str]:
        with self.store._lock:  # noqa: SLF001 - read-only convenience
            entities = sorted(
                self.store.entities.values(),
                key=lambda e: (len(e.interactions), e.last_seen),
                reverse=True,
            )
        suggestions: list[str] = []
        for entity in entities[:4]:
            label = entity.label.lower()
            suggestions.append(f"Where are my {label}?")
            if entity.interactions:
                suggestions.append(f"When did I last touch the {label}?")
        suggestions.append("What did I do in this episode?")
        return suggestions[:5]

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #

    def _match_entities(self, query: str) -> list[Entity]:
        lower = query.lower().strip().rstrip("?!. ")
        tokens = set(_TOKEN_RE.findall(lower))
        expanded = set(tokens)
        for token in tokens:
            expanded |= ALIASES.get(token, set())

        scored: list[tuple[float, float, Entity]] = []
        for entity in self.store.entities.values():
            haystack = f"{entity.label} {entity.cls} {entity.id}".lower()
            words = set(_TOKEN_RE.findall(haystack))
            score = 0.0
            if lower in {entity.label.lower(), entity.id.lower()}:
                score = 1.0
            elif words & tokens:
                # Tokens written by the user outrank alias-expanded words.
                score = 0.96
            elif words & expanded:
                score = 0.82
            for token in tokens:
                if len(token) >= 3 and token in haystack:
                    score = max(score, 0.9)
            for token in expanded:
                if len(token) >= 4:
                    ratio = difflib.SequenceMatcher(None, token, entity.label.lower()).ratio()
                    if ratio >= 0.72:
                        score = max(score, 0.68 + 0.25 * ratio)
            if score <= 0:
                continue
            if entity.state not in {"OCCLUDED", "OUT_OF_VIEW", "LOST"}:
                score += 0.04
            score += min(len(entity.interactions) * 0.01, 0.04)
            scored.append((score, entity.last_seen, entity))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:3]]

    def _match_zone(self, query: str) -> Zone | None:
        lower = query.lower()
        for zone in self.store.zones.values():
            if zone.label.lower() in lower:
                return zone
        return None

    @staticmethod
    def _wants_timeline(query: str) -> bool:
        return bool(
            re.search(
                r"\b(what did i do|what happened|summar|recap|timeline|this episode|before leaving)\b",
                query.lower(),
            )
        )

    # ------------------------------------------------------------------ #
    # Subgraph
    # ------------------------------------------------------------------ #

    def _build_subgraph(self, entity: Entity, t: float, hops: int = 2) -> dict[str, Any]:
        entity_ids: set[str] = {entity.id}
        zone_ids: set[str] = set()
        edges: list[Edge] = []
        frontier: set[str] = {entity.id}

        all_edges = sorted(self.store.edges.values(), key=lambda e: e.t_start)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for edge in all_edges:
                if edge.t_start > t + 0.05:
                    continue
                if edge.source not in frontier and edge.target not in frontier:
                    continue
                if edge not in edges:
                    edges.append(edge)
                for ref in (edge.source, edge.target):
                    if ref in self.store.entities:
                        if ref not in entity_ids:
                            next_frontier.add(ref)
                        entity_ids.add(ref)
                    elif ref in self.store.zones:
                        zone_ids.add(ref)
            frontier = next_frontier
            if not frontier:
                break

        if entity.zone_id and entity.zone_id in self.store.zones:
            zone_ids.add(entity.zone_id)
        for entity_id in list(entity_ids):
            candidate = self.store.entities[entity_id]
            if candidate.zone_id and candidate.zone_id in self.store.zones:
                zone_ids.add(candidate.zone_id)

        entities = [self.store.entities[eid] for eid in entity_ids if eid in self.store.entities]
        zones = [self.store.zones[zid] for zid in zone_ids if zid in self.store.zones]
        return {
            "entity": entity,
            "entities": entities,
            "zones": zones,
            "edges": sorted(edges, key=lambda e: e.t_start),
        }

    def _serialise(self, subgraph: dict[str, Any], t: float) -> tuple[list[dict[str, Any]], str]:
        rows: list[dict[str, Any]] = []
        for entity in subgraph["entities"]:
            state = entity.state_at(t)
            zone = self.store.zones.get(entity.zone_id or "")
            rows.append(
                {
                    "node": entity.label,
                    "id": entity.id,
                    "type": "entity",
                    "state": state,
                    "relation": "IN_ZONE",
                    "target": zone.label if zone else "—",
                    "t_start": round(entity.first_seen, 3),
                    "t_end": round(entity.last_seen, 3),
                    "clip_sim": entity.clip_sim,
                }
            )
        for edge in subgraph["edges"]:
            source = self.store.entities.get(edge.source)
            target_entity = self.store.entities.get(edge.target)
            source_label = (
                source.label
                if source
                else self.store.zones[edge.source].label
                if edge.source in self.store.zones
                else self.store.anchor_labels.get(edge.source) or pretty_ref(edge.source)
            )
            target_label = (
                target_entity.label
                if target_entity
                else self.store.zones[edge.target].label
                if edge.target in self.store.zones
                else self.store.anchor_labels.get(edge.target) or pretty_ref(edge.target)
            )
            rows.append(
                {
                    "node": source_label,
                    "id": edge.source,
                    "type": "relation",
                    "state": source.state_at(t) if source else "ANCHOR",
                    "relation": edge.relation,
                    "target": target_label,
                    "t_start": round(edge.t_start, 3),
                    "t_end": None if edge.t_end is None else round(edge.t_end, 3),
                    "clip_sim": edge.clip_sim,
                }
            )

        header = "| NODE | STATE | RELATION | TARGET | T_START | T_END | CLIP |"
        divider = "|---|---|---|---|---|---|---|"
        lines = [header, divider]
        for row in rows:
            lines.append(
                "| {node} | {state} | {relation} | {target} | {t_start:.1f}s | {t_end} | {clip} |".format(
                    node=row["node"],
                    state=row["state"],
                    relation=row["relation"],
                    target=row["target"],
                    t_start=row["t_start"],
                    t_end="—" if row["t_end"] is None else f"{row['t_end']:.1f}s",
                    clip="—" if row["clip_sim"] is None else f"{row['clip_sim']:.2f}",
                )
            )
        return rows, "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Answer synthesis
    # ------------------------------------------------------------------ #

    def _answer_entity(self, entity: Entity, query: str, t: float) -> dict[str, Any]:
        subgraph = self._build_subgraph(entity, t)
        rows, table = self._serialise(subgraph, t)
        state = entity.state_at(t)
        label = entity.label

        edges = [
            e
            for e in subgraph["edges"]
            if e.source == entity.id and e.relation in SPATIAL_RELATIONS and e.t_start <= t + 0.05
        ]
        active_edges = [e for e in edges if e.active_at(t)]
        active_edges.sort(key=lambda e: (e.relation not in {"INSIDE", "CONTAINED_IN"}, -e.t_start))
        current_edge = active_edges[0] if active_edges else None
        last_edge = current_edge or (max(edges, key=lambda e: e.t_start) if edges else None)

        action_edges = [
            e
            for e in subgraph["edges"]
            if e.target == entity.id and e.relation in ACTION_RELATIONS and e.t_start <= t + 0.05
        ]
        action_edges.sort(key=lambda e: e.t_start)
        last_action = action_edges[-1] if action_edges else None

        vlm = [i for i in entity.interactions if i.t <= t + 0.05]
        last_vlm = vlm[-1] if vlm else None
        last_interaction_t = max(
            [r.t for r in entity.history if r.to_state in {"INTERACTED", "HELD"} and r.t <= t + 0.05]
            + [i.t for i in vlm]
            + ([last_action.t_start] if last_action else []),
            default=None,
        )
        zone = self.store.zones.get(entity.zone_id or "")

        # ---- location phrasing --------------------------------------- #
        location = self._location_phrase(entity, last_edge, current_edge, t)
        prefix = "last known location" if state in {"OCCLUDED", "OUT_OF_VIEW", "LOST"} and not current_edge else "location"

        if state == "HELD":
            sentence = f"You are holding the {label}."
            if location:
                sentence += f" Its last stable anchor was {location}."
        elif state == "CONTAINED":
            sentence = f"The {label} {self._be(label)} contained {location}."
        elif state in {"VISIBLE_STATIC", "INTERACTED"}:
            sentence = f"The {label} {self._be(label)} {location or 'visible in the current view'}."
        elif state == "VISIBLE_MOVING":
            sentence = f"The {label} {self._be(label)} moving independently near {location or 'the current view'}."
        elif state == "OCCLUDED":
            sentence = f"The {label} {self._be(label)} currently occluded; {prefix} is {location or 'unknown'}."
        elif state == "OUT_OF_VIEW":
            sentence = (
                f"The {label} left the camera frustum when you panned away; its retained anchor is "
                f"{location or 'unknown'}."
            )
        elif state == "LOST":
            sentence = (
                f"The {label} {self._be(label)} marked LOST/dormant, but its terminal anchor is permanently retained: "
                f"{location or 'unknown'}."
            )
        elif state == "NEW":
            sentence = (
                f"The {label} was first detected at {entity.first_seen:.1f}s and has not settled on an anchor yet."
            )
        else:
            sentence = f"The {label} {self._be(label)} currently {state.replace('_', ' ').lower()}."

        # ---- interaction narrative ----------------------------------- #
        action_sentence = ""
        if last_action is not None:
            verb = RELATION_VERBS.get(last_action.relation, last_action.relation.replace("_", " ").lower())
            target = self.store.entities.get(last_action.target)
            if last_action.relation in {"PLACED_IN", "INSERT_INTO"}:
                destination = f"the {target.label}" if target else pretty_ref(last_action.target)
                action_sentence = f"You {verb} it into {destination} at {last_action.t_start:.1f}s."
            elif last_action.relation == "PUT_DOWN":
                action_sentence = f"You {verb} it at {last_action.t_start:.1f}s."
            else:
                action_sentence = f"You {verb} it at {last_action.t_start:.1f}s."
            if last_vlm and abs(last_vlm.t - last_action.t_start) > 0.2:
                action_sentence = action_sentence.rstrip(".") + f" (VLM: {last_vlm.action} at {last_vlm.t:.1f}s)."

        if self._wants_when(query):
            if last_interaction_t is not None:
                citation = last_vlm.action if last_vlm else RELATION_VERBS.get(last_action.relation if last_action else "", "INTERACTED")
                sentence = f"You last interacted with the {label} at {last_interaction_t:.1f}s ({citation.upper()})."
                if state in {"VISIBLE_STATIC", "CONTAINED", "VISIBLE_MOVING"}:
                    sentence += f" It is now {state.replace('_', ' ').lower()} {location}."
                elif state == "HELD":
                    sentence += " You are still holding it."
            else:
                sentence = f"No hand interaction has been recorded for the {label}."
                if location:
                    sentence += f" It is {location}."

        answer = " ".join(part for part in (sentence, action_sentence) if part).strip()
        if zone and state in {"VISIBLE_STATIC", "CONTAINED"} and zone.label.lower() not in answer.lower():
            answer += f" Anchor zone: {zone.label}."

        citations = self._citations_for(entity, subgraph, t, last_edge, last_action, last_vlm)
        confidence = self._confidence(entity, [last_edge, last_action], last_vlm)
        return {
            "route": "entity",
            "answer": answer,
            "citations": citations,
            "rows": rows,
            "table": table,
            "entities": [e.as_dict() for e in subgraph["entities"]],
            "zones": [z.as_dict() for z in subgraph["zones"]],
            "edges": [e.as_dict() for e in subgraph["edges"]],
            "confidence": confidence,
            "prompt": self._build_prompt(query, table, entity, state, t),
            "subgraph": {
                "entity_id": entity.id,
                "entities": [e.as_dict() for e in subgraph["entities"]],
                "zones": [z.as_dict() for z in subgraph["zones"]],
                "edges": [e.as_dict() for e in subgraph["edges"]],
            },
        }

    def _answer_zone(self, zone: Zone, query: str, t: float) -> dict[str, Any]:
        entities = [
            e
            for e in self.store.entities.values()
            if e.zone_id == zone.id and e.first_seen <= t + 0.05
        ]
        active = [e for e in entities if e.state_at(t) not in {"OCCLUDED", "OUT_OF_VIEW", "LOST"}]
        listing = ", ".join(f"{e.label} ({e.state_at(t).replace('_', ' ').lower()})" for e in active[:6])
        answer = f"You are in the {zone.label}."
        if listing:
            answer += f" Tracked here: {listing}."
        else:
            answer += " No active objects are tracked in this zone."
        rows = [
            {
                "node": e.label,
                "id": e.id,
                "type": "entity",
                "state": e.state_at(t),
                "relation": "IN_ZONE",
                "target": zone.label,
                "t_start": round(e.first_seen, 3),
                "t_end": round(e.last_seen, 3),
                "clip_sim": e.clip_sim,
            }
            for e in entities
        ]
        citations = [
            {
                "t": round(e.last_seen, 3),
                "label": e.state_at(t),
                "kind": "state",
                "entity_id": e.id,
                "detail": f"{e.label} in {zone.label}",
            }
            for e in active[:6]
        ]
        return {
            "route": "zone",
            "answer": answer,
            "citations": citations,
            "rows": rows,
            "table": self._rows_to_table(rows),
            "entities": [e.as_dict() for e in entities],
            "zones": [zone.as_dict()],
            "edges": [],
            "confidence": zone.clip_sim,
            "prompt": "",
            "subgraph": {"entity_id": None, "entities": [e.as_dict() for e in entities], "zones": [zone.as_dict()], "edges": []},
        }

    def _answer_timeline(self, query: str, t: float) -> dict[str, Any]:
        actions = [i for i in self.store.interactions if i.t <= t + 0.05]
        actions.sort(key=lambda item: item.t)
        if not actions:
            return {
                "route": "timeline",
                "answer": "No VLM-verified interactions have been recorded yet in this episode.",
                "citations": [],
                "rows": [],
                "table": "| no interactions |",
                "entities": [],
                "zones": [],
                "edges": [],
                "confidence": None,
                "prompt": "",
                "subgraph": {},
            }
        recent = actions[-6:]
        parts = [
            f"{record.t:.1f}s — {RELATION_VERBS.get(record.action, record.action.replace('_', ' ').lower())} "
            f"{self._label_for_entity(record)}"
            for record in recent
        ]
        answer = "Recent verified interactions: " + "; ".join(parts) + "."
        citations = [
            {
                "t": round(record.t, 3),
                "label": record.action,
                "kind": "interaction",
                "entity_id": None,
                "detail": record.caption,
            }
            for record in recent
        ]
        rows = [
            {
                "node": self._label_for_entity(record),
                "id": "",
                "type": "interaction",
                "state": record.action,
                "relation": "VLM",
                "target": "—",
                "t_start": round(record.t, 3),
                "t_end": None,
                "clip_sim": record.clip_sim,
            }
            for record in recent
        ]
        return {
            "route": "timeline",
            "answer": answer,
            "citations": citations,
            "rows": rows,
            "table": self._rows_to_table(rows),
            "entities": [],
            "zones": [],
            "edges": [],
            "confidence": None,
            "prompt": "",
            "subgraph": {},
        }

    def _no_match(self, query: str, t: float) -> dict[str, Any]:
        labels = sorted({e.label for e in self.store.entities.values()})
        tracked = ", ".join(labels) if labels else "no entities yet"
        answer = (
            f"I could not resolve '{query}' to a tracked entity in this episode. "
            f"Tracked objects: {tracked}. Try asking where one of them is."
        )
        return {
            "route": "none",
            "answer": answer,
            "citations": [],
            "rows": [],
            "table": "| no matching entity |",
            "entities": [],
            "zones": [],
            "edges": [],
            "confidence": None,
            "prompt": "",
            "subgraph": {},
        }

    def _empty(self, message: str, t: float) -> dict[str, Any]:
        return {
            "route": "none",
            "answer": message,
            "citations": [],
            "rows": [],
            "table": "",
            "entities": [],
            "zones": [],
            "edges": [],
            "confidence": None,
            "provider": "local-graphrag",
            "model": "chronicle-local",
            "duration_ms": 0.0,
            "query": "",
            "playhead": t,
            "subgraph": {},
        }

    # ------------------------------------------------------------------ #
    # Phrasing helpers
    # ------------------------------------------------------------------ #

    def _label_for_entity(self, record: Any) -> str:
        for entity in self.store.entities.values():
            if record in entity.interactions:
                return entity.label
        return "object"

    @staticmethod
    def _be(label: str) -> str:
        return "are" if label.strip().lower().endswith("s") else "is"

    def _zone_for(self, entity: Entity | None) -> Zone | None:
        if entity and entity.zone_id:
            return self.store.zones.get(entity.zone_id)
        return None

    def _location_phrase(
        self,
        entity: Entity,
        last_edge: Edge | None,
        current_edge: Edge | None,
        t: float,
        depth: int = 0,
    ) -> str:
        if last_edge is None:
            zone = self._zone_for(entity)
            return f"in the {zone.label}" if zone else ""
        edge = current_edge or last_edge
        target_entity = self.store.entities.get(edge.target)
        target_zone = self.store.zones.get(edge.target)
        relation = edge.relation

        if relation in {"INSIDE", "CONTAINED_IN"}:
            if target_entity is not None:
                phrase = f"inside the {target_entity.label}"
                container_edge = self._active_spatial_edge(target_entity, t)
                if container_edge is not None and depth < 2:
                    phrase += ", " + self._location_phrase(
                        target_entity, container_edge, container_edge, t, depth + 1
                    )
                else:
                    zone = self._zone_for(target_entity)
                    if zone:
                        phrase += f" in the {zone.label}"
            else:
                phrase = f"inside the {target_zone.label if target_zone else pretty_ref(edge.target)}"
            if edge.t_end is not None and edge.t_end < t - 0.05:
                phrase += f" (until {edge.t_end:.1f}s)"
            return phrase

        if target_zone is not None:
            phrase = f"resting on the {target_zone.label}"
            parent = self._zone_for(entity)
            if parent and parent.id != target_zone.id:
                phrase += f" in the {parent.label}"
            return phrase

        if target_entity is not None:
            phrase = f"on the {target_entity.label}"
            parent = self._zone_for(target_entity)
            if parent:
                phrase += f" in the {parent.label}"
            return phrase

        return f"on the {pretty_ref(edge.target)}"

    def _active_spatial_edge(self, entity: Entity, t: float) -> Edge | None:
        candidates = [
            edge
            for edge in self.store.edges.values()
            if edge.source == entity.id and edge.relation in SPATIAL_RELATIONS and edge.active_at(t)
        ]
        candidates.sort(key=lambda edge: (edge.relation not in {"INSIDE", "CONTAINED_IN"}, -edge.t_start))
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------ #
    # Citations / confidence
    # ------------------------------------------------------------------ #

    def _citations_for(
        self,
        entity: Entity,
        subgraph: dict[str, Any],
        t: float,
        last_edge: Edge | None,
        last_action: Edge | None,
        last_vlm: Any,
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        state_record = None
        for record in entity.history:
            if record.t <= t + 0.05:
                state_record = record
        if state_record is not None:
            citations.append(
                {
                    "t": round(state_record.t, 3),
                    "label": state_record.to_state,
                    "kind": "state",
                    "entity_id": entity.id,
                    "detail": state_record.reason,
                }
            )
        if last_edge is not None:
            citations.append(
                {
                    "t": round(last_edge.t_start, 3),
                    "label": last_edge.relation,
                    "kind": "edge",
                    "entity_id": entity.id,
                    "detail": f"{last_edge.source} → {last_edge.target}",
                }
            )
        if last_action is not None:
            citations.append(
                {
                    "t": round(last_action.t_start, 3),
                    "label": last_action.relation,
                    "kind": "action",
                    "entity_id": entity.id,
                    "detail": f"{last_action.source} → {last_action.target}",
                }
            )
        if last_vlm is not None:
            citations.append(
                {
                    "t": round(last_vlm.t, 3),
                    "label": last_vlm.action,
                    "kind": "interaction",
                    "entity_id": entity.id,
                    "detail": last_vlm.caption,
                }
            )
        for edge in subgraph["edges"]:
            if edge.relation in ACTION_RELATIONS and edge.target == entity.id and edge.t_start <= t + 0.05:
                citations.append(
                    {
                        "t": round(edge.t_start, 3),
                        "label": edge.relation,
                        "kind": "action",
                        "entity_id": entity.id,
                        "detail": f"{edge.source} → {edge.target}",
                    }
                )
        # de-duplicate and keep the most recent six
        seen: set[tuple[float, str]] = set()
        unique: list[dict[str, Any]] = []
        for citation in sorted(citations, key=lambda c: c["t"], reverse=True):
            key = (round(citation["t"], 1), citation["label"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(citation)
        return sorted(unique[:6], key=lambda c: c["t"])

    def _confidence(self, entity: Entity, edges: Iterable[Edge | None], vlm: Any) -> float | None:
        scores = [entity.clip_sim, entity.confidence or None]
        for edge in edges:
            if edge is not None and edge.clip_sim is not None:
                scores.append(edge.clip_sim)
        if vlm is not None and vlm.clip_sim is not None:
            scores.append(vlm.clip_sim)
        values = [score for score in scores if score is not None]
        return round(sum(values) / len(values), 3) if values else None

    # ------------------------------------------------------------------ #
    # Prompt / external LLM
    # ------------------------------------------------------------------ #

    def _build_prompt(self, query: str, table: str, entity: Entity, state: str, t: float) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"PLAYHEAD: {t:.1f}s\n"
            f"QUESTION: {query}\n\n"
            f"RESOLVED ENTITY: {entity.label} ({entity.id}), state={state}\n\n"
            f"2-HOP SUBGRAPH:\n{table}"
        )

    # ------------------------------------------------------------------ #
    # Local Qwen via Ollama
    # ------------------------------------------------------------------ #

    def _llm_enabled(self) -> bool:
        return self.llm_backend not in {"template", "none", "off"}

    def _ensure_chat_client(self) -> Any:
        if not self._llm_enabled():
            return None
        with self._llm_lock:
            if self._chat_client is not None:
                return self._chat_client
            if self._llm_checked:
                return None
            self._llm_checked = True
            try:
                from chronicle.semantic.config import SemanticConfig
                from chronicle.semantic.llm import OllamaChatClient, ensure_local_ollama
            except Exception as exc:  # pragma: no cover - defensive import guard
                self._llm_status = {
                    "server": False,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return None
            config = SemanticConfig(
                llm_backend="ollama",
                llm_model=self.llm_model,
                ollama_url=self.ollama_url,
            )
            try:
                ensure_local_ollama(config, wait_seconds=3.0)
            except Exception:
                pass
            self._chat_client = OllamaChatClient(
                model=self.llm_model,
                base_url=self.ollama_url,
                timeout_s=config.llm_timeout_s,
                keep_alive=config.llm_keep_alive,
                num_ctx=config.llm_num_ctx,
                num_predict=config.llm_num_predict,
                temperature=config.llm_temperature,
            )
            return self._chat_client

    def _probe_llm(self) -> dict[str, Any]:
        try:
            from chronicle.semantic.llm import probe_ollama_runtime
        except Exception as exc:  # pragma: no cover - defensive import guard
            return {"server": False, "available": False, "error": f"{type(exc).__name__}: {exc}"}
        return probe_ollama_runtime(self.ollama_url, self.llm_model, timeout=0.8)

    def _ask_local_llm(self, query: str, t: float, local: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_chat_client()
        if client is None:
            return {"llm_error": "local Ollama client is unavailable"}
        prompt = str(local.get("prompt") or "")
        if prompt.startswith(SYSTEM_PROMPT):
            prompt = prompt[len(SYSTEM_PROMPT) :].lstrip()
        if not prompt:
            prompt = (
                f"PLAYHEAD: {t:.1f}s\nQUESTION: {query}\n\n"
                f"2-HOP SUBGRAPH:\n{local.get('table') or '| no rows |'}"
            )
        try:
            raw = client.complete(SYSTEM_PROMPT, prompt)
        except Exception as exc:
            return {"llm_error": f"{type(exc).__name__}: {exc}"}
        active = str(getattr(client, "active_model", "") or self.llm_model)

        answer, citations, model = self._parse_external(raw)
        if answer and answer.lstrip().startswith("{"):
            salvaged = self._salvage_json_answer(raw)
            if salvaged is not None:
                answer, citations = salvaged
        if not answer or answer.strip().lower().rstrip(".") == NO_MEMORY_PHRASE:
            return {"llm_error": "Qwen returned no grounded answer"}
        citations = citations or local.get("citations") or []
        answer = self._ensure_citation(answer, citations)
        return {
            "answer": answer,
            "citations": citations,
            "provider": f"ollama:{model or active}",
            "model": model or active,
        }

    @staticmethod
    def _ensure_citation(answer: str, citations: list[dict[str, Any]]) -> str:
        """Keep the answer seekable: attach the newest evidence stamp if absent."""
        if re.search(r"\[t=|\bt=\d", answer):
            return answer
        stamped = [
            citation
            for citation in citations
            if isinstance(citation, dict) and citation.get("t") is not None
        ]
        if not stamped:
            return answer
        latest = max(stamped, key=lambda citation: float(citation.get("t") or 0.0))
        return f"{answer.rstrip('. ')} [t={float(latest['t']):.1f}s]."

    @staticmethod
    def _salvage_json_answer(raw: str) -> tuple[str, list[dict[str, Any]]] | None:
        """Recover ``{"answer": ...}`` from slightly malformed model JSON.

        Small local models often emit ``[t=4.8s]`` inside JSON strings or forget
        to quote numeric stamps, which ``json.loads`` rejects.  Keep the prose
        and lift whatever citations are readable.
        """
        match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if match is None:
            return None
        try:
            answer = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            answer = match.group(1).replace('\\"', '"')
        citations: list[dict[str, Any]] = []
        for object_match in re.finditer(r"\{[^{}]*?\"t\"[^{}]*?\}", raw):
            body = object_match.group(0)
            stamp = re.search(r'"t"\s*:\s*"?(\d+(?:\.\d+)?)s?"?', body)
            if stamp is None:
                continue
            label = re.search(r'"label"\s*:\s*"([^"]*)"', body)
            entity_id = re.search(r'"entity_id"\s*:\s*"([^"]*)"', body)
            detail = re.search(r'"detail"\s*:\s*"([^"]*)"', body)
            citations.append(
                {
                    "t": round(float(stamp.group(1)), 3),
                    "label": label.group(1) if label else "STATE",
                    "kind": "action" if label else "state",
                    "entity_id": entity_id.group(1) if entity_id else None,
                    "detail": detail.group(1) if detail else "",
                }
            )
        if not citations and re.search(r"\[t=|\bt=\d", raw):
            for stamp in re.finditer(r"\[t=(\d+(?:\.\d+)?)s\]", raw):
                citations.append(
                    {
                        "t": round(float(stamp.group(1)), 3),
                        "label": "TIMESTAMP",
                        "kind": "state",
                        "entity_id": None,
                        "detail": "",
                    }
                )
        return answer.strip(), citations

    def _ask_external(self, query: str, t: float, local: dict[str, Any]) -> dict[str, Any] | None:
        if not self.endpoint:
            return None
        payload = {
            "schema_version": 1,
            "query": query,
            "playhead": round(t, 3),
            "prompt": local.get("prompt") or "",
            "serialization": local.get("table") or "",
            "subgraph": local.get("subgraph") or {},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv("CHRONICLE_QUERY_TIMEOUT", "10"))) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "provider": "local-graphrag",
                "model": "chronicle-local",
                "llm_error": f"{type(exc).__name__}: {exc}",
            }

        answer, citations, model = self._parse_external(body)
        if not answer:
            return {
                "provider": "local-graphrag",
                "model": "chronicle-local",
                "llm_error": "LLM endpoint returned no usable answer",
            }

        merged_citations = citations or local.get("citations") or []
        return {
            "answer": answer,
            "citations": merged_citations,
            "model": model or "role-c",
            "llm_error": None,
        }

    @staticmethod
    def _parse_external(body: str) -> tuple[str | None, list[dict[str, Any]], str | None]:
        body = body.strip()
        if not body:
            return None, [], None
        if body.startswith("```"):
            body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
            body = re.sub(r"\s*```$", "", body)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return _HTML_RE.sub("", body), [], None
        if not isinstance(data, dict):
            return str(data), [], None

        model = data.get("model")
        candidate: Any = None
        if isinstance(data.get("choices"), list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message") or {}
                candidate = message.get("content") if isinstance(message, dict) else choice.get("text")
        for key in ("answer", "response", "text", "output", "result"):
            if candidate is None and isinstance(data.get(key), str):
                candidate = data[key]
        if isinstance(candidate, str):
            text = candidate.strip()
            if text.startswith("{"):
                try:
                    nested = json.loads(text)
                    if isinstance(nested, dict) and isinstance(nested.get("answer"), str):
                        data = nested
                        text = nested["answer"]
                except json.JSONDecodeError:
                    pass
            citations = _coerce_citations(data.get("citations"))
            return _HTML_RE.sub("", text), citations, model
        return None, [], model

    @staticmethod
    def _rows_to_table(rows: list[dict[str, Any]]) -> str:
        lines = ["| NODE | STATE | RELATION | TARGET | T_START | T_END | CLIP |", "|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(
                "| {node} | {state} | {relation} | {target} | {t_start:.1f}s | {t_end} | {clip} |".format(
                    node=row.get("node", ""),
                    state=row.get("state", ""),
                    relation=row.get("relation", ""),
                    target=row.get("target", ""),
                    t_start=float(row.get("t_start") or 0),
                    t_end="—" if row.get("t_end") is None else f"{float(row['t_end']):.1f}s",
                    clip="—" if row.get("clip_sim") is None else f"{float(row['clip_sim']):.2f}",
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _wants_when(query: str) -> bool:
        return bool(re.search(r"\b(when|what time|how long ago|last touch|last interact)\b", query.lower()))


def _coerce_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("t"))
        except (TypeError, ValueError):
            continue
        citations.append(
            {
                "t": round(t, 3),
                "label": clean_text(item.get("label") or "EVIDENCE", 60),
                "kind": clean_text(item.get("kind") or "graph", 20),
                "entity_id": item.get("entity_id"),
                "detail": clean_text(item.get("detail"), 200),
            }
        )
    return citations
