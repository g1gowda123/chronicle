"""Lightweight transcript evidence for audio-grounded memory.

The long-video evaluator can attach a Whisper transcript to a runtime.  The
runtime uses it only for questions that explicitly ask what someone said or
whether an action was corrected; visual questions continue through the graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptIndex:
    """Time-indexed speech segments with tolerant nearest-segment lookup."""

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self.segments = tuple(sorted(segments, key=lambda segment: segment.start))

    @classmethod
    def from_json(cls, path: str | Path) -> TranscriptIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
        segments = [
            TranscriptSegment(
                start=float(row["start"]),
                end=float(row["end"]),
                text=str(row["text"]).strip(),
            )
            for row in rows
            if str(row.get("text", "")).strip()
        ]
        return cls(segments)

    def at(self, timestamp: float, *, tolerance: float = 10.0) -> TranscriptSegment | None:
        """Return the segment containing ``timestamp`` or the nearest one nearby."""
        if not self.segments:
            return None
        containing = [
            segment
            for segment in self.segments
            if segment.start <= timestamp <= segment.end
        ]
        if containing:
            return min(containing, key=lambda segment: segment.end - segment.start)
        nearest = min(
            self.segments,
            key=lambda segment: min(
                abs(segment.start - timestamp), abs(segment.end - timestamp)
            ),
        )
        distance = min(abs(nearest.start - timestamp), abs(nearest.end - timestamp))
        return nearest if distance <= tolerance else None
