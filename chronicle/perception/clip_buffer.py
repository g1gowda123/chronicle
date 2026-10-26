"""30-frame RAM ring buffer and structured interaction-clip assembly."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from chronicle.perception.config import ClipConfig
from chronicle.types import (
    BBox,
    ClipFrame,
    HandObservation,
    InteractionClip,
    InteractionEvent,
    TrackedObject,
)


@dataclass(slots=True)
class FrameSnapshot:
    frame_index: int
    timestamp: float
    image: np.ndarray
    boxes: dict[int, BBox] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    hand_points: dict[str, np.ndarray] = field(default_factory=dict)

    def track_box(self, track_id: int) -> BBox | None:
        return self.boxes.get(track_id)

    def hand(self, hand: str) -> np.ndarray | None:
        return self.hand_points.get(hand.lower())

    def clip_frame(self, track_id: int, hand: str) -> ClipFrame:
        return ClipFrame(
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            image=self.image,
            target_box=self.track_box(track_id),
            hand_points=self.hand(hand),
        )


class FrameRingBuffer:
    """Bounded deque of downscaled frame snapshots.

    Keeping pixels at 640 px (rather than full 1080p) holds a 30-frame window
    to about 20 MB while still giving the VLM a usable temporal strip.
    """

    def __init__(self, maxlen: int = 30, max_width: int = 640) -> None:
        self.maxlen = max(1, int(maxlen))
        self.max_width = max(1, int(max_width))
        self._frames: deque[FrameSnapshot] = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def append(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        tracks: list[TrackedObject],
        hands: list[HandObservation],
    ) -> FrameSnapshot:
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / max(1, width))
        if scale < 0.999:
            image = cv2.resize(
                frame,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image = frame.copy()

        boxes = {
            track.track_id: BBox(
                track.bbox.x1 * scale,
                track.bbox.y1 * scale,
                track.bbox.x2 * scale,
                track.bbox.y2 * scale,
            )
            for track in tracks
        }
        labels = {track.track_id: track.label for track in tracks}
        hand_points = {
            hand.handedness.lower(): hand.landmarks * scale for hand in hands
        }
        snapshot = FrameSnapshot(
            frame_index=frame_index,
            timestamp=timestamp,
            image=image,
            boxes=boxes,
            labels=labels,
            hand_points=hand_points,
        )
        self._frames.append(snapshot)
        return snapshot

    def snapshot(self) -> list[FrameSnapshot]:
        return list(self._frames)

    def last(self, count: int) -> list[FrameSnapshot]:
        if count <= 0:
            return []
        return list(self._frames)[-count:]


@dataclass
class _ClipBuilder:
    clip_id: str
    track_id: int
    label: str
    hand: str
    start_ts: float
    trigger: str
    max_ios: float
    frames: list[FrameSnapshot]
    contact_index: int
    ended: bool = False
    end_ts: float | None = None
    end_frame: int | None = None

    def add(self, snapshot: FrameSnapshot) -> None:
        if self.frames and snapshot.frame_index <= self.frames[-1].frame_index:
            return
        self.frames.append(snapshot)

    def mark_end(self, event: InteractionEvent) -> None:
        self.ended = True
        self.end_ts = event.timestamp
        self.end_frame = event.frame_index
        self.max_ios = max(self.max_ios, event.ios)


class InteractionClipRecorder:
    """Turn dwell-gated interaction events into 1x3-ready temporal clips."""

    def __init__(self, config: ClipConfig | None = None) -> None:
        self.config = config or ClipConfig()
        self._pending: dict[str, _ClipBuilder] = {}
        self._counter = 0

    def _start(
        self, event: InteractionEvent, buffer: FrameRingBuffer
    ) -> _ClipBuilder | None:
        if not self.config.enabled:
            return None
        pre_roll = max(0, min(int(self.config.pre_roll), buffer.maxlen))
        snapshot_frames = buffer.snapshot()
        if not snapshot_frames:
            return None
        # The start event arrives only after the dwell gate, so the ring already
        # contains both pre-contact and contact frames. Keep the last
        # ``pre_roll`` frames from before contact plus everything since, rather
        # than blindly taking the last N frames (which would be all contact).
        before = [frame for frame in snapshot_frames if frame.timestamp < event.start_timestamp]
        after = [frame for frame in snapshot_frames if frame.timestamp >= event.start_timestamp]
        pre_frames = before[-pre_roll:] if pre_roll else []
        max_length = max(1, int(self.config.length))
        if len(pre_frames) >= max_length:
            pre_frames = pre_frames[-max_length:]
            after = []
        remaining = max(0, max_length - len(pre_frames))
        frames = pre_frames + after[:remaining]
        if not frames:
            frames = snapshot_frames[-max(1, min(pre_roll or 1, max_length)):]
        self._counter += 1
        clip_id = f"clip_{self._counter:04d}_t{int(round(event.start_timestamp * 1000)):08d}"
        contact_index = min(
            range(len(frames)),
            key=lambda index: abs(frames[index].timestamp - event.start_timestamp),
        )
        builder = _ClipBuilder(
            clip_id=clip_id,
            track_id=event.track_id,
            label=event.label,
            hand=event.hand.lower(),
            start_ts=event.start_timestamp,
            trigger=event.trigger,
            max_ios=event.ios,
            frames=frames,
            contact_index=contact_index,
        )
        return builder

    def _finalize(self, builder: _ClipBuilder, timestamp: float) -> InteractionClip:
        frames = [snapshot.clip_frame(builder.track_id, builder.hand) for snapshot in builder.frames]
        end_ts = builder.end_ts if builder.ended and builder.end_ts is not None else (
            frames[-1].timestamp if frames else timestamp
        )
        clip = InteractionClip(
            clip_id=builder.clip_id,
            track_id=builder.track_id,
            label=builder.label,
            hand=builder.hand,
            # start_ts is the window's first frame (including pre-roll); the
            # dwell-gated contact time is kept separately as contact_start_ts.
            start_ts=frames[0].timestamp if frames else builder.start_ts,
            end_ts=max(end_ts, builder.start_ts),
            trigger=builder.trigger,
            frames=frames,
            contact_index=max(0, min(builder.contact_index, len(frames) - 1)),
            max_ios=builder.max_ios,
            metadata={
                "start_frame": frames[0].frame_index if frames else None,
                "end_frame": frames[-1].frame_index if frames else None,
                "contact_start_ts": builder.start_ts,
                "ended": builder.ended,
                "pre_roll": self.config.pre_roll,
                "post_roll": self.config.post_roll,
            },
        )
        clip.build_composite()
        return clip

    def update(
        self,
        buffer: FrameRingBuffer,
        snapshot: FrameSnapshot,
        events: list[InteractionEvent],
    ) -> list[InteractionClip]:
        if not self.config.enabled:
            return []
        started: set[str] = set()
        for event in events:
            if event.kind == "start":
                builder = self._start(event, buffer)
                if builder is not None:
                    self._pending[builder.clip_id] = builder
                    started.add(builder.clip_id)
            elif event.kind == "end":
                for builder in self._pending.values():
                    if builder.track_id == event.track_id and builder.hand == event.hand.lower():
                        builder.mark_end(event)

        for clip_id, builder in self._pending.items():
            if clip_id not in started:
                builder.add(snapshot)

        finished: list[InteractionClip] = []
        while len(self._pending) > max(1, int(self.config.max_pending)):
            oldest_id = next(iter(self._pending))
            finished.append(self._finalize(self._pending[oldest_id], snapshot.timestamp))
            del self._pending[oldest_id]

        for clip_id, builder in list(self._pending.items()):
            if not builder.frames:
                del self._pending[clip_id]
                continue
            too_long = len(builder.frames) >= max(1, int(self.config.length))
            settled = (
                builder.ended
                and builder.end_frame is not None
                and snapshot.frame_index - builder.end_frame >= max(0, int(self.config.post_roll))
            )
            if too_long or settled:
                finished.append(self._finalize(builder, snapshot.timestamp))
                del self._pending[clip_id]
        return finished

    def flush(self, timestamp: float | None = None) -> list[InteractionClip]:
        clips: list[InteractionClip] = []
        for builder in self._pending.values():
            clips.append(self._finalize(builder, timestamp or builder.frames[-1].timestamp))
        self._pending.clear()
        return clips

    @property
    def pending(self) -> int:
        return len(self._pending)
