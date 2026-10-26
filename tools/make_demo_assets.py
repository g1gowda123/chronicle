"""Build the self-contained judge demo used by the Role D dashboard.

Outputs (all under ``data/``):

* ``demo.webm``        synthetic egocentric episode (26 s, 15 fps)
* ``demo_events.jsonl`` timeline in the exact pipeline event contract
* ``crops/*.png``      object crops for the inspector drawer
* ``strips/*.png``     1x3 pre/contact/post strips for VLM evidence
* ``zones/*.png``      keyframes for topological anchors

Run:  python tools/make_demo_assets.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_scene as scene  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CROPS = DATA / "crops"
STRIPS = DATA / "strips"
ZONES = DATA / "zones"

FPS = scene.FPS
DURATION = scene.DURATION

EVENTS: list[dict] = []
VIDEO_URL = "/media/demo/chronicle_demo.mp4"

ENTITY_META = {
    "wallet": ("wallet_1", "Wallet", 0.89, 0.87, "101", "wallet.png"),
    "bottle": ("bottle_1", "Bottle", 0.88, 0.84, "102", "bottle.png"),
    "notebook": ("notebook_1", "Notebook", 0.93, 0.9, "103", "notebook.png"),
    "mug": ("mug_1", "Mug", 0.91, 0.92, "104", "mug.png"),
    "phone": ("phone_1", "Phone", 0.95, 0.94, "105", "phone.png"),
    "keys": ("keys_1", "Keys", 0.94, 0.91, "106", "keys.png"),
    "tote": ("tote_1", "Tote Bag", 0.92, 0.9, "107", "tote.png"),
    "backpack": ("backpack_1", "Backpack", 0.9, 0.88, "108", "backpack.png"),
}


# --------------------------------------------------------------------------- #
# Event helpers
# --------------------------------------------------------------------------- #


def emit(t: float, kind: str, data: dict, *, order: int = 5, intro: bool = False) -> None:
    EVENTS.append(
        {
            "v": 1,
            "kind": kind,
            "t": round(max(0.0, t), 3),
            "data": data,
            "_order": order,
            "_intro": intro,
        }
    )


def log(
    t: float,
    channel: str,
    message: str,
    entity_id: str | None = None,
    *,
    level: str = "info",
    **data,
) -> None:
    payload = {"channel": channel, "level": level, "message": message}
    if entity_id:
        payload["entity_id"] = entity_id
    payload.update({key: value for key, value in data.items() if value is not None})
    emit(t, "telemetry", payload, order=7)


def entity_intro(name: str, t: float, state: str | None = None, **extra) -> None:
    eid, label, clip, conf, track, crop = ENTITY_META[name]
    data = {
        "id": eid,
        "label": label,
        "class": name,
        "clip_sim": clip,
        "confidence": conf,
        "track_id": track,
        "zone_id": "kitchen",
        "crop_url": f"/media/crops/{crop}",
        "first_seen": round(t, 3),
        "attributes": {"source": "YOLO-World", "reid": "CLIP ViT-B-32"},
    }
    bbox = scene.normalised_bbox(name, t)
    if bbox:
        data["bbox"] = bbox
    if state:
        data["state"] = state
    data.update(extra)
    emit(t, "entity", data, order=2, intro=True)
    log(
        t + 0.001,
        "TRACK",
        f"minted {eid} ({name}) · state {state or 'NEW'} · conf {conf:.2f} · ByteTrack {track}",
        eid,
    )


def pose_updates(name: str, start: float, end: float, step: float = 0.4) -> None:
    eid, label, clip, conf, _track, _crop = ENTITY_META[name]
    t = start
    while t <= end + 1e-6:
        if name == "keys" and t > scene.HELD_KEYS[1]:
            t += step
            continue
        bbox = scene.normalised_bbox(name, t)
        if bbox is not None:
            emit(
                t,
                "entity",
                {
                    "id": eid,
                    "label": label,
                    "bbox": bbox,
                    "confidence": round(conf - 0.02 * math.sin(t), 3),
                    "clip_sim": clip,
                },
                order=8,
            )
        t += step


def state(
    t: float,
    name: str,
    from_state: str,
    to_state: str,
    reason: str,
    *,
    confidence: float | None = None,
    **trigger,
) -> None:
    eid, label, _clip, conf, _track, _crop = ENTITY_META[name]
    emit(
        t,
        "state",
        {
            "entity_id": eid,
            "label": label,
            "from": from_state,
            "to": to_state,
            "reason": reason,
            "trigger": {key: value for key, value in trigger.items() if value is not None},
            "confidence": confidence if confidence is not None else conf,
        },
        order=3,
    )


def edge(
    t: float,
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    active: bool = True,
    clip_sim: float | None = None,
) -> None:
    emit(
        t,
        "edge",
        {
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "t_start": round(t if t_start is None else t_start, 3),
            "t_end": None if t_end is None else round(t_end, 3),
            "active": active,
            "clip_sim": clip_sim,
        },
        order=4,
    )


def interaction(
    t: float,
    name: str,
    action: str,
    caption: str,
    *,
    model: str = "Florence-2-base",
    clip_sim: float,
    latency_ms: int,
    strip: str | None,
    crop: str | None,
) -> None:
    eid, _label, _clip, _conf, _track, _crop = ENTITY_META[name]
    emit(
        t,
        "interaction",
        {
            "entity_id": eid,
            "action": action,
            "caption": caption,
            "clip_sim": clip_sim,
            "strip_url": f"/media/strips/{strip}" if strip else None,
            "crop_url": f"/media/crops/{crop}" if crop else None,
            "model": model,
            "latency_ms": latency_ms,
        },
        order=5,
    )


def zone(t: float, zone_id: str, label: str, clip: float, keyframe: str, bbox, *, active: bool = False) -> None:
    emit(
        t,
        "zone",
        {
            "id": zone_id,
            "label": label,
            "clip_sim": clip,
            "confidence": clip,
            "keyframe_url": f"/media/zones/{keyframe}",
            "bbox": bbox,
            "first_seen": round(t, 3),
            "last_seen": round(t, 3),
            "active": active,
        },
        order=1,
    )
    log(t + 0.001, "GRAPH", f'zone "{label}" upserted · VPR CLIP {clip:.2f} · anchor active', **{"zone_id": zone_id})


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


def build_timeline(video_url: str) -> None:
    global VIDEO_URL
    VIDEO_URL = video_url

    emit(
        0.0,
        "session",
        {
            "title": "Kitchen Counter · Episode 01",
            "video_url": video_url,
            "duration": DURATION,
            "fps": FPS,
            "width": scene.SCREEN_W,
            "height": scene.SCREEN_H,
            "source": "demo",
            "notes": "Synthetic egocentric episode · judge telemetry fallback",
        },
        order=0,
    )
    log(0.0, "SYSTEM", "Chronicle online · YOLO-World · ByteTrack · MediaPipe Hands · Florence-2-base")
    log(0.1, "SYSTEM", f"source {video_url} · {scene.SCREEN_W}x{scene.SCREEN_H} @ {FPS}fps · 30-frame ring buffer")

    # ---- pre-roll: dormant wallet with retained terminal anchor ------- #
    entity_intro("wallet", 0.3, first_seen=-600.0, last_seen=-598.0)
    state(
        0.32,
        "wallet",
        "OUT_OF_VIEW",
        "LOST",
        "unobserved 612s > 600s · terminal anchor preserved",
        dt=612.0,
        dormancy_threshold=600.0,
    )
    edge(0.35, "e_wallet_shelf", "wallet_1", "shelf", "RESTING_ON", t_start=-600.0, clip_sim=0.89)
    log(
        0.4,
        "GRAPH",
        "terminal anchor retained · wallet_1 --[RESTING_ON]--> shelf (row pinned in SQLite)",
        "wallet_1",
    )

    # ---- bottle: OUT_OF_VIEW during the pan, reacquired on return ------ #
    entity_intro("bottle", 0.45)
    state(2.0, "bottle", "NEW", "VISIBLE_STATIC", "velocity 0.03px/frame < τ_move · support ymax contact", dt=0.6, velocity=0.03)
    edge(2.02, "e_bottle_counter_1", "bottle_1", "countertop", "RESTING_ON", t_start=2.0, clip_sim=0.88)

    # ---- camera pan in ------------------------------------------------- #
    for t in (0.5, 1.25, 2.0, 2.75, 3.5, 4.25, 4.9):
        speed = abs(scene.camera_x(t + 1 / FPS) - scene.camera_x(t))
        if speed > 0.6:
            log(
                t,
                "GEOM",
                f"‖T‖={speed:.2f}px > τ_pan=2.0 · RANSAC inliers={196 + int(t * 7) % 48}/256 · camera=PAN",
                **{"dx": round(speed, 2), "dy": 0.0, "theta": 0.0},
            )
        else:
            log(t, "GEOM", f"‖T‖={speed:.2f}px < τ_still=2.5 · camera=STATIONARY · Δθ=0.00°")

    # ---- objects minted as the camera settles ------------------------- #
    intro_times = {
        "notebook": 4.55,
        "mug": 4.62,
        "phone": 4.7,
        "tote": 4.76,
        "keys": 4.84,
        "backpack": 4.9,
    }
    for name, t in intro_times.items():
        entity_intro(name, t)

    for name, t in intro_times.items():
        log(t + 0.001, "TRACK", f"{ENTITY_META[name][0]} pending CLIP Re-ID check · cosine < 0.82 to active nodes → new node")

    log(5.0, "GEOM", "‖T‖=0.34px < τ_still=2.5 · camera=STATIONARY · 30-frame keyframe captured")
    log(5.3, "TRACK", "ByteTrack: 8 tracks · 0 unassigned · CLIP Re-ID idle (no duplicates)")

    # ---- zone relocalisation ------------------------------------------ #
    zone(5.02, "kitchen", "Kitchen", 0.93, "kitchen.png", [0.0, 0.0, 1.0, 1.0], active=True)
    zone(5.06, "countertop", "Countertop", 0.9, "countertop.png", [0.0, 0.55, 1.0, 1.0])
    zone(5.1, "shelf", "Pantry Shelf", 0.88, "shelf.png", [0.0, 0.18, 0.14, 0.42])
    log(
        5.04,
        "GRAPH",
        "VPR relocalization · Kitchen CLIP 0.93 ≥ 0.85 → existing zone; 0 new nodes",
    )
    log(5.65, "GRAPH", "SQLite commit · 8 entity nodes / 8 spatial edges · 12ms")

    # ---- static confirmations ----------------------------------------- #
    state(5.32, "notebook", "NEW", "VISIBLE_STATIC", "velocity 0.02px/frame · support ymax contact", dt=0.62, velocity=0.02)
    state(5.38, "mug", "NEW", "VISIBLE_STATIC", "velocity 0.01px/frame · support ymax contact", dt=0.58, velocity=0.01)
    state(5.44, "phone", "NEW", "VISIBLE_STATIC", "velocity 0.02px/frame · support ymax contact", dt=0.54, velocity=0.02)
    state(5.5, "tote", "NEW", "VISIBLE_STATIC", "velocity 0.03px/frame · support ymax contact", dt=0.52, velocity=0.03)
    state(5.56, "keys", "NEW", "VISIBLE_STATIC", "velocity 0.02px/frame · support ymax contact", dt=0.5, velocity=0.02)
    state(5.62, "backpack", "NEW", "VISIBLE_STATIC", "velocity 0.04px/frame · support ymax contact", dt=0.5, velocity=0.04)

    for name, t in (("notebook", 5.33), ("mug", 5.39), ("phone", 5.45), ("tote", 5.51), ("keys", 5.57), ("backpack", 5.63)):
        edge(t, f"e_{name}_counter", ENTITY_META[name][0], "countertop", "RESTING_ON", t_start=t, clip_sim=ENTITY_META[name][2])

    # ---- mug tap: INTERACTED -> OCCLUDED -> VISIBLE_STATIC ------------- #
    state(
        8.0,
        "mug",
        "VISIBLE_STATIC",
        "INTERACTED",
        "fingertip inside convex hull · IoS 0.64 sustained Δt=0.48s",
        ios=0.64,
        dt=0.48,
        hand_contact=True,
    )
    interaction(
        8.25,
        "mug",
        "TOUCHED",
        "A hand brushes the rim of a ceramic mug resting on the wooden counter.",
        clip_sim=0.83,
        latency_ms=640,
        strip="mug_touch.png",
        crop="mug.png",
    )
    state(8.55, "mug", "INTERACTED", "OCCLUDED", "track lost · camera STATIONARY · last bbox overlaps hand hull", dt=0.3)
    state(9.05, "mug", "OCCLUDED", "VISIBLE_STATIC", "track reacquired · ymax support contact", dt=0.2)

    # ---- keys: pick up, carry, insert into tote ----------------------- #
    log(10.05, "VLM", "gate closed: hand contact 0.18s < Δt=0.30s · keys_1 · no composite strip dispatched")
    log(10.2, "TRACK", "hand landmarks 21/21 · wrist (0.50,0.84) · index tip inside keys_1 hull")
    state(
        10.2,
        "keys",
        "VISIBLE_STATIC",
        "INTERACTED",
        "fingertip inside convex hull · IoS 0.71 sustained Δt=0.42s",
        ios=0.71,
        dt=0.42,
        hand_contact=True,
    )
    edge(10.2, "e_keys_counter", "keys_1", "countertop", "RESTING_ON", t_start=5.57, t_end=10.2, active=False)
    state(
        10.9,
        "keys",
        "INTERACTED",
        "HELD",
        "centroid displacement matches hand velocity vector · cos 0.93",
        ios=0.74,
        velocity=1.8,
        cos=0.93,
    )
    interaction(
        11.15,
        "keys",
        "PICK_UP",
        "A hand grasps a set of keys resting on the counter and lifts them clear of the surface.",
        clip_sim=0.94,
        latency_ms=910,
        strip="keys_pick_up.png",
        crop="keys_hand.png",
    )
    edge(11.15, "a_user_keys_pick", "USER", "keys_1", "PICKED_UP", clip_sim=0.94)
    log(
        11.18,
        "VLM",
        "Florence-2 <MORE_DETAILED_CAPTION> · 1×3 strip 10.6/11.15/11.7s · PICK_UP conf 0.94 · 910ms",
        "keys_1",
    )
    log(16.9, "TRACK", "keys_1 track persists · IoS(hand,obj)=0.86 · tote_1 hull overlap 0.93")
    state(
        16.9,
        "keys",
        "HELD",
        "CONTAINED",
        "object boundary collapses inside tote hull · IoS 0.93",
        ios=0.93,
        dt=0.35,
    )
    edge(16.92, "e_keys_tote", "keys_1", "tote_1", "INSIDE", t_start=16.9, clip_sim=0.93)
    log(16.95, "GRAPH", "terminal anchor preserved · keys_1 --[INSIDE]--> tote_1 · t=16.9s", "keys_1")
    interaction(
        17.15,
        "keys",
        "INSERT_INTO",
        "The hand lowers the keys through the opening of a canvas tote bag.",
        clip_sim=0.91,
        latency_ms=870,
        strip="keys_insert.png",
        crop="keys_tote.png",
    )
    edge(17.15, "a_user_keys_place", "USER", "keys_1", "PLACED_IN", clip_sim=0.91)
    log(17.2, "GRAPH", "action edge USER --[PLACED_IN]--> keys_1 @ t=17.15s · SQLite commit 9ms", "keys_1")
    state(16.15, "tote", "VISIBLE_STATIC", "INTERACTED", "hand enters container hull · IoS 0.42", ios=0.42, dt=0.32)
    state(17.35, "tote", "INTERACTED", "VISIBLE_STATIC", "hand withdrawn · ymax support contact", dt=0.25)

    # ---- phone: pick up, inspect, put down ---------------------------- #
    log(18.7, "VLM", "gate closed: hand contact 0.14s < Δt=0.30s · phone_1 · no composite strip dispatched")
    log(18.85, "TRACK", "hand landmarks 21/21 · index tip inside phone_1 hull · IoS 0.58")
    state(
        18.9,
        "phone",
        "VISIBLE_STATIC",
        "INTERACTED",
        "fingertip inside convex hull · IoS 0.58 sustained Δt=0.36s",
        ios=0.58,
        dt=0.36,
        hand_contact=True,
    )
    edge(18.9, "e_phone_counter_1", "phone_1", "countertop", "RESTING_ON", t_start=5.45, t_end=18.9, active=False)
    state(19.4, "phone", "INTERACTED", "HELD", "object centroid tracks palm · cos 0.95", ios=0.69, velocity=1.2, cos=0.95)
    interaction(
        19.75,
        "phone",
        "PICK_UP",
        "A hand picks up a black smartphone from the counter and raises it toward the camera.",
        clip_sim=0.9,
        latency_ms=780,
        strip="phone_pick_up.png",
        crop="phone_hand.png",
    )
    edge(19.75, "a_user_phone_pick", "USER", "phone_1", "PICKED_UP", clip_sim=0.9)
    state(21.7, "phone", "HELD", "VISIBLE_STATIC", "released over counter · ymax support contact", dt=0.25)
    state(21.95, "phone", "VISIBLE_STATIC", "VISIBLE_MOVING", "residual slide 0.6px/frame · IoS(Hand,Obj)=0.04 < 0.10", velocity=0.6, ios=0.04)
    interaction(
        22.1,
        "phone",
        "PUT_DOWN",
        "The hand places the smartphone back onto the wooden countertop.",
        clip_sim=0.88,
        latency_ms=820,
        strip="phone_put_down.png",
        crop="phone.png",
    )
    edge(22.1, "a_user_phone_put", "USER", "phone_1", "PUT_DOWN", clip_sim=0.88)
    edge(21.75, "e_phone_counter_2", "phone_1", "countertop", "RESTING_ON", t_start=21.75, clip_sim=0.89)
    state(22.4, "phone", "VISIBLE_MOVING", "VISIBLE_STATIC", "velocity 0.02px/frame for Δt=0.45s", dt=0.45, velocity=0.02)

    # ---- pan back out: bottle reacquired ------------------------------ #
    log(23.0, "SYSTEM", "camera pan out · ‖T‖>τ_pan · detection gated, anchors retained")
    state(
        5.05,
        "bottle",
        "VISIBLE_STATIC",
        "OUT_OF_VIEW",
        "track lost during camera pan ‖T‖=4.8px > τ_pan · spatial anchor retained",
        dt=0.2,
        camera_pan=4.8,
    )
    edge(5.05, "e_bottle_counter_1", "bottle_1", "countertop", "RESTING_ON", t_start=2.0, t_end=5.05, active=False)
    log(25.2, "GEOM", "‖T‖=0.6px < τ_still=2.5 · camera=STATIONARY · relocalization keyframe captured")
    state(
        25.45,
        "bottle",
        "OUT_OF_VIEW",
        "VISIBLE_STATIC",
        "CLIP Re-ID 0.91 after pan settles · countertop support re-validated",
        dt=20.4,
        clip_sim=0.91,
    )
    edge(25.45, "e_bottle_counter_2", "bottle_1", "countertop", "RESTING_ON", t_start=25.45, clip_sim=0.91)
    log(25.5, "GRAPH", "re-bound bottle_1 via CLIP Re-ID 0.91 after 20.4s OUT_OF_VIEW", "bottle_1")
    log(25.6, "GRAPH", "VPR relocalization · Kitchen CLIP 0.94 ≥ 0.85 · 0 new zone nodes")
    log(25.7, "GRAPH", "2-hop subgraph index warm · 8 entities / 16 edges · avg query 38ms")

    # ---- pose stream (bbox micro-trajectories) ------------------------ #
    pose_updates("bottle", 0.5, 25.9)
    for name in ("notebook", "mug", "phone", "tote", "keys", "backpack"):
        pose_updates(name, intro_times[name] + 0.15, DURATION - 0.1)

    # ---- metrics ------------------------------------------------------ #
    _emit_metrics()


def _emit_metrics() -> None:
    vlm_times = sorted(event["t"] for event in EVENTS if event["kind"] == "interaction")
    t = 0.5
    while t <= DURATION:
        speed = abs(scene.camera_x(t + 1 / FPS) - scene.camera_x(t))
        frames = int(t * FPS)
        gated = int(frames * 0.79)
        calls = sum(1 for value in vlm_times if value <= t)
        last_latency = 0
        for event in EVENTS:
            if event["kind"] == "interaction" and event["t"] <= t:
                last_latency = event["data"].get("latency_ms", 0)
        emit(
            t,
            "metrics",
            {
                "fps": FPS,
                "latency_ms": 44 + int(9 * abs(math.sin(t * 1.7))),
                "last_latency_ms": last_latency,
                "frames": frames,
                "gated_frames": gated,
                "yolo_frames": frames - gated,
                "detections": 7,
                "track_count": 8,
                "vlm_calls": calls,
                "camera": {
                    "state": "PAN" if speed > 0.6 else "STATIONARY",
                    "dx": round(speed, 2),
                    "dy": 0.0,
                    "theta": 0.0,
                    "inliers": 214,
                },
            },
            order=6,
        )
        t += 0.5


# --------------------------------------------------------------------------- #
# Asset rendering
# --------------------------------------------------------------------------- #


def render_assets() -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    STRIPS.mkdir(parents=True, exist_ok=True)
    ZONES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    crops = [
        ("wallet", 0.8, "wallet.png", 10),
        ("bottle", 1.2, "bottle.png", 10),
        ("bottle", 25.7, "bottle_reacquired.png", 10),
        ("notebook", 6.0, "notebook.png", 14),
        ("mug", 6.2, "mug.png", 12),
        ("phone", 6.0, "phone.png", 10),
        ("keys", 6.0, "keys.png", 12),
        ("tote", 6.2, "tote.png", 12),
        ("backpack", 6.0, "backpack.png", 14),
        ("keys", 12.2, "keys_hand.png", 16),
        ("tote", 17.2, "keys_tote.png", 14),
        ("phone", 20.0, "phone_hand.png", 14),
    ]
    for name, t, filename, pad in crops:
        scene.save_crop(t, name, CROPS / filename, pad=pad)
        print(f"  crop  {filename}")

    strips = [
        ("mug", 8.25, "mug_touch.png"),
        ("keys", 11.15, "keys_pick_up.png"),
        ("tote", 17.15, "keys_insert.png"),
        ("phone", 19.75, "phone_pick_up.png"),
        ("phone", 22.1, "phone_put_down.png"),
    ]
    for name, t, filename in strips:
        scene.save_strip(t, name, STRIPS / filename)
        print(f"  strip {filename}")

    # Zone keyframes.
    scene.render_frame(6.0).resize((640, 360), scene.Image.Resampling.LANCZOS).save(ZONES / "kitchen.png")
    scene.crop_region(6.0, [0, 430, 1280, 720], pad=0).resize((640, 145)).save(ZONES / "countertop.png")
    scene.crop_region(0.8, [46, 150, 470, 306], pad=0).resize((640, 236)).save(ZONES / "shelf.png")
    scene.render_frame(12.0).save(DATA / "preview_t12.png")
    scene.render_frame(8.2).save(DATA / "preview_t8.png")
    print("  zone keyframes + previews")


def encode_video() -> str:
    candidates = [("webm", "VP80"), ("webm", "VP90"), ("mp4", "mp4v")]
    writer = None
    path = None
    for extension, codec in candidates:
        candidate = DATA / f"demo.{extension}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate), fourcc, FPS, (scene.SCREEN_W, scene.SCREEN_H))
        if writer.isOpened():
            path = candidate
            print(f"  encoder {codec} -> {candidate.name}")
            break
        writer.release()
        writer = None
    if writer is None or path is None:
        raise RuntimeError("No usable OpenCV video encoder (tried VP80/VP90/mp4v)")

    total = int(DURATION * FPS) + 1
    for index in range(total):
        t = index / FPS
        frame = scene.render_frame(t)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        if index % 60 == 0:
            print(f"  frame {index}/{total}  t={t:5.1f}s")
    writer.release()
    print(f"  video {path.name} · {path.stat().st_size / 1e6:.1f} MB")
    return f"/media/{path.name}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print("Chronicle demo asset builder")
    print("  rendering crops, strips and keyframes…")
    render_assets()
    print("  encoding egocentric video…")
    video_url = encode_video()
    print("  building event timeline…")
    build_timeline(video_url)

    EVENTS.sort(key=lambda event: (event["t"], event["_order"], event.get("seq", 0)))
    cleaned: list[dict] = []
    for index, event in enumerate(EVENTS):
        event.pop("_order", None)
        event.pop("_intro", None)
        event["seq"] = index
        cleaned.append(event)

    DATA.mkdir(parents=True, exist_ok=True)
    output = DATA / "demo_events.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for event in cleaned:
            handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")

    manifest = {
        "video_url": video_url,
        "duration": DURATION,
        "fps": FPS,
        "events": len(cleaned),
        "entities": len(ENTITY_META),
        "channels": sorted({e["data"]["channel"] for e in cleaned if e["kind"] == "telemetry"}),
        "generated_by": "tools/make_demo_assets.py",
    }
    (DATA / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  events {len(cleaned)} -> {output}")


if __name__ == "__main__":
    main()
