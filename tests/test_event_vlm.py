"""VLM verification for event clips (3 frames -> event -> vault)."""

from __future__ import annotations

import json
from pathlib import Path

from chronicle.vfm import event_verify


def _write_video(path: Path, frames: int = 30, fps: int = 10) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = (index * 3 % 255, 80, 160)
        writer.write(frame)
    writer.release()


def _write_replay(path: Path, *, anchor: float = 1.5) -> None:
    rows = [
        {
            "v": 1,
            "kind": "session",
            "t": 0.0,
            "data": {
                "title": "VLM fixture",
                "video_url": "/media/fixture.mp4",
                "duration": 3.0,
                "fps": 10,
                "width": 160,
                "height": 120,
            },
        },
        {
            "v": 1,
            "kind": "entity",
            "t": 0.1,
            "data": {"id": "mug_1", "label": "Mug", "confidence": 0.9},
        },
        {
            "v": 1,
            "kind": "state",
            "t": anchor,
            "data": {
                "entity_id": "mug_1",
                "label": "Mug",
                "from": "INTERACTED",
                "to": "HELD",
                "trigger": {"hand_contact": True},
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _payload() -> dict[str, object]:
    return {
        "event": "hold",
        "caption": "A hand holds the ceramic mug above the counter.",
        "objects": ["mug", "hand", "counter"],
        "actions": ["hold"],
        "zone": "kitchen",
        "confidence": 0.88,
        "agrees": True,
        "correction": "",
    }


def test_strong_graph_action_wins_over_disagreeing_vlm() -> None:
    result = event_verify._parse_verification(
        {
            "event": "insert into phone",
            "caption": "A hand inserts a wallet into a phone.",
            "objects": ["wallet", "phone"],
            "actions": ["insert"],
            "zone": "kitchen",
            "confidence": 0.95,
            "agrees": False,
        },
        "",
        model="test",
        hint={"subject": "keys_1", "state": "HELD", "label": "HELD · Keys"},
    )
    assert result["action"] == "HOLD"
    assert result["vlm_action"] == "INSERT_INTO"
    assert result["disagreement"] is True
    assert result["accepted"] is False


def test_weak_graph_state_can_be_corrected_by_vlm() -> None:
    result = event_verify._parse_verification(
        {
            "event": "hold",
            "caption": "A hand holds a set of keys.",
            "confidence": 0.9,
            "agrees": False,
        },
        "",
        model="test",
        hint={"subject": "keys_1", "state": "VISIBLE_STATIC", "label": "VISIBLE_STATIC · Keys"},
    )
    assert result["action"] == "HOLD"
    assert result["agrees"] is False
    assert result["accepted"] is True


def test_verify_event_writes_frames_snapshot_interaction_and_vault(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)
    replay = tmp_path / "events.jsonl"
    _write_replay(replay)

    def fake_ollama(model, prompt, blobs, *, host, timeout):
        assert len(blobs) == 3
        assert "candidate object: mug_1" in prompt
        return _payload(), json.dumps(_payload())

    monkeypatch.setattr(event_verify, "_ollama_verify", fake_ollama)

    result = event_verify.verify_event(
        video,
        anchor_rel=1.5,
        duration=3.0,
        hint={
            "label": "HELD · Mug",
            "subject": "mug_1",
            "entity_id": "mug_1",
            "state": "HELD",
            "zone": "kitchen",
        },
        frames_dir=tmp_path / "frames",
        replay_path=replay,
        out_dir=tmp_path,
        title="HELD · Mug",
    )

    assert result["status"] == "ok"
    assert result["action"] == "HOLD"
    assert result["agrees"] is True
    assert result["snapshot"] == "frames/snapshot.jpg"
    for name in ("pre.jpg", "contact.jpg", "post.jpg", "snapshot.jpg"):
        assert (tmp_path / "frames" / name).is_file()
    assert (tmp_path / "vault.db").is_file()
    assert (tmp_path / "ledger.json").is_file()

    rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    interactions = [
        row
        for row in rows
        if row.get("kind") == "interaction"
        and (row.get("data") or {}).get("source") == "vlm-event"
    ]
    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction["data"]["action"] == "HOLD"
    assert interaction["data"]["caption"].startswith("A hand holds")
    assert interaction["data"]["frames"]["contact"] == "frames/contact.jpg"
    assert rows[0]["kind"] == "session"


def test_verify_event_without_vlm_still_writes_rule_fallback_and_vault(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video)
    replay = tmp_path / "events.jsonl"
    _write_replay(replay)

    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(event_verify, "_ollama_verify", fail)
    monkeypatch.setattr(event_verify, "_florence_verify", fail)

    result = event_verify.verify_event(
        video,
        anchor_rel=1.5,
        duration=3.0,
        hint={
            "label": "HELD · Mug",
            "subject": "mug_1",
            "entity_id": "mug_1",
            "state": "HELD",
            "zone": "kitchen",
        },
        frames_dir=tmp_path / "frames",
        replay_path=replay,
        out_dir=tmp_path,
        title="HELD · Mug",
    )

    assert result["status"] == "unavailable"
    assert result["provider"] == "rule-fallback"
    assert result["accepted"] is False
    assert (tmp_path / "vault.db").is_file()
    rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    assert any(
        (row.get("data") or {}).get("source") == "vlm-event"
        for row in rows
        if row.get("kind") == "interaction"
    )


def test_build_event_clips_can_verify_and_expose_snapshot(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    video = data / "fixture.mp4"
    _write_video(video, frames=60)
    _write_replay(data / "fixture.events.jsonl")
    # Point the session at the real fixture video so the cutter runs.
    rows = (data / "fixture.events.jsonl").read_text(encoding="utf-8").splitlines()
    session = json.loads(rows[0])
    session["data"]["video_url"] = "/media/fixture.mp4"
    session["data"]["duration"] = 6.0
    (data / "fixture.events.jsonl").write_text(
        "\n".join([json.dumps(session)] + rows[1:]) + "\n", encoding="utf-8"
    )

    def fake_verify(video, **kwargs):
        assert Path(video).is_file()
        assert kwargs["frames_dir"].name == "frames"
        snapshot = kwargs["frames_dir"] / "snapshot.jpg"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"jpeg")
        return {
            "status": "ok",
            "provider": "test",
            "action": "HOLD",
            "caption": "A hand holds the mug.",
            "snapshot": "frames/snapshot.jpg",
            "vault": "vault.db",
        }

    monkeypatch.setattr("chronicle.vfm.event_verify.verify_event", fake_verify)

    from dashboard.event_clips import build_event_clips

    index = build_event_clips("events:fixture.events.jsonl", data, verify=True)
    assert index["verify"] is True
    assert index["verified"] >= 1
    entry = index["clips"][0]
    assert entry["vlm"]["status"] == "ok"
    assert entry["snapshot"] == "frames/snapshot.jpg"
    assert entry["action"] == "HOLD"
    assert json.loads((data / entry["graph"]).read_text(encoding="utf-8"))["vlm"]
