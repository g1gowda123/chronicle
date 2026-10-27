"""Screenshot-at-answer-time and fullscreen affordances (zine dashboard)."""

from __future__ import annotations

from pathlib import Path

from dashboard.ui import theme

ROOT = Path(__file__).resolve().parents[1]


def test_video_toolbar_exposes_screenshot_and_fullscreen() -> None:
    html = theme.video_html({"video_url": "/media/demo.mp4", "title": "Demo"})
    assert "Chronicle.screenshot()" in html
    assert "Chronicle.fullscreen()" in html
    assert 'id="chron-shot"' in html
    assert 'id="chron-full"' in html


def test_client_has_answer_time_screenshot_and_lightbox() -> None:
    js = (ROOT / "dashboard" / "static" / "chronicle.js").read_text(encoding="utf-8")
    assert "screenshotAt(t)" in js
    assert "captureFrame(t)" in js
    assert "showShot(" in js
    assert "frame @ " in js  # button added to grounded answers
    assert "chron-shot-modal" in js
    css = (ROOT / "dashboard" / "static" / "chronicle.css").read_text(encoding="utf-8")
    assert ".chron-shot-modal" in css
    assert ".video-stage:fullscreen" in css
    assert "chronicle.js?v=" in theme.HEAD_HTML
    assert "chronicle.css?v=" in theme.HEAD_HTML
