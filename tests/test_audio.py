from __future__ import annotations

from chronicle.audio import TranscriptIndex, TranscriptSegment
from chronicle.integration import ChronicleRuntime


def test_transcript_index_prefers_containing_segment() -> None:
    index = TranscriptIndex(
        [
            TranscriptSegment(0.0, 5.0, "first"),
            TranscriptSegment(4.0, 8.0, "second"),
        ]
    )

    assert index.at(4.5) == TranscriptSegment(4.0, 8.0, "second")
    assert index.at(20.0, tolerance=1.0) is None


def test_runtime_answers_explicit_speech_questions(tmp_path) -> None:
    index = TranscriptIndex(
        [
            TranscriptSegment(40.0, 50.0, "Very very top. Now change the controller."),
            TranscriptSegment(80.0, 90.0, "No, don't flip it."),
        ]
    )

    with ChronicleRuntime(tmp_path / "memory.db", transcript=index) as runtime:
        said = runtime.ask("what did the instructor say at 42 seconds?")
        corrected = runtime.ask("how was the person corrected at 85 seconds?")
        visual = runtime.ask("what did the person do at 67 seconds?")

    assert said.method == "transcript"
    assert "Very very top" in said.answer
    assert corrected.method == "transcript"
    assert "verbally" in corrected.answer
    assert "don't flip it" in corrected.answer
    assert visual.method != "transcript"
