from __future__ import annotations

import pytest

from scripts.egotempo_manifest import build_manifest, parse_clip_id
from scripts.evaluate_egotempo import token_score
from scripts.run_egotempo import _clip_qa_payload


def test_parse_clip_id_handles_uuid_and_negative_start() -> None:
    video, start, end = parse_clip_id(
        "141b24d2-2fc1-4f9d-89ef-8dc3055213f7_-0.048_56.708"
    )
    assert video == "141b24d2-2fc1-4f9d-89ef-8dc3055213f7"
    assert start == pytest.approx(-0.048)
    assert end == pytest.approx(56.708)


def test_parse_clip_id_rejects_malformed_value() -> None:
    with pytest.raises(ValueError):
        parse_clip_id("not-a-clip")


def test_build_manifest_groups_questions_by_clip_and_video() -> None:
    payload = {
        "annotations": [
            {
                "question_id": "v1_1_2_0",
                "clip_id": "v1_1_2",
                "question_type": "counting objects",
                "question": "How many cups?",
                "answer": "2",
            },
            {
                "question_id": "v1_1_2_1",
                "clip_id": "v1_1_2",
                "question_type": "locating object",
                "question": "Where is the cup?",
                "answer": "On the table.",
            },
            {
                "question_id": "v2_5_7_0",
                "clip_id": "v2_5_7",
                "question_type": "counting objects",
                "question": "How many bowls?",
                "answer": "1",
            },
        ]
    }

    manifest = build_manifest(payload)

    assert manifest["questions"] == 3
    assert manifest["clip_count"] == 2
    assert manifest["videos"] == 2
    assert manifest["clips_by_video"] == {"v1": ["v1_1_2"], "v2": ["v2_5_7"]}
    assert manifest["category_counts"] == {
        "counting objects": 2,
        "locating object": 1,
    }
    assert len(manifest["clips"][0]["questions"]) == 2


def test_token_score_exact_numeric_and_empty() -> None:
    assert token_score("q", "On the table.", "The cup is on the table")[0] == 5
    assert token_score("q", "3", "There are 3 cups")[0] >= 3
    assert token_score("q", "cyan", "")[0] == 0
    assert token_score("q", "a red controller", "a blue controller")[0] < 3


def test_clip_qa_payload_preserves_question_ids() -> None:
    clip = {
        "clip_id": "v1_1_2",
        "questions": [
            {
                "question_id": "v1_1_2_0",
                "question": "How many cups?",
                "answer": "2",
                "question_type": "counting objects",
            }
        ],
    }
    payload = _clip_qa_payload(clip)
    assert payload["video_id"] == "v1_1_2"
    assert payload["questions"][0]["question_id"] == "v1_1_2_0"
    assert payload["questions"][0]["category"] == "counting objects"
