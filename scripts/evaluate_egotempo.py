"""Score EgoTempo predictions.

EgoTempo is an open-ended QA benchmark.  The official protocol uses an LLM
judge with a 0-5 correctness score; a score >= 3 is treated as correct here.
A deterministic token judge is also provided for quick diagnostics, but it is
not a substitute for the official judge.

Predictions can be either a list of ``{"question_id": ..., "prediction": ...}``
objects or a mapping from question_id to prediction.  The output contains the
per-question score, the aggregate accuracy and a breakdown by question type.

Example
-------
    python scripts/evaluate_egotempo.py \
        --predictions runs/egotempo/predictions.json \
        --judge ollama --model qwen2.5:7b-instruct \
        --out runs/egotempo/eval.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

JUDGE_PROMPT = """You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not.

## INSTRUCTIONS
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as "it is impossible to answer the question from the video", as incorrect, unless the ground truth answer also says that.

Question: {question}
Ground truth correct Answer: {answer}
Predicted Answer: {prediction}

Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Return ONLY a Python dictionary string with keys 'pred', 'score' and 'reason'. 'pred' is 'correct' or 'incorrect', 'score' is an INTEGER, and 'reason' is a short explanation.
"""


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_score(question: str, gold: str, prediction: str) -> tuple[int, str]:
    gold_n = _normalise(gold)
    pred_n = _normalise(prediction)
    if not pred_n:
        return 0, "empty prediction"
    if gold_n and gold_n in pred_n:
        return 5, "gold string contained in prediction"
    gold_tokens = set(gold_n.split())
    pred_tokens = set(pred_n.split())
    if gold_tokens:
        overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
        if overlap >= 0.75:
            return 4, f"token overlap {overlap:.2f}"
        if overlap >= 0.6:
            return 3, f"token overlap {overlap:.2f}"
    # Numeric answers: a number in the prediction is the signal.
    gold_numbers = re.findall(r"\d+", gold)
    if gold_numbers and any(number in pred_n for number in gold_numbers):
        return 3, "gold number present"
    return 0, "no match"


def _parse_judge_output(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def ollama_judge(
    row: dict[str, Any],
    *,
    host: str,
    model: str,
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        answer=row["answer"],
        prediction=row["prediction"],
    )
    error = ""
    for _ in range(retries + 1):
        try:
            response = requests.post(
                f"{host.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            parsed = _parse_judge_output(str(response.json().get("response", "")))
            if parsed is not None:
                score = int(parsed.get("score", 0))
                return {
                    "score": max(0, min(5, score)),
                    "pred": str(parsed.get("pred", "")),
                    "reason": str(parsed.get("reason", ""))[:500],
                    "judge": "ollama",
                }
            error = "judge output was not parseable"
        except Exception as exc:  # pragma: no cover - depends on local Ollama
            error = repr(exc)
    score, reason = token_score(row["question"], row["answer"], row["prediction"])
    return {"score": score, "pred": "", "reason": f"fallback: {error}; {reason}", "judge": "token-fallback"}


def load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        output: dict[str, str] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            question_id = row.get("question_id")
            prediction = row.get("prediction", row.get("answer", row.get("predicted")))
            if question_id is None or prediction is None:
                continue
            output[str(question_id)] = str(prediction)
        return output
    raise ValueError("predictions must be a list or a mapping")


def evaluate(
    annotations: Sequence[dict[str, Any]],
    predictions: dict[str, str],
    *,
    judge: str,
    host: str,
    model: str,
    timeout: float,
    workers: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for item in annotations:
        question_id = str(item["question_id"])
        prediction = predictions.get(question_id, "")
        if not prediction:
            missing += 1
        row = {
            "question_id": question_id,
            "clip_id": str(item["clip_id"]),
            "question_type": str(item.get("question_type", "unknown")),
            "question": str(item["question"]),
            "answer": str(item["answer"]),
            "prediction": prediction,
        }
        rows.append(row)

    if judge == "ollama":
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            scored = list(
                pool.map(
                    lambda row: ollama_judge(
                        row, host=host, model=model, timeout=timeout
                    ),
                    rows,
                )
            )
    else:
        scored = []
        for row in rows:
            score, reason = token_score(row["question"], row["answer"], row["prediction"])
            scored.append({"score": score, "pred": "", "reason": reason, "judge": "token"})

    for row, result in zip(rows, scored, strict=True):
        row.update(result)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    categories = {
        name: {
            "correct": sum(1 for row in items if row["score"] >= 3),
            "total": len(items),
            "accuracy": (
                sum(1 for row in items if row["score"] >= 3) / len(items) if items else 0.0
            ),
            "mean_score": (
                sum(row["score"] for row in items) / len(items) if items else 0.0
            ),
        }
        for name, items in sorted(by_type.items())
    }
    correct = sum(1 for row in rows if row["score"] >= 3)
    return {
        "benchmark": "EgoTempo",
        "judge": judge if judge == "token" else f"ollama:{model}",
        "questions": len(rows),
        "missing_predictions": missing,
        "correct@score>=3": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "mean_score": sum(row["score"] for row in rows) / len(rows) if rows else 0.0,
        "categories": categories,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/benchmarks/egotempo/egotempo_openQA.json"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", choices=("token", "ollama"), default="token")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    annotations = payload.get("annotations") or []
    report = evaluate(
        annotations,
        load_predictions(args.predictions),
        judge=args.judge,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"EgoTempo {report['correct@score>=3']}/{report['questions']} "
        f"({100.0 * report['accuracy']:.1f}%), mean score {report['mean_score']:.2f} "
        f"[{report['judge']}] -> {args.out}"
    )
    for name, info in report["categories"].items():
        print(
            f"  {name:<28} {info['correct']}/{info['total']} "
            f"({100.0 * info['accuracy']:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
