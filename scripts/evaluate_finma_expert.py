#!/usr/bin/env python3
"""Evaluate the FinMA expert tool on a small JSONL sample set."""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "backend" / "packages" / "harness"
if str(HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HARNESS_PATH))

from deerflow.community.finma.tools import financial_analysis_tool  # noqa: E402


def _load_samples(path: Path) -> list[dict]:
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return samples


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _score_result(result: dict, expected: dict) -> dict:
    expected_label = _normalize(expected.get("label"))
    expected_impact = _normalize(expected.get("impact_direction"))
    actual_label = _normalize(result.get("label"))
    actual_impact = _normalize(result.get("impact_direction"))

    label_match = bool(expected_label and actual_label and expected_label in actual_label)
    impact_match = bool(expected_impact and actual_impact == expected_impact)
    return {
        "label_match": label_match,
        "impact_match": impact_match,
        "passed": impact_match and (label_match or not actual_label),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        default=str(REPO_ROOT / "evals" / "finma_expert_samples.jsonl"),
        help="Path to JSONL samples.",
    )
    parser.add_argument("--real", action="store_true", help="Call the configured FinMA API instead of mock mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit.")
    args = parser.parse_args()

    samples = _load_samples(Path(args.samples))
    if args.limit:
        samples = samples[: args.limit]

    rows = []
    passed = 0
    failures = 0
    for sample in samples:
        raw = financial_analysis_tool.invoke(
            {
                "task": sample["task"],
                "ticker": sample.get("ticker", ""),
                "text": sample["text"],
                "output_schema": "label,impact_direction,affected_factors,confidence,rationale",
                "use_mock": not args.real,
            }
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw_output": raw}
            failures += 1

        score = _score_result(result, sample["expected"])
        passed += int(score["passed"])
        rows.append(
            {
                "id": sample["id"],
                "task": sample["task"],
                "expected": sample["expected"],
                "actual": {
                    "provider": result.get("provider"),
                    "label": result.get("label"),
                    "impact_direction": result.get("impact_direction"),
                    "confidence": result.get("confidence"),
                },
                "score": score,
            }
        )

    total = len(samples)
    summary = {
        "mode": "real" if args.real else "mock",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "json_failures": failures,
        "accuracy": round(passed / total, 4) if total else 0,
        "results": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
