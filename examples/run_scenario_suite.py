#!/usr/bin/env python3
"""Grade ten scenarios of 100 questions against the local decision model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from scenario_suite import SCENARIOS, request_for  # noqa: E402
from jev_like import JevLikeEngine  # noqa: E402

RESULTS = ROOT / "examples" / "scenario_suite_results.json"
SUITE = ROOT / "examples" / "scenario_suite.json"


def predicted(answer: dict) -> tuple[bool, object]:
    kind = answer["type"]
    if kind == "noul":
        return answer["noul"] >= 0.5, answer["noul"]
    if kind == "choice":
        return True, answer["choice"]
    probabilities = answer["probabilities"]
    chosen = max(probabilities, key=probabilities.__getitem__)
    return True, int(chosen)


def grade(answer: dict, expected: object) -> tuple[bool, object]:
    _, value = predicted(answer)
    if answer["type"] == "noul":
        return bool(value >= 0.5) == bool(expected), value
    return value == expected, value


def main() -> None:
    suite_dump = []
    for case in SCENARIOS:
        request, gold = request_for(case)
        suite_dump.append(
            {
                "id": case["id"],
                "name": case["name"],
                "state": case["state"],
                "questions": [
                    {"id": item["id"], "tag": item["tag"], "expected": item["expected"], **request["questions"][item["id"]]}
                    for item in case["items"]
                ],
            }
        )
    SUITE.write_text(json.dumps(suite_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {SUITE} ({len(SCENARIOS)} scenarios)", flush=True)

    engine = JevLikeEngine()
    records = []
    started = time.perf_counter()
    for index, case in enumerate(SCENARIOS, start=1):
        request, gold = request_for(case)
        mark = time.perf_counter()
        response = engine.evaluate(request)
        elapsed = time.perf_counter() - mark
        rows = []
        correct = 0
        by_type = {"noul": [0, 0], "choice": [0, 0], "score": [0, 0]}
        by_tag = {"lookup": [0, 0], "judgment": [0, 0]}
        for item_id, spec in gold.items():
            answer = response["answers"][item_id]
            ok, value = grade(answer, spec["expected"])
            correct += int(ok)
            by_type[spec["type"]][0] += int(ok)
            by_type[spec["type"]][1] += 1
            by_tag[spec["tag"]][0] += int(ok)
            by_tag[spec["tag"]][1] += 1
            rows.append(
                {
                    "id": item_id,
                    "tag": spec["tag"],
                    "type": spec["type"],
                    "instructions": spec["instructions"],
                    "expected": spec["expected"],
                    "predicted": value,
                    "correct": ok,
                    "answer": answer,
                }
            )
        record = {
            "id": case["id"],
            "name": case["name"],
            "elapsed_seconds": round(elapsed, 3),
            "usage": response["usage"],
            "correct": correct,
            "questions": len(rows),
            "accuracy": correct / len(rows),
            "by_type": {key: {"correct": value[0], "total": value[1]} for key, value in by_type.items()},
            "by_tag": {key: {"correct": value[0], "total": value[1]} for key, value in by_tag.items()},
            "rows": rows,
        }
        records.append(record)
        _write(records, time.perf_counter() - started, engine.temperatures)
        print(
            f"{index}/10 {case['name']} {correct}/100 {elapsed:.1f}s "
            f"prefix {response['usage']['prefix_tokens']} suffix {response['usage']['suffix_tokens']}",
            flush=True,
        )
    _write(records, time.perf_counter() - started, engine.temperatures)
    print(f"wrote {RESULTS}", flush=True)


def _write(records: list[dict], elapsed: float, temperature: dict[str, float]) -> None:
    total = sum(record["questions"] for record in records)
    correct = sum(record["correct"] for record in records)
    payload = {
        "summary": {
            "scenarios": len(records),
            "questions": total,
            "correct": correct,
            "accuracy": correct / total if total else 0,
            "elapsed_seconds": round(elapsed, 3),
            "temperature": temperature,
        },
        "scenarios": records,
    }
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
