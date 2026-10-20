#!/usr/bin/env python3
"""Apply human-review decisions and atomically rebuild dataset exports.

The decisions file is JSONL with one object per example, for example:

{"example_id":"SFT-0001","decision":"approve","reviewer":"alice"}
{"example_id":"SFT-0002","decision":"approve","corrected_answer":"Purchase recommendation:\n- Result: DO_NOT_BUY\n- Rationale: ...\n\nPros:\n- ...\n\nRisks and limitations:\n- ..."}
{"example_id":"SFT-0003","decision":"reject","notes":"Underlying evidence is unrelated to the product."}

By default the command validates and previews changes. Pass --apply to write them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "dataset"
ALLOWED_RESULTS = {"BUY", "CONDITIONAL_BUY", "DO_NOT_BUY", "INSUFFICIENT_EVIDENCE"}
DEFAULT_REVIEWER = "Andres Vasile"
DECISION_TO_STATUS = {
    "approve": "approved",
    "approved": "approved",
    "reject": "rejected",
    "rejected": "rejected",
    "pending": "pending_human_review",
    "pending_human_review": "pending_human_review",
}
RESULT_PATTERN = re.compile(r"(?m)^- Result: ([A-Z_]+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True, help="JSONL decisions file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--apply", action="store_true",
        help="write changes; without this flag the command is a dry run",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            rows.append(row)
    return rows


def extract_result(answer: str, example_id: str) -> str:
    matches = RESULT_PATTERN.findall(answer)
    if len(matches) != 1:
        raise ValueError(f"{example_id}: assistant answer must contain exactly one '- Result: ...' line")
    result = matches[0]
    if result not in ALLOWED_RESULTS:
        raise ValueError(f"{example_id}: invalid purchase result {result!r}")
    return result


def validate_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    allowed_fields = {"example_id", "decision", "corrected_answer", "reviewer", "notes"}
    for row in rows:
        unknown = set(row) - allowed_fields
        if unknown:
            raise ValueError(f"Unknown decision fields: {sorted(unknown)}")
        example_id = row.get("example_id")
        decision = row.get("decision")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("Every decision requires a non-empty string example_id")
        if example_id in decisions:
            raise ValueError(f"Duplicate decision for {example_id}")
        if decision not in DECISION_TO_STATUS:
            raise ValueError(f"{example_id}: decision must be approve, reject, or pending")
        for optional in ("corrected_answer", "reviewer", "notes"):
            if optional in row and (not isinstance(row[optional], str) or not row[optional].strip()):
                raise ValueError(f"{example_id}: {optional} must be a non-empty string")
        decisions[example_id] = row
    if not decisions:
        raise ValueError("The decisions file is empty")
    return decisions


def validate_master(master: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    products: set[str] = set()
    for row in master:
        metadata = row.get("metadata", {})
        example_id = metadata.get("example_id")
        parent_asin = metadata.get("parent_asin")
        if not isinstance(example_id, str) or example_id in by_id:
            raise ValueError(f"Missing or duplicate example_id: {example_id!r}")
        if not isinstance(parent_asin, str) or parent_asin in products:
            raise ValueError(f"Missing or duplicate parent_asin: {parent_asin!r}")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{example_id}: final message must have role assistant")
        by_id[example_id] = row
        products.add(parent_asin)
    return by_id


def apply_decisions(
    master: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    by_id = validate_master(master)
    missing = set(decisions) - set(by_id)
    if missing:
        raise ValueError(f"Unknown example IDs: {sorted(missing)}")

    changes: list[dict[str, str]] = []
    for example_id, decision in decisions.items():
        example = by_id[example_id]
        metadata = example["metadata"]
        previous = metadata.get("review_status")
        if previous != "pending_human_review":
            raise ValueError(f"{example_id}: expected pending_human_review, found {previous!r}")

        corrected = decision.get("corrected_answer")
        if corrected is not None:
            example["messages"][-1]["content"] = corrected.strip()
        answer = example["messages"][-1].get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{example_id}: assistant answer is empty")
        actual_result = extract_result(answer, example_id)
        status = DECISION_TO_STATUS[decision["decision"]]
        metadata["review_status"] = status
        metadata["actual_result"] = actual_result
        audit = {
            "decision": status,
            "reviewer": decision.get("reviewer", DEFAULT_REVIEWER).strip(),
        }
        if "notes" in decision:
            audit["notes"] = decision["notes"].strip()
        audit["answer_corrected"] = corrected is not None
        metadata["human_review"] = audit
        changes.append({"example_id": example_id, "from": previous, "to": status})
    return changes


def build_manifest(manifest: dict[str, Any], master: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(row["metadata"]["review_status"] for row in master)
    results = Counter(row["metadata"]["actual_result"] for row in master)
    manifest["total_examples"] = len(master)
    manifest["unique_products"] = len({row["metadata"]["parent_asin"] for row in master})
    manifest["statuses"] = dict(sorted(statuses.items()))
    manifest["actual_results"] = dict(sorted(results.items()))
    manifest["training_examples"] = statuses.get("approved", 0)
    return manifest


def serialize_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_outputs(data_dir: Path, master: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    approved = [
        {"messages": row["messages"]}
        for row in master if row["metadata"]["review_status"] == "approved"
    ]
    pending = [row for row in master if row["metadata"]["review_status"] == "pending_human_review"]
    rejected = [row for row in master if row["metadata"]["review_status"] == "rejected"]
    contents = {
        "examples_master.jsonl": serialize_jsonl(master),
        "training.jsonl": serialize_jsonl(approved),
        "pending_human_review.jsonl": serialize_jsonl(pending),
        "rejected.jsonl": serialize_jsonl(rejected),
        "manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    }
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for name, content in contents.items():
            destination = data_dir / name
            temporary = data_dir / f".{name}.human-review.tmp"
            temporary.write_text(content, encoding="utf-8")
            temporary_paths.append((temporary, destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    master_path = data_dir / "examples_master.jsonl"
    manifest_path = data_dir / "manifest.json"
    decisions = validate_decisions(read_jsonl(args.decisions.resolve()))
    master = read_jsonl(master_path)
    changes = apply_decisions(master, decisions)
    manifest = build_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), master)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "changes": changes,
        "statuses_after": manifest["statuses"],
        "total_examples": manifest["total_examples"],
    }
    if args.apply:
        write_outputs(data_dir, master, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
