#!/usr/bin/env python3
"""Approve selected pending examples and rebuild routed dataset exports."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    master_path = data_dir / "examples_master.jsonl"
    master = read_jsonl(master_path)
    requested = set(args.ids)
    found: set[str] = set()

    for example in master:
        metadata = example["metadata"]
        example_id = metadata["example_id"]
        if example_id not in requested:
            continue
        found.add(example_id)
        if metadata["review_status"] != "pending_human_review":
            raise ValueError(f"{example_id} is not pending human review")
        judge = metadata.get("judge", {})
        if not (
            judge.get("verdict") == "APPROVE"
            and judge.get("decision_correct") is True
            and judge.get("grounded") is True
            and judge.get("sft_ready") is True
            and not judge.get("issues")
            and not judge.get("recommended_changes")
        ):
            raise ValueError(f"{example_id} does not satisfy the conservative approval gate")
        metadata["review_status"] = "approved"

    missing = requested - found
    if missing:
        raise ValueError(f"Unknown example IDs: {sorted(missing)}")

    approved = [row for row in master if row["metadata"]["review_status"] == "approved"]
    pending = [row for row in master if row["metadata"]["review_status"] == "pending_human_review"]
    rejected = [row for row in master if row["metadata"]["review_status"] == "rejected"]

    write_jsonl_atomic(master_path, master)
    write_jsonl_atomic(data_dir / "training.jsonl", [{"messages": row["messages"]} for row in approved])
    write_jsonl_atomic(data_dir / "pending_human_review.jsonl", pending)
    write_jsonl_atomic(data_dir / "rejected.jsonl", rejected)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statuses = Counter(row["metadata"]["review_status"] for row in master)
    manifest["statuses"] = dict(sorted(statuses.items()))
    manifest["training_examples"] = statuses.get("approved", 0)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)

    print(json.dumps({
        "approved_ids": sorted(requested),
        "approved_total": len(approved),
        "pending_total": len(pending),
        "rejected_total": len(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
