#!/usr/bin/env python3
"""Update dataset exports to the current explicit purchasing response contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "dataset"
PIPELINE = Path(__file__).with_name("generate_validated_sft_pipeline.py")
TARGET_FILES = (
    "examples_master.jsonl",
    "training.jsonl",
    "evaluation.jsonl",
    "pending_human_review.jsonl",
    "rejected.jsonl",
)


def current_system_message() -> str:
    spec = importlib.util.spec_from_file_location("validated_pipeline_prompt", PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import prompt from {PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEM_MESSAGE


def update_file(path: Path, system_message: str) -> dict:
    rows = []
    changed = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
            raise ValueError(f"Missing leading system message at {path}:{line_number}")
        if messages[0].get("content") != system_message:
            messages[0]["content"] = system_message
            changed += 1
        rows.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return {
        "rows": len(rows),
        "changed": changed,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    prompt = current_system_message()
    report = {}
    for filename in TARGET_FILES:
        path = args.dataset_dir / filename
        if path.exists():
            report[filename] = update_file(path, prompt)
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["system_prompt_contract"] = {
        "explicit_response_structure": True,
        "required_sections": [
            "Purchase recommendation",
            "Result",
            "Rationale",
            "Pros",
            "Risks and limitations",
        ],
        "optional_sections": ["Missing information"],
        "files": report,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps({"system_message": prompt, "files": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
