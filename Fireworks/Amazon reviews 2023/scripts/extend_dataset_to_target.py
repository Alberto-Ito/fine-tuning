#!/usr/bin/env python3
"""Extend the combined SFT dataset in validated, restartable chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path(__file__).with_name("generate_sft_single_pass.py")
DEFAULT_DATASET = ROOT / "data" / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def merge_chunk(dataset: Path, staging: Path, batch_name: str) -> dict:
    master_path = dataset / "examples_master.jsonl"
    existing = read_jsonl(master_path)
    incoming = read_jsonl(staging / "examples_master.jsonl")
    existing_products = {row["metadata"]["parent_asin"] for row in existing}
    incoming_products = {row["metadata"]["parent_asin"] for row in incoming}
    overlap = existing_products & incoming_products
    if overlap:
        raise RuntimeError(f"Product overlap while merging {batch_name}: {sorted(overlap)[:5]}")
    start = len(existing)
    for offset, row in enumerate(incoming, start=1):
        metadata = row["metadata"]
        metadata["source_batch"] = batch_name
        metadata["source_example_id"] = metadata.get("example_id")
        metadata["example_id"] = f"SFT-{start + offset:04d}"
    combined = existing + incoming
    if len({row["metadata"]["parent_asin"] for row in combined}) != len(combined):
        raise RuntimeError("Combined dataset contains duplicate products")
    approved = [
        {"messages": row["messages"]} for row in combined
        if row["metadata"]["review_status"] == "approved"
    ]
    pending = [
        row for row in combined if row["metadata"]["review_status"] == "pending_human_review"
    ]
    rejected = [row for row in combined if row["metadata"]["review_status"] == "rejected"]
    write_jsonl_atomic(master_path, combined)
    write_jsonl_atomic(dataset / "training.jsonl", approved)
    write_jsonl_atomic(dataset / "pending_human_review.jsonl", pending)
    write_jsonl_atomic(dataset / "rejected.jsonl", rejected)
    old_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    history = list(old_manifest.get("generation_batches") or [])
    stage_manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    history.append({"batch": batch_name, **stage_manifest})
    statuses = Counter(row["metadata"]["review_status"] for row in combined)
    results = Counter(row["metadata"]["actual_result"] for row in combined)
    models = Counter(row["metadata"]["models"]["assistant"] for row in combined)
    source_counts = Counter(row["metadata"].get("source_batch", "unknown") for row in combined)
    manifest = {
        "dataset": "combined-industrial-and-scientific-sft",
        "total_examples": len(combined),
        "unique_products": len(combined),
        "source_counts": dict(sorted(source_counts.items())),
        "statuses": dict(sorted(statuses.items())),
        "actual_results": dict(sorted(results.items())),
        "assistant_models": dict(sorted(models.items())),
        "training_examples": statuses.get("approved", 0),
        "files": {
            "master": "examples_master.jsonl", "training": "training.jsonl",
            "pending": "pending_human_review.jsonl", "rejected": "rejected.jsonl",
        },
        "generation_batches": history,
    }
    manifest_tmp = dataset / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, dataset / "manifest.json")
    return manifest


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    master = dataset / "examples_master.jsonl"
    if not master.exists():
        raise FileNotFoundError(master)
    current = len(read_jsonl(master))
    if current > args.target:
        raise ValueError(f"Dataset already has {current} examples, above target {args.target}")
    while current < args.target:
        count = min(args.chunk_size, args.target - current)
        end = current + count
        batch_name = f"gpt6_products_{current + 1:04d}_{end:04d}"
        staging = ROOT / "data" / "generation_staging" / batch_name
        manifest_path = staging / "manifest.json"
        if not manifest_path.exists():
            command = [
                sys.executable, "-B", str(GENERATOR), "--examples", str(count),
                "--product-offset", str(current), "--batch-size", str(args.batch_size),
                "--workers", str(args.workers), "--assistant-model", "gpt-6-sol",
                "--judge-model", "gpt-6-sol", "--output-dir", str(staging),
            ]
            print(f"starting {batch_name}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Generator failed for {batch_name} with code {completed.returncode}")
        stage_rows = read_jsonl(staging / "examples_master.jsonl")
        if len(stage_rows) != count:
            raise RuntimeError(f"{batch_name}: expected {count} staged examples, found {len(stage_rows)}")
        manifest = merge_chunk(dataset, staging, batch_name)
        current = manifest["total_examples"]
        print(
            f"merged={current}/{args.target} approved={manifest['statuses'].get('approved', 0)} "
            f"pending={manifest['statuses'].get('pending_human_review', 0)} "
            f"rejected={manifest['statuses'].get('rejected', 0)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
