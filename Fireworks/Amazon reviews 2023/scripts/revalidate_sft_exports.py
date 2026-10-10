#!/usr/bin/env python3
"""Revalidate master SFT records and atomically rebuild routed JSONL exports."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path


def is_english_source(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    english = words & {
        "the", "and", "is", "it", "to", "of", "for", "with", "this",
        "that", "was", "are", "my", "but", "not", "have", "has", "in",
        "on", "as", "from", "works", "worked", "product", "use", "used",
    }
    non_english = words & {
        "el", "la", "los", "las", "que", "para", "una", "pero", "muy",
        "con", "por", "como", "funciona", "producto", "compre", "compré",
        "est", "les", "des", "une", "pas", "pour", "avec", "und", "der",
        "die", "das", "ist", "nicht", "mit", "ein", "eine",
    }
    return len(english) >= 2 and len(english) > len(non_english)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    master = args.data_dir / "examples_master.jsonl"
    database = args.data_dir / "staging.sqlite"
    outputs = {
        "master": (args.data_dir / "examples_master.jsonl.tmp"),
        "training": (args.data_dir / "training.jsonl.tmp"),
        "pending": (args.data_dir / "pending_human_review.jsonl.tmp"),
    }
    counts = {
        "master": 0, "approved": 0, "pending": 0,
        "language_rerouted": 0, "source_language_pending": 0,
    }
    connection = sqlite3.connect(database)
    cache: dict[str, bool] = {}
    with (
        master.open(encoding="utf-8") as source,
        outputs["master"].open("w", encoding="utf-8") as master_out,
        outputs["training"].open("w", encoding="utf-8") as training_out,
        outputs["pending"].open("w", encoding="utf-8") as pending_out,
    ):
        for line in source:
            example = json.loads(line)
            metadata = example["metadata"]
            parent = metadata["parent_asin"]
            if parent not in cache:
                rows = connection.execute(
                    "SELECT review_json FROM reviews WHERE parent_asin=?", (parent,)
                ).fetchall()
                selected = set(metadata["source_review_ids"])
                relevant = [json.loads(row[0]) for row in rows]
                relevant = [row for row in relevant if row["source"]["review_id"] in selected]
                cache[parent] = bool(relevant) and all(
                    is_english_source(row["review"].get("text") or "") for row in relevant
                )
            if not cache[parent] and metadata["review_status"] == "approved":
                metadata["review_status"] = "pending_human_review"
                metadata.setdefault("review_reasons", []).append("source_language_check_failed")
                counts["language_rerouted"] += 1
            if "source_language_check_failed" in metadata.get("review_reasons", []):
                counts["source_language_pending"] += 1
            serialized = json.dumps(example, ensure_ascii=False) + "\n"
            master_out.write(serialized)
            counts["master"] += 1
            if metadata["review_status"] == "approved":
                training_out.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + "\n")
                counts["approved"] += 1
            else:
                pending_out.write(serialized)
                counts["pending"] += 1
    connection.close()
    os.replace(outputs["master"], master)
    os.replace(outputs["training"], args.data_dir / "training.jsonl")
    os.replace(outputs["pending"], args.data_dir / "pending_human_review.jsonl")
    old = args.data_dir / "train_fireworks.jsonl"
    if old.exists():
        old.unlink()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts["language_rerouted_total"] = counts["source_language_pending"]
    manifest["final_counts"] = counts
    manifest["files"]["fireworks"] = str(args.data_dir / "training.jsonl")
    manifest["quality_gate"] = (
        "Approved examples passed schema, citation, domain-fit, assistant-language, "
        "and source-language checks. Pending examples require human review."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
