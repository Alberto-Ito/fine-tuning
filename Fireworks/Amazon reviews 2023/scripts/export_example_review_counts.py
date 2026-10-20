#!/usr/bin/env python3
"""Export total and selected review counts from SFT JSONL prompts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CONTEXT_PATTERN = re.compile(
    r"Original Amazon review context:\n(?P<context>\{.*\})\n\nAssess whether",
    re.DOTALL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-example-number", type=int)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for raw in args.input.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        example = json.loads(raw)
        example_id = example["metadata"]["example_id"]
        if args.max_example_number is not None and int(example_id.rsplit("-", 1)[1]) > args.max_example_number:
            continue
        user_content = next(
            message["content"] for message in example["messages"] if message["role"] == "user"
        )
        match = CONTEXT_PATTERN.search(user_content)
        if not match:
            raise ValueError(f"Review context not found for {example['metadata']['example_id']}")
        context = json.loads(match.group("context"))
        statistics = context["complete_variant_review_statistics"]
        rows.append({
            "example_id": example_id,
            "variant_asin": example["metadata"]["variant_asin"],
            "total_reviews": statistics["reviews"],
            "verified_purchase_reviews": statistics["verified_purchase_reviews"],
            "selected_reviews": len(context["selected_reviews"]),
        })

    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"examples": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
