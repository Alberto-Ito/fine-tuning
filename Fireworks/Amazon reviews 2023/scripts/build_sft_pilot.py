#!/usr/bin/env python3
"""Build four review-grounded SFT pilot examples for one product.

This pilot deliberately separates evidence selection (code) from teacher
answers (JSON). At scale, the teacher-answer file can be replaced by calls to
a teacher model followed by automatic and human quality checks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


TASKS = (
    "pros_and_cons",
    "use_cases",
    "common_problems",
    "purchase_recommendation",
)

TASK_INSTRUCTIONS = {
    "pros_and_cons": (
        "Summarize the observed advantages and limitations. Separate catalog "
        "facts from user experiences, and do not invent capabilities."
    ),
    "use_cases": (
        "Identify suitable use cases and scenarios where this product would "
        "not be a good choice. Ground every conclusion in the evidence."
    ),
    "common_problems": (
        "Identify recurring problems, their potential impact, and practical "
        "checks a buyer should perform."
    ),
    "purchase_recommendation": (
        "Provide a conditional purchase recommendation: who should consider "
        "it, who should not, and what to verify before buying."
    ),
}

KEYWORDS = {
    "pros_and_cons": (
        "homekit", "easy", "value", "display", "disconnect", "battery", "accur"
    ),
    "use_cases": (
        "homekit", "automation", "home assistant", "studio", "track", "history"
    ),
    "common_problems": (
        "disconnect", "offline", "battery", "reset", "freeze", "inaccur", "update"
    ),
    "purchase_recommendation": (
        "recommend", "hub", "firmware", "disconnect", "reliable", "accur"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parent-asin", required=True)
    parser.add_argument("--teacher-answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-evidence", type=int, default=8)
    return parser.parse_args()


def load_product_rows(path: Path, parent_asin: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["source"]["parent_asin"] == parent_asin:
                rows.append(row)
    if not rows:
        raise ValueError(f"parent_asin not found: {parent_asin}")
    return rows


def informative(row: dict) -> bool:
    text = row["review"].get("text") or ""
    return len(text.split()) >= 8 and looks_english(text)


def looks_english(text: str) -> bool:
    """Reject clearly Spanish reviews from this English-only pilot."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    english = words & {
        "the", "this", "and", "is", "it", "to", "of", "with", "for",
        "my", "that", "have", "not", "works", "device", "sensor", "but",
    }
    spanish = words & {
        "el", "la", "los", "las", "que", "para", "con", "una", "he",
        "este", "esta", "muy", "pero", "funciona", "sensor", "sensores",
    }
    return len(english) >= len(spanish)


def evidence_score(row: dict, task: str) -> tuple:
    review = row["review"]
    text = f'{review.get("title") or ""} {review.get("text") or ""}'.lower()
    matches = sum(keyword in text for keyword in KEYWORDS[task])
    helpful = int(review.get("helpful_votes") or 0)
    rating = float(review.get("rating") or 0)
    if task == "common_problems":
        polarity = 2 if rating <= 3 else 0
    elif task == "use_cases":
        polarity = 2 if rating >= 4 else 0
    else:
        polarity = 1
    return matches, polarity, helpful, len(text)


def select_evidence(rows: list[dict], task: str, limit: int) -> list[dict]:
    candidates = [row for row in rows if informative(row)]
    ranked = sorted(candidates, key=lambda row: evidence_score(row, task), reverse=True)

    if task in {"pros_and_cons", "purchase_recommendation"}:
        positive = [row for row in ranked if row["review"]["rating"] >= 4][: limit // 2]
        negative = [row for row in ranked if row["review"]["rating"] <= 3][
            : limit - len(positive)
        ]
        return positive + negative
    return ranked[:limit]


def compact_product(product: dict) -> dict:
    return {
        "name": product.get("name"),
        "brand_or_store": product.get("brand_or_store"),
        "category": product.get("category"),
        "subcategories": product.get("subcategories") or [],
        "features": product.get("features") or [],
        "technical_details": product.get("technical_details") or {},
        "price_usd": product.get("price_usd"),
        "catalog_average_rating": product.get("catalog_average_rating"),
        "catalog_rating_count": product.get("catalog_rating_count"),
    }


def evidence_text(rows: list[dict]) -> str:
    blocks = []
    for index, row in enumerate(rows, start=1):
        review = row["review"]
        text = " ".join((review.get("text") or "").split())
        if len(text) > 700:
            text = text[:697].rstrip() + "..."
        blocks.append(
            f'R{index} | {review.get("rating")}/5 | '
            f'verified={review.get("verified_purchase")} | '
            f'helpful_votes={review.get("helpful_votes", 0)}\n'
            f'Title: {review.get("title") or "(no title)"}\nText: {text}'
        )
    return "\n\n".join(blocks)


def build_user_message(
    task: str, product: dict, stats: dict, evidence: list[dict]
) -> str:
    return (
        "Product data:\n"
        + json.dumps(compact_product(product), ensure_ascii=False, indent=2)
        + "\n\nReview subset summary:\n"
        + json.dumps(stats, ensure_ascii=False, indent=2)
        + "\n\nSelected evidence:\n"
        + evidence_text(evidence)
        + "\n\nTask:\n"
        + TASK_INSTRUCTIONS[task]
    )


def main() -> None:
    args = parse_args()
    rows = load_product_rows(args.input, args.parent_asin)
    product = rows[0]["product"]
    ratings = Counter(int(row["review"]["rating"]) for row in rows)
    stats = {
        "reviews_available": len(rows),
        "informative_reviews": sum(informative(row) for row in rows),
        "verified_reviews": sum(
            bool(row["review"].get("verified_purchase")) for row in rows
        ),
        "ratings": {str(key): ratings.get(key, 0) for key in range(1, 6)},
    }

    answers = json.loads(args.teacher_answers.read_text(encoding="utf-8"))
    missing = [task for task in TASKS if not answers.get(task)]
    if missing:
        raise ValueError(f"Missing teacher answers for: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for task in TASKS:
        evidence = select_evidence(rows, task, args.max_evidence)
        example_id = hashlib.sha256(
            f"{args.parent_asin}:{task}:pilot-v1".encode()
        ).hexdigest()
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a technical purchasing copilot. Analyze only "
                            "the provided product data and reviews. Distinguish "
                            "facts, user experiences, and uncertainty; do not "
                            "invent information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_user_message(task, product, stats, evidence),
                    },
                    {"role": "assistant", "content": answers[task]},
                ],
                "metadata": {
                    "example_id": example_id,
                    "task": task,
                    "parent_asin": args.parent_asin,
                    "source_review_ids": [
                        row["source"]["review_id"] for row in evidence
                    ],
                    "source_review_count": len(evidence),
                    "split": "train_pilot",
                    "review_status": "pending_human_review",
                    "pipeline_version": "pilot-v1",
                },
            }
        )

    with args.output.open("w", encoding="utf-8") as destination:
        for example in examples:
            destination.write(json.dumps(example, ensure_ascii=False) + "\n")

    manifest_path = args.manifest or args.output.with_name("manifest.json")
    manifest = {
        "pipeline_version": "pilot-v1",
        "input": str(args.input),
        "output": str(args.output),
        "parent_asin": args.parent_asin,
        "product_name": product.get("name"),
        "source_statistics": stats,
        "examples": len(examples),
        "tasks": list(TASKS),
        "review_status": "pending_human_review",
        "important": (
            "Pilot only. Do not include in production training until answers "
            "and source evidence pass human review."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"product={product.get('name')}")
    print(f"source_reviews={len(rows)}")
    print(f"examples={len(examples)}")
    print(f"output={args.output}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
