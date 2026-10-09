#!/usr/bin/env python3
"""Validate the locally generated Industrial & Scientific raw subset."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" / "industrial_and_scientific_150k",
    )
    parser.add_argument("--expected-reviews", type=int, default=150_000)
    return parser.parse_args()


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    reviews_path = data_dir / "reviews.jsonl.gz"
    metadata_path = data_dir / "metadata.jsonl.gz"
    manifest_path = data_dir / "manifest.json"
    plan_path = data_dir / "product_plan.json"

    for path in (reviews_path, metadata_path, manifest_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_ids: set[str] = set()
    review_parent_asins: set[str] = set()
    reviews_per_product: Counter[str] = Counter()
    ratings: Counter[int] = Counter()
    verified: Counter[str] = Counter()
    duplicate_review_ids = 0
    missing_required = Counter()
    review_count = 0

    required_review_fields = {
        "review_id",
        "category",
        "rating",
        "text",
        "parent_asin",
        "timestamp",
    }

    for _, record in read_jsonl_gz(reviews_path):
        review_count += 1
        missing = required_review_fields - record.keys()
        for field in missing:
            missing_required[field] += 1
        review_id = record.get("review_id")
        if review_id in review_ids:
            duplicate_review_ids += 1
        review_ids.add(review_id)
        review_parent_asins.add(record.get("parent_asin"))
        reviews_per_product[record.get("parent_asin")] += 1
        ratings[int(float(record.get("rating")))] += 1
        verified[str(record.get("verified_purchase"))] += 1

    metadata_parent_asins: set[str] = set()
    duplicate_metadata_parent_asins = 0
    metadata_count = 0
    for _, record in read_jsonl_gz(metadata_path):
        metadata_count += 1
        parent_asin = record.get("parent_asin")
        if parent_asin in metadata_parent_asins:
            duplicate_metadata_parent_asins += 1
        metadata_parent_asins.add(parent_asin)

    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    planned_counts = {
        item["parent_asin"]: item["target_reviews"]
        for item in plan_payload["products"]
    }
    plan_mismatches = {
        parent_asin: {
            "planned": planned,
            "actual": reviews_per_product.get(parent_asin, 0),
        }
        for parent_asin, planned in planned_counts.items()
        if reviews_per_product.get(parent_asin, 0) != planned
    }
    unexpected_products = set(reviews_per_product) - set(planned_counts)

    report = {
        "data_dir": str(data_dir),
        "reviews": review_count,
        "metadata_rows": metadata_count,
        "unique_review_ids": len(review_ids),
        "unique_review_parent_asins": len(review_parent_asins),
        "unique_metadata_parent_asins": len(metadata_parent_asins),
        "reviews_by_rating": dict(sorted(ratings.items())),
        "verified_purchase": dict(sorted(verified.items())),
        "duplicate_review_ids": duplicate_review_ids,
        "duplicate_metadata_parent_asins": duplicate_metadata_parent_asins,
        "missing_required_fields": dict(sorted(missing_required.items())),
        "parent_asins_without_metadata": len(review_parent_asins - metadata_parent_asins),
        "manifest_target": manifest.get("selection", {}).get("target_reviews"),
        "planned_products": len(planned_counts),
        "product_plan_mismatches": len(plan_mismatches),
        "unexpected_products": len(unexpected_products),
        "reviews_per_product_min": min(reviews_per_product.values()),
        "reviews_per_product_max": max(reviews_per_product.values()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    errors = []
    if review_count != args.expected_reviews:
        errors.append(
            f"expected {args.expected_reviews:,} reviews, found {review_count:,}"
        )
    if duplicate_review_ids:
        errors.append(f"found {duplicate_review_ids:,} duplicate review IDs")
    if missing_required:
        errors.append(f"missing required fields: {dict(missing_required)}")
    if not metadata_count:
        errors.append("metadata output is empty")
    if plan_mismatches:
        errors.append(f"found {len(plan_mismatches):,} product plan mismatches")
    if unexpected_products:
        errors.append(f"found {len(unexpected_products):,} unexpected products")

    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
