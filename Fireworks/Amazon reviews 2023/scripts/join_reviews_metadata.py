#!/usr/bin/env python3
"""Join curated Amazon reviews with their product metadata.

The resulting JSONL is the canonical, human-readable intermediate dataset used
later to build train/validation/test and SFT conversations. Identifiers are
kept only under ``source`` for traceability and leakage-safe splitting.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default="enriched_reviews.jsonl.gz",
        help="Output filename inside --data-dir.",
    )
    return parser.parse_args()


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def clean_list(value) -> list:
    return value if isinstance(value, list) else []


def build_record(review: dict, product: dict) -> dict:
    return {
        "product": {
            "name": product.get("title"),
            "brand_or_store": product.get("store"),
            "category": product.get("main_category"),
            "subcategories": clean_list(product.get("categories")),
            "description": clean_list(product.get("description")),
            "features": clean_list(product.get("features")),
            "technical_details": product.get("details") or {},
            "price_usd": product.get("price"),
            "catalog_average_rating": product.get("average_rating"),
            "catalog_rating_count": product.get("rating_number"),
        },
        "review": {
            "rating": review.get("rating"),
            "title": review.get("title"),
            "text": review.get("text"),
            "verified_purchase": review.get("verified_purchase"),
            "helpful_votes": review.get("helpful_vote"),
            "timestamp_ms": review.get("timestamp"),
        },
        "source": {
            "dataset": "Amazon Reviews 2023",
            "category": review.get("category"),
            "cohort": review.get("cohort"),
            "review_id": review.get("review_id"),
            "asin": review.get("asin"),
            "parent_asin": review.get("parent_asin"),
        },
    }


def main() -> None:
    args = parse_args()
    reviews_path = args.data_dir / "reviews.jsonl.gz"
    metadata_path = args.data_dir / "metadata.jsonl.gz"
    output_path = args.data_dir / args.output_name

    for path in (reviews_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    metadata = {}
    for line_number, product in read_jsonl_gz(metadata_path):
        parent_asin = product.get("parent_asin")
        if not parent_asin:
            raise ValueError(f"Missing parent_asin in {metadata_path}:{line_number}")
        if parent_asin in metadata:
            raise ValueError(f"Duplicate metadata parent_asin: {parent_asin}")
        metadata[parent_asin] = product

    written = 0
    missing = set()
    with gzip.open(output_path, "wt", encoding="utf-8") as destination:
        for _, review in read_jsonl_gz(reviews_path):
            parent_asin = review.get("parent_asin")
            product = metadata.get(parent_asin)
            if product is None:
                missing.add(parent_asin)
                continue
            destination.write(
                json.dumps(build_record(review, product), ensure_ascii=False) + "\n"
            )
            written += 1
            if written % 25_000 == 0:
                print(f"joined={written:,}", flush=True)

    if missing:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Join failed: {len(missing):,} parent_asin values have no metadata"
        )

    print(f"completed={written:,}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
