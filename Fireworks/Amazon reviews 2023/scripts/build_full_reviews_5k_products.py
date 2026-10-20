#!/usr/bin/env python3
"""Extract every source review for 4,000 core and 1,000 deterministic tail products."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


CATEGORY = "Industrial_and_Scientific"
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
REVIEWS_URL = f"{BASE_URL}/review_categories/{CATEGORY}.jsonl"
METADATA_URL = f"{BASE_URL}/meta_categories/meta_{CATEGORY}.jsonl"
TAIL_SEED = "industrial-and-scientific-5k-products-tail-v1"
PROGRESS_EVERY = 100_000


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-plan",
        type=Path,
        default=project_dir / "data" / "industrial_and_scientific_150k" / "product_plan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "industrial_and_scientific_5k_products",
    )
    parser.add_argument("--tail-products", type=int, default=1_000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--user-agent", default="amazon-reviews-full-product-extractor/1.0")
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{TAIL_SEED}\x1e{value}".encode()).hexdigest()


def load_selection(plan_path: Path, tail_limit: int) -> tuple[dict[str, str], list[dict]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    products = payload.get("products", [])
    core = sorted(
        str(item["parent_asin"]) for item in products if item.get("cohort") == "core"
    )
    tail_candidates = [
        str(item["parent_asin"]) for item in products if item.get("cohort") == "tail"
    ]
    tail = sorted(tail_candidates, key=lambda parent_asin: (stable_hash(parent_asin), parent_asin))[
        :tail_limit
    ]
    if len(core) != 4_000:
        raise ValueError(f"Expected 4,000 core products, found {len(core):,}")
    if len(tail) != tail_limit:
        raise ValueError(f"Expected {tail_limit:,} tail products, found {len(tail):,}")
    selection = {parent_asin: "core" for parent_asin in core}
    selection.update({parent_asin: "tail" for parent_asin in tail})
    product_rows = [
        {"parent_asin": parent_asin, "cohort": selection[parent_asin]}
        for parent_asin in sorted(selection)
    ]
    return selection, product_rows


@contextlib.contextmanager
def stream_jsonl(url: str, timeout: int, user_agent: str) -> Iterator[TextIO]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    response: BinaryIO = urllib.request.urlopen(
        request, timeout=timeout, context=ssl.create_default_context()
    )
    try:
        with io.TextIOWrapper(response, encoding="utf-8") as stream:
            yield stream
    finally:
        response.close()


@contextlib.contextmanager
def atomic_gzip_writer(destination: Path) -> Iterator[TextIO]:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as writer:
            yield writer
        temporary.replace(destination)
        destination.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def report(stage: str, rows: int, kept: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"[{stage}] source_rows={rows:,} kept={kept:,} "
        f"rate={rows / elapsed:,.0f}/s elapsed={elapsed:,.1f}s",
        file=sys.stderr,
        flush=True,
    )


def extract_reviews(
    selection: dict[str, str], destination: Path, timeout: int, user_agent: str
) -> tuple[dict, Counter[str]]:
    source_rows = kept = invalid_json = missing_parent_asin = 0
    counts: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    started = time.monotonic()
    with atomic_gzip_writer(destination) as writer:
        with stream_jsonl(REVIEWS_URL, timeout, user_agent) as stream:
            for line in stream:
                source_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                parent_asin = str(record.get("parent_asin") or "").strip()
                if not parent_asin:
                    missing_parent_asin += 1
                elif parent_asin in selection:
                    # Preserve every matching source row; no filtering or deduplication.
                    writer.write(line if line.endswith("\n") else line + "\n")
                    counts[parent_asin] += 1
                    ratings[str(record.get("rating"))] += 1
                    kept += 1
                if source_rows % PROGRESS_EVERY == 0:
                    report("reviews", source_rows, kept, started)
    report("reviews complete", source_rows, kept, started)
    return (
        {
            "source_rows": source_rows,
            "reviews_kept": kept,
            "invalid_json_rows": invalid_json,
            "rows_missing_parent_asin": missing_parent_asin,
            "products_with_reviews": len(counts),
            "products_without_reviews": len(set(selection) - set(counts)),
            "reviews_per_product_min": min(counts.values()) if counts else 0,
            "reviews_per_product_max": max(counts.values()) if counts else 0,
            "reviews_by_rating": dict(sorted(ratings.items())),
        },
        counts,
    )


def extract_metadata(
    selection: dict[str, str], destination: Path, timeout: int, user_agent: str
) -> dict:
    source_rows = kept = invalid_json = 0
    found: set[str] = set()
    started = time.monotonic()
    with atomic_gzip_writer(destination) as writer:
        with stream_jsonl(METADATA_URL, timeout, user_agent) as stream:
            for line in stream:
                source_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                parent_asin = str(record.get("parent_asin") or "").strip()
                if parent_asin in selection:
                    writer.write(line if line.endswith("\n") else line + "\n")
                    found.add(parent_asin)
                    kept += 1
                if source_rows % PROGRESS_EVERY == 0:
                    report("metadata", source_rows, kept, started)
    report("metadata complete", source_rows, kept, started)
    return {
        "source_rows": source_rows,
        "metadata_rows_kept": kept,
        "invalid_json_rows": invalid_json,
        "products_with_metadata": len(found),
        "products_without_metadata": len(set(selection) - found),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reviews_path = output / "reviews.jsonl.gz"
    metadata_path = output / "metadata.jsonl.gz"
    products_path = output / "products.json"
    manifest_path = output / "manifest.json"
    existing = [path for path in (reviews_path, metadata_path, products_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))

    selection, product_rows = load_selection(args.source_plan, args.tail_products)
    write_json(
        products_path,
        {
            "category": CATEGORY,
            "selection": {
                "core_products": 4_000,
                "tail_products": args.tail_products,
                "tail_strategy": "lowest deterministic SHA-256 ranks",
                "tail_seed": TAIL_SEED,
                "source_plan": str(args.source_plan.resolve()),
            },
            "products": product_rows,
        },
    )
    review_stats, review_counts = extract_reviews(
        selection, reviews_path, args.timeout, args.user_agent
    )
    metadata_stats = extract_metadata(
        selection, metadata_path, args.timeout, args.user_agent
    )
    for item in product_rows:
        item["review_count"] = review_counts[item["parent_asin"]]
    write_json(
        products_path,
        {
            "category": CATEGORY,
            "selection": {
                "core_products": 4_000,
                "tail_products": args.tail_products,
                "tail_strategy": "lowest deterministic SHA-256 ranks",
                "tail_seed": TAIL_SEED,
                "source_plan": str(args.source_plan.resolve()),
            },
            "products": product_rows,
        },
    )
    manifest = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": CATEGORY,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"reviews_url": REVIEWS_URL, "metadata_url": METADATA_URL},
        "selection": {
            "strategy": "existing_4000_core_plus_deterministic_1000_tail",
            "products": len(selection),
            "core_products": 4_000,
            "tail_products": args.tail_products,
            "tail_seed": TAIL_SEED,
        },
        "review_policy": {
            "all_matching_source_rows": True,
            "text_length_filter": None,
            "rating_filter": None,
            "deduplication": False,
            "per_product_limit": None,
        },
        "reviews": review_stats,
        "metadata": metadata_stats,
        "files": {
            "reviews": reviews_path.name,
            "metadata": metadata_path.name,
            "products": products_path.name,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", **manifest["selection"], **review_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
