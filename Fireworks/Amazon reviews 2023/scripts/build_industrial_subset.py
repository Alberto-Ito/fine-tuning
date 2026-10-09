#!/usr/bin/env python3
"""Build a product-balanced Industrial & Scientific subset via streaming."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import heapq
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
DEFAULT_SEED = "industrial-scientific-product-balanced-v2"
PROGRESS_EVERY = 100_000


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data" / "industrial_and_scientific_150k")
    parser.add_argument("--core-products", type=int, default=4_000)
    parser.add_argument("--core-reviews", type=int, default=120_000)
    parser.add_argument("--core-min", type=int, default=20)
    parser.add_argument("--core-max", type=int, default=40)
    parser.add_argument("--tail-products", type=int, default=10_000)
    parser.add_argument("--tail-reviews-per-product", type=int, default=3)
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--user-agent", default="amazon-reviews-subset-builder/2.0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (args.core_products, args.core_reviews, args.core_min, args.core_max,
              args.tail_products, args.tail_reviews_per_product, args.min_text_chars)
    if any(value < 1 for value in values):
        raise ValueError("All numeric selection arguments must be positive")
    if args.core_min > args.core_max:
        raise ValueError("--core-min cannot exceed --core-max")
    if not args.core_products * args.core_min <= args.core_reviews <= args.core_products * args.core_max:
        raise ValueError("--core-reviews is incompatible with product/min/max settings")


def stable_hash(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\x1e{value}".encode()).hexdigest()


def review_key(record: dict) -> str:
    return "\x1f".join(str(record.get(key) or "").strip() for key in
                        ("user_id", "parent_asin", "asin", "timestamp", "rating", "title", "text"))


def parse_valid_review(line: str, min_chars: int) -> tuple[dict | None, str]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, "invalid_json"
    text = str(record.get("text") or "").strip()
    parent_asin = str(record.get("parent_asin") or "").strip()
    try:
        rating = float(record.get("rating"))
    except (TypeError, ValueError):
        return None, "invalid_rating"
    if not 1 <= rating <= 5:
        return None, "rating_out_of_range"
    if len(text) < min_chars:
        return None, "text_too_short"
    if not parent_asin:
        return None, "missing_parent_asin"
    record.update(text=text, parent_asin=parent_asin, rating=rating)
    return record, "valid"


@contextlib.contextmanager
def stream_jsonl(url: str, timeout: int, user_agent: str) -> Iterator[TextIO]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    response: BinaryIO = urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context())
    try:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=response, mode="rb") as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    yield stream
        else:
            with io.TextIOWrapper(response, encoding="utf-8") as stream:
                yield stream
    finally:
        response.close()


def atomic_writer(destination: Path) -> tuple[TextIO, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    path = Path(name)
    return gzip.open(path, "wt", encoding="utf-8", compresslevel=6), path


def progress(stage: str, rows: int, started: float, extra: str = "") -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    print(f"[{stage}] rows={rows:,} rate={rows / elapsed:,.0f}/s elapsed={elapsed:,.1f}s {extra}",
          file=sys.stderr, flush=True)


def count_products(args: argparse.Namespace) -> tuple[Counter[str], dict]:
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    seen_review_ids: set[str] = set()
    rows = valid = 0
    started = time.monotonic()
    print("[pass 1/3] Counting valid reviews per parent_asin", file=sys.stderr, flush=True)
    with stream_jsonl(REVIEWS_URL, args.timeout, args.user_agent) as stream:
        for line in stream:
            rows += 1
            record, reason = parse_valid_review(line, args.min_text_chars)
            if record is None:
                rejected[reason] += 1
            else:
                review_id = stable_hash(review_key(record), args.seed + ":review")
                if review_id in seen_review_ids:
                    rejected["duplicate"] += 1
                else:
                    seen_review_ids.add(review_id)
                    counts[record["parent_asin"]] += 1
                    valid += 1
            if rows % PROGRESS_EVERY == 0:
                progress("pass 1/3", rows, started, f"valid={valid:,} products={len(counts):,}")
    progress("pass 1/3 complete", rows, started, f"valid={valid:,} products={len(counts):,}")
    return counts, {"source_rows": rows, "valid_rows": valid,
                    "invalid_rows": dict(rejected), "unique_products": len(counts)}


def make_plan(args: argparse.Namespace, counts: Counter[str]) -> dict[str, int]:
    candidates = [key for key, count in counts.items() if count >= args.core_min]
    candidates.sort(key=lambda key: stable_hash(key, args.seed + ":core"))
    core = candidates[:args.core_products]
    if len(core) < args.core_products:
        raise RuntimeError(f"Only {len(core):,} products meet core minimum")
    plan = {key: args.core_min for key in core}
    remaining = args.core_reviews - sum(plan.values())
    while remaining:
        changed = False
        for key in core:
            cap = min(counts[key], args.core_max)
            if plan[key] < cap:
                plan[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise RuntimeError("Selected core products lack capacity; adjust seed or constraints")

    core_set = set(core)
    tail = [key for key, count in counts.items()
            if key not in core_set and count >= args.tail_reviews_per_product]
    tail.sort(key=lambda key: stable_hash(key, args.seed + ":tail"))
    tail = tail[:args.tail_products]
    if len(tail) < args.tail_products:
        raise RuntimeError(f"Only {len(tail):,} products meet tail minimum")
    plan.update({key: args.tail_reviews_per_product for key in tail})
    return plan


def select_reviews(args: argparse.Namespace, plan: dict[str, int], destination: Path) -> dict:
    heaps: dict[str, list[tuple[int, str, dict]]] = {key: [] for key in plan}
    seen_review_ids: set[str] = set()
    rows = eligible = 0
    started = time.monotonic()
    print(f"[pass 2/3] Selecting reviews for {len(plan):,} products", file=sys.stderr, flush=True)
    with stream_jsonl(REVIEWS_URL, args.timeout, args.user_agent) as stream:
        for line in stream:
            rows += 1
            record, _ = parse_valid_review(line, args.min_text_chars)
            if record is not None and record["parent_asin"] in plan:
                eligible += 1
                review_id = stable_hash(review_key(record), args.seed + ":review")
                if review_id in seen_review_ids:
                    continue
                seen_review_ids.add(review_id)
                rank = int(review_id, 16)
                heap = heaps[record["parent_asin"]]
                entry = (-rank, review_id, record)
                if len(heap) < plan[record["parent_asin"]]:
                    heapq.heappush(heap, entry)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
            if rows % PROGRESS_EVERY == 0:
                progress("pass 2/3", rows, started,
                         f"eligible={eligible:,} retained={sum(map(len, heaps.values())):,}")
    progress("pass 2/3 scan complete", rows, started, f"eligible={eligible:,}")
    shortfalls = [key for key, heap in heaps.items() if len(heap) != plan[key]]
    if shortfalls:
        raise RuntimeError(f"{len(shortfalls):,} products did not meet planned review count")

    writer, temp = atomic_writer(destination)
    ratings: Counter[int] = Counter()
    verified: Counter[str] = Counter()
    kept = 0
    try:
        for parent_asin in sorted(heaps):
            cohort = "core" if plan[parent_asin] > args.tail_reviews_per_product else "tail"
            for _, review_id, record in sorted(heaps[parent_asin], key=lambda item: item[1]):
                normalized = {
                    "review_id": review_id, "category": CATEGORY, "cohort": cohort,
                    "rating": record["rating"], "title": record.get("title"),
                    "text": record["text"], "images": record.get("images") or [],
                    "asin": record.get("asin"), "parent_asin": parent_asin,
                    "user_id": record.get("user_id"), "timestamp": record.get("timestamp"),
                    "helpful_vote": record.get("helpful_vote", 0),
                    "verified_purchase": record.get("verified_purchase"),
                }
                writer.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                ratings[int(record["rating"])] += 1
                verified[str(record.get("verified_purchase"))] += 1
                kept += 1
        writer.close()
        temp.replace(destination)
    except Exception:
        writer.close()
        temp.unlink(missing_ok=True)
        raise
    return {"source_rows": rows, "eligible_rows_for_selected_products": eligible,
            "reviews_kept": kept, "reviews_by_rating": dict(sorted(ratings.items())),
            "verified_purchase": dict(verified)}


def select_metadata(args: argparse.Namespace, plan: dict[str, int], destination: Path) -> dict:
    selected, found = set(plan), set()
    rows = kept = 0
    started = time.monotonic()
    writer, temp = atomic_writer(destination)
    print("[pass 3/3] Selecting matching product metadata", file=sys.stderr, flush=True)
    try:
        with stream_jsonl(METADATA_URL, args.timeout, args.user_agent) as stream:
            for line in stream:
                rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parent_asin = str(record.get("parent_asin") or "").strip()
                if parent_asin in selected:
                    record["category"] = CATEGORY
                    record["cohort"] = "core" if plan[parent_asin] > args.tail_reviews_per_product else "tail"
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    found.add(parent_asin)
                    kept += 1
                if rows % PROGRESS_EVERY == 0:
                    progress("pass 3/3", rows, started, f"metadata_kept={kept:,}")
        writer.close()
        temp.replace(destination)
    except Exception:
        writer.close()
        temp.unlink(missing_ok=True)
        raise
    progress("pass 3/3 complete", rows, started, f"metadata_kept={kept:,}")
    return {"source_rows": rows, "metadata_rows_kept": kept,
            "parent_asins_with_metadata": len(found),
            "parent_asins_without_metadata": len(selected - found)}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reviews, metadata = output / "reviews.jsonl.gz", output / "metadata.jsonl.gz"
    plan_path, manifest = output / "product_plan.json", output / "manifest.json"
    existing = [str(path) for path in (reviews, metadata, plan_path, manifest) if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    print(f"Plan: core={args.core_products:,}/{args.core_reviews:,}; "
          f"tail={args.tail_products:,}/{args.tail_products * args.tail_reviews_per_product:,}",
          file=sys.stderr, flush=True)
    counts, count_stats = count_products(args)
    plan = make_plan(args, counts)
    write_json(plan_path, {"category": CATEGORY, "seed": args.seed,
        "products": [{"parent_asin": key,
                      "cohort": "core" if value > args.tail_reviews_per_product else "tail",
                      "target_reviews": value} for key, value in sorted(plan.items())]})
    print(f"[plan] products={len(plan):,} reviews={sum(plan.values()):,}", file=sys.stderr, flush=True)
    del counts
    review_stats = select_reviews(args, plan, reviews)
    metadata_stats = select_metadata(args, plan, metadata)
    write_json(manifest, {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023", "category": CATEGORY,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"reviews_url": REVIEWS_URL, "metadata_url": METADATA_URL},
        "selection": {"strategy": "product_balanced_core_plus_long_tail",
            "target_reviews": args.core_reviews + args.tail_products * args.tail_reviews_per_product,
            "core_products": args.core_products, "core_reviews": args.core_reviews,
            "core_min": args.core_min, "core_max": args.core_max,
            "tail_products": args.tail_products,
            "tail_reviews_per_product": args.tail_reviews_per_product,
            "min_text_chars": args.min_text_chars, "seed": args.seed},
        "count_pass": count_stats,
        "plan": {"products": len(plan), "planned_reviews": sum(plan.values()),
                 "target_distribution": dict(sorted(Counter(plan.values()).items()))},
        "reviews": review_stats, "metadata": metadata_stats,
        "files": {"reviews": reviews.name, "metadata": metadata.name,
                  "product_plan": plan_path.name}})
    print(json.dumps({"status": "complete", "output_dir": str(output),
          "reviews": review_stats["reviews_kept"], "products": len(plan),
          "products_with_metadata": metadata_stats["parent_asins_with_metadata"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
