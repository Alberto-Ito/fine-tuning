#!/usr/bin/env python3
"""Generate grounded SFT examples with a Foundry teacher and SQLite staging."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


TASKS = {
    "pros_and_cons": "Summarize evidence-supported advantages and limitations.",
    "use_cases": "Identify suitable and unsuitable use cases from the evidence.",
    "common_problems": "Identify recurring problems, impacts, and buyer checks.",
    "purchase_recommendation": "Give a conditional recommendation with risks and verification steps.",
}
SYSTEM_MESSAGE = (
    "You are a technical purchasing copilot. Use only the supplied product data "
    "and review evidence. Separate catalog facts from user reports, quantify "
    "uncertainty, and never invent specifications or safety guarantees."
)
SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS products (
    parent_asin TEXT PRIMARY KEY,
    product_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    parent_asin TEXT NOT NULL,
    rating INTEGER NOT NULL,
    review_json TEXT NOT NULL,
    FOREIGN KEY(parent_asin) REFERENCES products(parent_asin)
);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(parent_asin);
CREATE INDEX IF NOT EXISTS idx_reviews_product_rating ON reviews(parent_asin, rating);
"""
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--target-examples", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-reviews", type=int, default=12)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def stage(connection: sqlite3.Connection, source_path: Path, product_limit: int) -> None:
    existing = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if existing:
        print(f"staging=reused products={existing:,}", flush=True)
        return
    selected: set[str] = set()
    inserted = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as source:
        for source_rows, line in enumerate(source, start=1):
            row = json.loads(line)
            parent = row["source"]["parent_asin"]
            if parent not in selected:
                if len(selected) >= product_limit:
                    continue
                selected.add(parent)
                connection.execute(
                    "INSERT INTO products(parent_asin, product_json) VALUES (?, ?)",
                    (parent, json.dumps(row["product"], ensure_ascii=False)),
                )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?)",
                (
                    row["source"]["review_id"], parent,
                    int(row["review"]["rating"]),
                    json.dumps({"review": row["review"], "source": row["source"]}, ensure_ascii=False),
                ),
            )
            inserted += cursor.rowcount
            if inserted and inserted % 500 == 0:
                connection.commit()
        connection.commit()
    print(
        f"staging=created source_rows={source_rows:,} products={len(selected):,} reviews={inserted:,}",
        flush=True,
    )


def select_evidence(rows: list[dict], limit: int) -> list[dict]:
    def score(row: dict) -> tuple:
        review = row["review"]
        text = review.get("text") or ""
        return int(review.get("helpful_votes") or 0), len(text.split())
    valid = [
        row for row in rows
        if len((row["review"].get("text") or "").split()) >= 8
        and is_english_source(row["review"].get("text") or "")
    ]
    buckets = {
        "negative": sorted((r for r in valid if r["review"]["rating"] <= 2), key=score, reverse=True),
        "mixed": sorted((r for r in valid if r["review"]["rating"] == 3), key=score, reverse=True),
        "positive": sorted((r for r in valid if r["review"]["rating"] >= 4), key=score, reverse=True),
    }
    chosen = buckets["negative"][:4] + buckets["mixed"][:2] + buckets["positive"][:6]
    seen = {r["source"]["review_id"] for r in chosen}
    remaining = sorted((r for r in valid if r["source"]["review_id"] not in seen), key=score, reverse=True)
    return (chosen + remaining)[:limit]


def compact_product(product: dict) -> dict:
    return {
        "name": product.get("name"), "brand_or_store": product.get("brand_or_store"),
        "category": product.get("category"), "subcategories": product.get("subcategories") or [],
        "features": (product.get("features") or [])[:12],
        "technical_details": product.get("technical_details") or {},
        "price_usd": product.get("price_usd"),
        "catalog_average_rating": product.get("catalog_average_rating"),
        "catalog_rating_count": product.get("catalog_rating_count"),
    }


def evidence_payload(rows: list[dict]) -> list[dict]:
    evidence = []
    for index, row in enumerate(rows, start=1):
        review = row["review"]
        text = " ".join((review.get("text") or "").split())[:900]
        evidence.append({
            "id": f"R{index}", "rating": review.get("rating"),
            "verified_purchase": review.get("verified_purchase"),
            "helpful_votes": review.get("helpful_votes"),
            "title": review.get("title"), "text": text,
        })
    return evidence


def teacher_prompt(product: dict, evidence: list[dict]) -> str:
    return (
        "Create exactly four English training answers, one for each requested task. "
        "Every material claim must be supported by the supplied evidence. Mark "
        "needs_human_review=true for category contamination, variant confusion, "
        "contradictory evidence, safety-sensitive recommendations, insufficient "
        "evidence, or any uncertainty that could teach an unreliable behavior. "
        "Classify domain_fit as mro, industrial_support, off_domain, or uncertain. "
        "Educational, entertainment, beauty, or unrelated consumer products are "
        "off_domain even if the catalog category says Industrial & Scientific. "
        "Return JSON only with this shape: {\"domain_fit\":string,"
        "\"domain_fit_reason\":string,\"examples\":[{\"task\":string,"
        "\"answer\":string,\"cited_reviews\":[\"R1\"],"
        "\"needs_human_review\":boolean,\"review_reason\":string}]}.\n\n"
        f"Tasks:\n{json.dumps(TASKS, ensure_ascii=False)}\n\n"
        f"Product:\n{json.dumps(compact_product(product), ensure_ascii=False)}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


def call_foundry(endpoint: str, model: str, api_key: str, prompt: str, retries: int) -> dict:
    url = endpoint.rstrip("/") + "/openai/v1/responses"
    payload = json.dumps({
        "model": model,
        "instructions": SYSTEM_MESSAGE,
        "input": prompt,
        "temperature": 0.1,
        "max_output_tokens": 3000,
    }).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "api-key": api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            text = "".join(
                content.get("text", "")
                for item in body.get("output", []) if item.get("type") == "message"
                for content in item.get("content", []) if content.get("type") == "output_text"
            ).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            return json.loads(text)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(str(error)) from None
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def build_user(task: str, product: dict, evidence: list[dict]) -> str:
    return (
        f"Product data:\n{json.dumps(compact_product(product), ensure_ascii=False)}\n\n"
        f"Selected evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Task:\n{TASKS[task]}"
    )


def is_english(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & {"the", "and", "is", "to", "of", "for", "with", "this", "that"}) >= 3


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


def process_product(item: tuple, args: argparse.Namespace, api_key: str) -> dict:
    parent, product_json, review_rows = item
    product = json.loads(product_json)
    rows = [json.loads(row[0]) for row in review_rows]
    selected = select_evidence(rows, args.max_reviews)
    evidence = evidence_payload(selected)
    if len(evidence) < 3:
        return {"parent": parent, "error": "fewer than three informative reviews"}
    result = call_foundry(args.endpoint, args.model, api_key, teacher_prompt(product, evidence), args.retries)
    domain_fit = result.get("domain_fit", "uncertain")
    domain_reason = result.get("domain_fit_reason") or "domain fit was not provided"
    generated = result.get("examples", [])
    by_task = {example.get("task"): example for example in generated}
    outputs = []
    valid_refs = {item["id"] for item in evidence}
    source_ids = [row["source"]["review_id"] for row in selected]
    for task in TASKS:
        teacher = by_task.get(task) or {}
        answer = (teacher.get("answer") or "").strip()
        citations = teacher.get("cited_reviews") or []
        reasons = []
        if len(answer) < 120: reasons.append("answer_too_short")
        if not is_english(answer): reasons.append("language_check_failed")
        if not citations or any(ref not in valid_refs for ref in citations): reasons.append("invalid_citations")
        if parent.lower() in answer.lower(): reasons.append("identifier_leakage")
        if domain_fit not in {"mro", "industrial_support"}:
            reasons.append(f"domain_fit={domain_fit}: {domain_reason}")
        if teacher.get("needs_human_review", True): reasons.append(teacher.get("review_reason") or "teacher_flagged")
        status = "approved" if not reasons else "pending_human_review"
        example_id = hashlib.sha256(f"{parent}:{task}:foundry-v1".encode()).hexdigest()
        outputs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_user(task, product, evidence)},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "example_id": example_id, "task": task, "parent_asin": parent,
                "source_review_ids": source_ids, "source_review_count": len(source_ids),
                "cited_reviews": citations, "split": "train_candidate", "language": "en",
                "review_status": status, "review_reasons": reasons,
                "teacher_model": args.model, "pipeline_version": "foundry-v1",
                "domain_fit": domain_fit, "domain_fit_reason": domain_reason,
            },
        })
    return {"parent": parent, "examples": outputs}


def load_existing_ids(path: Path) -> set[str]:
    ids = set()
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip(): ids.add(json.loads(line)["metadata"]["example_id"])
    return ids


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("FOUNDRY_API_KEY")
    if not api_key: raise RuntimeError("FOUNDRY_API_KEY is required")
    if args.target_examples % len(TASKS):
        raise ValueError("--target-examples must be divisible by four")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "staging.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    stage(connection, args.input, args.target_examples // len(TASKS))
    connection.execute("UPDATE products SET status='pending' WHERE status='processing'")
    connection.commit()
    master_path = args.output_dir / "examples_master.jsonl"
    fireworks_path = args.output_dir / "training.jsonl"
    pending_path = args.output_dir / "pending_human_review.jsonl"
    existing_ids = load_existing_ids(master_path)
    products = connection.execute(
        "SELECT parent_asin, product_json FROM products WHERE status != 'completed' ORDER BY parent_asin"
    ).fetchall()
    items = []
    for parent, product_json in products:
        rows = connection.execute(
            "SELECT review_json FROM reviews WHERE parent_asin=? ORDER BY rating, review_id", (parent,)
        ).fetchall()
        items.append((parent, product_json, rows))
    counts = {"master": len(existing_ids), "approved": 0, "pending": 0, "failed_products": 0}
    mode = "a" if master_path.exists() else "w"
    with (
        master_path.open(mode, encoding="utf-8") as master,
        fireworks_path.open("a" if fireworks_path.exists() else "w", encoding="utf-8") as fireworks,
        pending_path.open("a" if pending_path.exists() else "w", encoding="utf-8") as pending,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        future_map = {pool.submit(process_product, item, args, api_key): item[0] for item in items}
        for future in concurrent.futures.as_completed(future_map):
            parent = future_map[future]
            try:
                result = future.result()
                if result.get("error"): raise RuntimeError(result["error"])
                for example in result["examples"]:
                    eid = example["metadata"]["example_id"]
                    if eid in existing_ids: continue
                    line = json.dumps(example, ensure_ascii=False) + "\n"
                    master.write(line); master.flush()
                    if example["metadata"]["review_status"] == "approved":
                        fireworks.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + "\n"); fireworks.flush()
                        counts["approved"] += 1
                    else:
                        pending.write(line); pending.flush(); counts["pending"] += 1
                    existing_ids.add(eid); counts["master"] += 1
                connection.execute("UPDATE products SET status='completed', attempts=attempts+1, error=NULL WHERE parent_asin=?", (parent,))
            except Exception as error:
                counts["failed_products"] += 1
                connection.execute("UPDATE products SET status='failed', attempts=attempts+1, error=? WHERE parent_asin=?", (str(error)[:1000], parent))
            connection.commit()
            done = counts["master"]
            if done % 100 == 0 or done >= args.target_examples:
                print(f"progress={done:,}/{args.target_examples:,} approved={counts['approved']:,} pending={counts['pending']:,} failed_products={counts['failed_products']:,}", flush=True)
    manifest = {
        "pipeline_version": "foundry-v1", "teacher_model": args.model,
        "target_examples": args.target_examples, "counts_this_run": counts,
        "files": {"master": str(master_path), "fireworks": str(fireworks_path), "pending": str(pending_path), "staging": str(db_path)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    connection.close()
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
