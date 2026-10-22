#!/usr/bin/env python3
"""Generate and audit product-level SFT examples with one model pass per stage."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("generate_validated_sft_pipeline.py")
DEFAULT_DATA = ROOT / "data" / "industrial_and_scientific_5k_products"
DEFAULT_OUTPUT = ROOT / "data" / "validated_sft_single_pass_50"
VERSION = "single-pass-batched-v1"


def load_base():
    spec = importlib.util.spec_from_file_location("validated_pipeline_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared pipeline code from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--product-offset", type=int, default=0)
    parser.add_argument("--min-product-reviews", type=int, default=30)
    parser.add_argument("--exclude-products-from", type=Path)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-reviews", type=int, default=8)
    parser.add_argument("--max-evidence-chars", type=int, default=6_000)
    parser.add_argument("--user-model", default="gpt-4.1-mini")
    parser.add_argument("--assistant-model", default="gpt-4o")
    parser.add_argument("--judge-model", default="gpt-6-sol")
    return parser.parse_args()


def chunks(rows: list, size: int) -> list[list]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def compact_context(row: dict) -> dict:
    return {
        "id": row["id"],
        "product": row["product"],
        "reviews": row["reviews"],
    }


def call_batches(batches: list[list[dict]], worker, workers: int) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    errors: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(worker, batch): [row["id"] for row in batch] for batch in batches}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            ids = future_map[future]
            try:
                payload = future.result()
                items = payload.get("items")
                if not isinstance(items, list):
                    raise ValueError("batch response has no items array")
                results.extend(items)
            except Exception as error:
                errors.append({"ids": ids, "error": str(error)})
            print(f"batches={completed}/{len(batches)} items={len(results)} errors={len(errors)}", flush=True)
    return results, errors


def validate_ids(items: list[dict], expected: set[str], stage: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for item in items:
        item_id = item.get("id")
        if item_id not in expected or item_id in indexed:
            raise ValueError(f"{stage}: unexpected or duplicate id {item_id!r}")
        indexed[item_id] = item
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"{stage}: missing ids: {sorted(missing)}")
    return indexed


def triage_prompt(batch: list[dict]) -> str:
    return (
        "For each product, first decide which procurement result is best supported by the supplied "
        "catalog and review evidence: BUY, CONDITIONAL_BUY, DO_NOT_BUY, or INSUFFICIENT_EVIDENCE. "
        "Then create one realistic procurement request for which that result follows naturally. Do not "
        "force a requested label. BUY requires supported core suitability; CONDITIONAL_BUY requires "
        "supported core suitability plus one bounded non-fundamental check; DO_NOT_BUY requires a clear "
        "failure of a realistic mandatory requirement; INSUFFICIENT_EVIDENCE requires one essential "
        "unknown and no independently demonstrated failure. Use two or three non-absolute mandatory "
        "requirements. Do not invent certifications, tolerances, guarantees, minimum lifetime, or safety "
        "claims. `explicitly_unknown` is only missing buyer context, never missing product evidence. The "
        "visible request must not mention Amazon, reviews, ratings, seller, listing, or the chosen label.\n\n"
        "Return JSON only as {\"items\":[{\"id\":\"...\",\"result\":\"BUY|CONDITIONAL_BUY|"
        "DO_NOT_BUY|INSUFFICIENT_EVIDENCE\",\"decision_basis\":\"...\",\"request\":{" 
        "\"item_needed\":\"...\",\"intended_application\":\"...\",\"operating_context\":{},"
        "\"mandatory_requirements\":[{\"requirement\":\"...\",\"reason\":\"...\"}],"
        "\"preferred_requirements\":[],\"explicitly_unknown\":[]}}]}. Include every id once.\n\n"
        f"Products:\n{json.dumps([compact_context(row) for row in batch], ensure_ascii=False)}"
    )


def answer_prompt(batch: list[dict], rules: str) -> str:
    cases = [{"id": row["id"], "system": base.SYSTEM_MESSAGE, "user": row["user"]} for row in batch]
    return (
        "Answer every procurement case independently. Select the correct result from the evidence; you "
        "are not given an authoring target. Attribute catalog statements as seller/listing claims and "
        "reviews as reported experiences. Never generalize one report, infer prevalence from the selected "
        "subset, invent a consequence, or repeat the same point in multiple sections. Missing information "
        "must be decision-relevant. Return JSON only as {\"items\":[{\"id\":\"...\",\"answer\":{" 
        "\"result\":\"BUY|CONDITIONAL_BUY|DO_NOT_BUY|INSUFFICIENT_EVIDENCE\",\"rationale\":\"...\","
        "\"missing_information\":[],\"pros\":[],\"risks_and_limitations\":[],"
        "\"cited_reviews\":[\"R1\"]}}]}. Include every id once.\n\n"
        f"Dataset rules:\n{rules}\n\nCases:\n{json.dumps(cases, ensure_ascii=False)}"
    )


def judge_prompt(batch: list[dict], rules: str) -> str:
    cases = [{
        "id": row["id"], "original_product": row["product"], "original_reviews": row["reviews"],
        "conversation": row["messages"], "author_selected_result": row["selected_result"],
    } for row in batch]
    return (
        "Audit each SFT conversation independently against its original evidence. Validate system, user, "
        "and assistant; label correctness; decision precedence; realism; variant integrity; attribution; "
        "grounding; and concision. The author-selected result is not proof and must be rejected when wrong. "
        "APPROVE only if usable for SFT unchanged. Return JSON only as {\"items\":[{\"id\":\"...\","
        "\"judgment\":{\"verdict\":\"APPROVE|REVISE|REJECT\",\"system_score\":1,\"user_score\":1,"
        "\"assistant_score\":1,\"decision_correct\":true,\"grounded\":true,\"sft_ready\":true,"
        "\"issues\":[],\"recommended_changes\":[]}}]}. Scores are integers from 1 to 5. Include "
        "every id once.\n\n"
        f"Dataset rules:\n{rules}\n\nCases:\n{json.dumps(cases, ensure_ascii=False)}"
    )


def passed(judgment: dict) -> bool:
    return all((
        judgment.get("verdict") == "APPROVE", judgment.get("sft_ready") is True,
        judgment.get("grounded") is True, judgment.get("decision_correct") is True,
        judgment.get("system_score", 0) >= 4, judgment.get("user_score", 0) >= 5,
        judgment.get("assistant_score", 0) >= 4, not judgment.get("issues"),
    ))


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str):
            text = base.normalize_text(item)
        elif isinstance(item, dict):
            text = base.normalize_text(", ".join(str(part) for part in item.values() if part))
        else:
            text = base.normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


def normalize_request(request: dict) -> dict:
    mandatory = []
    for item in request.get("mandatory_requirements") or []:
        if isinstance(item, dict):
            requirement = base.normalize_text(item.get("requirement"))
            reason = base.normalize_text(item.get("reason"))
        else:
            requirement = base.normalize_text(item)
            reason = "Required for the stated procurement application."
        if requirement:
            mandatory.append({"requirement": requirement, "reason": reason or "Required for the stated application."})
    return {
        "item_needed": base.normalize_text(request.get("item_needed")),
        "intended_application": base.normalize_text(request.get("intended_application")),
        "operating_context": request.get("operating_context") if isinstance(request.get("operating_context"), dict) else {},
        "mandatory_requirements": mandatory,
        "preferred_requirements": normalize_string_list(request.get("preferred_requirements")),
        "explicitly_unknown": normalize_string_list(request.get("explicitly_unknown")),
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / filename for name, filename in {
        "master": "examples_master.jsonl", "training": "training.jsonl",
        "pending": "pending_human_review.jsonl", "rejected": "rejected.jsonl",
        "review": "REVIEW.md", "manifest": "manifest.json",
    }.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    env = base.load_env()
    rules = base.RULES_FILE.read_text(encoding="utf-8").strip()
    metadata, counts = base.load_products(args.data_dir)
    if args.product_offset < 0:
        raise ValueError("--product-offset must be non-negative")
    if args.min_product_reviews < 4:
        raise ValueError("--min-product-reviews must be at least 4")
    excluded_parents: set[str] = set()
    if args.exclude_products_from:
        with args.exclude_products_from.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    parent = json.loads(line).get("metadata", {}).get("parent_asin")
                    if parent:
                        excluded_parents.add(str(parent))
    required_pool = args.product_offset + args.examples
    pool_size = max(required_pool, args.candidate_pool_size or required_pool)
    candidate_pool = base.candidate_products(
        metadata, counts, pool_size,
        min_review_count=args.min_product_reviews, excluded_parents=excluded_parents,
    )
    parents = candidate_pool[args.product_offset:]
    all_reviews = base.load_candidate_reviews(args.data_dir, set(parents))
    rows: list[dict] = []
    skipped_products: list[str] = []
    for parent in parents:
        variant, variant_reviews = base.dominant_variant(all_reviews[parent], 12)
        selected = base.select_evidence(variant_reviews, args.max_reviews, args.max_evidence_chars)
        if len(selected) < 4:
            skipped_products.append(parent)
            continue
        index = len(rows) + 1
        rows.append({
            "id": f"E{index:04d}", "index": index, "parent_asin": parent,
            "variant_asin": variant, "product": base.normalized_product(metadata[parent]),
            "reviews": base.review_context(variant_reviews, selected, variant),
        })
        if len(rows) == args.examples:
            break
    if len(rows) < args.examples:
        raise RuntimeError(
            f"Only {len(rows)} products had at least four usable reviews in the dominant variant"
        )

    timings: dict[str, float] = {}
    stage = time.perf_counter()
    triage_items, triage_errors = call_batches(
        chunks(rows, args.batch_size),
        lambda batch: base.call_json(
            env, args.user_model, "Select evidence-supported outcomes and write realistic procurement requests. Return JSON only.",
            triage_prompt(batch), 5000, 2, 0.1,
        )[0], args.workers,
    )
    timings["triage_and_user_seconds"] = round(time.perf_counter() - stage, 2)
    if triage_errors:
        raise RuntimeError(f"triage batch failures: {triage_errors}")
    triage = validate_ids(triage_items, {row["id"] for row in rows}, "triage")
    for row in rows:
        item = triage[row["id"]]
        result, request = item.get("result"), item.get("request")
        if result not in base.RESULTS or not isinstance(request, dict):
            raise ValueError(f"{row['id']}: invalid triage result")
        request = normalize_request(request)
        request = {**request, "hidden_decision_basis": item.get("decision_basis", ""), "target_result": result}
        base.validate_user_result(request, result)
        row["selected_result"] = result
        row["request"] = request
        row["user"] = base.format_user(request, row["product"], row["reviews"])

    stage = time.perf_counter()
    answer_items, answer_errors = call_batches(
        chunks(rows, args.batch_size),
        lambda batch: base.call_json(
            env, args.assistant_model, "Write rigorous purchasing recommendations. Return JSON only.",
            answer_prompt(batch, rules), 6000, 2,
            None if args.assistant_model == "gpt-6-sol" else 0.1,
        )[0], args.workers,
    )
    timings["assistant_seconds"] = round(time.perf_counter() - stage, 2)
    if answer_errors:
        raise RuntimeError(f"assistant batch failures: {answer_errors}")
    answers = validate_ids(answer_items, {row["id"] for row in rows}, "assistant")
    for row in rows:
        answer = answers[row["id"]].get("answer")
        if not isinstance(answer, dict):
            raise ValueError(f"{row['id']}: invalid answer")
        base.validate_answer(answer, len(row["reviews"]["selected_reviews"]))
        row["answer"] = answer
        row["messages"] = [
            {"role": "system", "content": base.SYSTEM_MESSAGE},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": base.format_answer(answer)},
        ]

    stage = time.perf_counter()
    judge_items, judge_errors = call_batches(
        chunks(rows, args.batch_size),
        lambda batch: base.call_json(
            env, args.judge_model, "Audit SFT examples strictly and independently. Return JSON only.",
            judge_prompt(batch, rules), 6500, 2, None,
        )[0], args.workers,
    )
    timings["judge_seconds"] = round(time.perf_counter() - stage, 2)
    if judge_errors:
        raise RuntimeError(f"judge batch failures: {judge_errors}")
    judgments = validate_ids(judge_items, {row["id"] for row in rows}, "judge")

    examples = []
    for row in rows:
        judgment = judgments[row["id"]].get("judgment")
        if not isinstance(judgment, dict):
            raise ValueError(f"{row['id']}: invalid judgment")
        base.validate_judgment(judgment)
        status = "approved" if passed(judgment) else ("rejected" if judgment["verdict"] == "REJECT" else "pending_human_review")
        examples.append({
            "messages": row["messages"],
            "metadata": {
                "example_id": row["id"], "parent_asin": row["parent_asin"],
                "variant_asin": row["variant_asin"], "selected_result": row["selected_result"],
                "actual_result": row["answer"]["result"], "review_status": status,
                "judge": judgment, "pipeline_version": VERSION,
                "models": {"user": args.user_model, "assistant": args.assistant_model, "judge": args.judge_model},
                "source_review_ids": row["reviews"]["source_review_ids"],
            },
        })

    for key in ("master", "training", "pending", "rejected"):
        paths[key].write_text("", encoding="utf-8")
    with paths["master"].open("a", encoding="utf-8") as master, \
         paths["training"].open("a", encoding="utf-8") as training, \
         paths["pending"].open("a", encoding="utf-8") as pending, \
         paths["rejected"].open("a", encoding="utf-8") as rejected:
        for example in examples:
            master.write(json.dumps(example, ensure_ascii=False) + "\n")
            status = example["metadata"]["review_status"]
            destination = training if status == "approved" else rejected if status == "rejected" else pending
            payload = {"messages": example["messages"]} if status == "approved" else example
            destination.write(json.dumps(payload, ensure_ascii=False) + "\n")

    review_lines = ["# Single-pass 50-example pilot", ""]
    for example in examples:
        meta, judge = example["metadata"], example["metadata"]["judge"]
        review_lines.extend([
            f"## {meta['example_id']} — {meta['parent_asin']}", "",
            f"- Selected / actual: `{meta['selected_result']}` / `{meta['actual_result']}`",
            f"- Status: `{meta['review_status']}`; scores: system {judge['system_score']}/5, user {judge['user_score']}/5, assistant {judge['assistant_score']}/5",
            "", "### User", "", "```text", example["messages"][1]["content"], "```", "",
            "### Assistant", "", example["messages"][2]["content"], "", "### Judge issues", "",
            *[f"- {issue}" for issue in judge["issues"] or ["None"]], "",
        ])
    paths["review"].write_text("\n".join(review_lines), encoding="utf-8")
    timings["total_seconds"] = round(time.perf_counter() - started, 2)
    status_counts = Counter(example["metadata"]["review_status"] for example in examples)
    selected_counts = Counter(example["metadata"]["selected_result"] for example in examples)
    actual_counts = Counter(example["metadata"]["actual_result"] for example in examples)
    manifest = {
        "pipeline_version": VERSION, "examples": len(examples), "batch_size": args.batch_size,
        "product_offset": args.product_offset, "min_product_reviews": args.min_product_reviews,
        "excluded_products": len(excluded_parents), "candidate_pool_size": len(candidate_pool),
        "skipped_products": skipped_products,
        "model_invocations": {"triage_and_user": len(chunks(rows, args.batch_size)),
                              "assistant": len(chunks(rows, args.batch_size)),
                              "judge": len(chunks(rows, args.batch_size)),
                              "total": 3 * len(chunks(rows, args.batch_size))},
        "timings": timings, "selected_results": dict(selected_counts),
        "actual_results": dict(actual_counts), "statuses": dict(status_counts),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
