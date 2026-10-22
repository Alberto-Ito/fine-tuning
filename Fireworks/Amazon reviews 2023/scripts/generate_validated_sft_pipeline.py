#!/usr/bin/env python3
"""Generate product-level SFT examples with separate author, answer, and judge models."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "industrial_and_scientific_5k_products"
DEFAULT_OUTPUT = ROOT / "data" / "validated_sft_pilot_10_v2"
ENV_FILE = ROOT.parent / "ZavaRetailAgent" / ".env"
RULES_FILE = ROOT / "DATASET_REVIEW_RULES.md"
PIPELINE_VERSION = "validated-three-model-v2"
SELECTION_SEED = "validated-sft-pipeline-v2"
RESULTS = ("BUY", "CONDITIONAL_BUY", "DO_NOT_BUY", "INSUFFICIENT_EVIDENCE")
LOCK = threading.Lock()

SYSTEM_MESSAGE = (
    "You are a purchasing copilot for an organizational procurement department. "
    "Evaluate the supplied product against the procurement request using only catalog claims, "
    "complete review statistics, and selected review evidence. Distinguish seller claims from "
    "user-reported experience. Do not invent requirements, specifications, compatibility, "
    "certifications, guarantees, causes, frequency, or safety. Use exactly one result: BUY, "
    "CONDITIONAL_BUY, DO_NOT_BUY, or INSUFFICIENT_EVIDENCE. Use CONDITIONAL_BUY only for a "
    "concrete limited verification that does not determine fundamental compatibility; if a missing "
    "fact determines whether the product can perform the intended application, use "
    "INSUFFICIENT_EVIDENCE. Keep the response concise. Return exactly this structure, with no "
    "prose before or after it:\n\n"
    "Purchase recommendation:\n"
    "- Result: BUY | CONDITIONAL_BUY | DO_NOT_BUY | INSUFFICIENT_EVIDENCE\n"
    "- Rationale: <concise primary reason for the decision>\n\n"
    "Missing information:\n"
    "- <decision-relevant missing information>\n\n"
    "Pros:\n"
    "- <evidence-supported advantage>\n\n"
    "Risks and limitations:\n"
    "- <evidence-supported risk or limitation>\n\n"
    "Include exactly one Result. Omit the entire Missing information section when no missing "
    "information materially affects the decision. Keep Pros and Risks and limitations grounded "
    "in the supplied evidence; when evidence is limited, state that limitation instead of "
    "inventing content."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--target-result", choices=RESULTS)
    parser.add_argument("--user-model", default="gpt-4.1-mini")
    parser.add_argument("--assistant-model", default="gpt-4o")
    parser.add_argument("--judge-model", default="gpt-6-sol")
    parser.add_argument("--max-reviews", type=int, default=20)
    parser.add_argument("--max-evidence-chars", type=int, default=14_000)
    parser.add_argument("--min-variant-reviews", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    required = ("FOUNDRY_BASE_URL", "FOUNDRY_API_KEY")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError("Missing Foundry settings: " + ", ".join(missing))
    return values


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\x1e{value}".encode()).hexdigest()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def looks_english(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    english = words & {
        "the", "and", "is", "it", "to", "of", "for", "with", "this", "that",
        "was", "are", "my", "but", "not", "have", "has", "in", "on", "as",
        "from", "works", "worked", "product", "use", "used", "good", "great",
    }
    other = words & {
        "el", "la", "los", "las", "que", "para", "una", "pero", "con", "por",
        "producto", "les", "des", "une", "pas", "avec", "und", "der", "die", "das",
    }
    return len(english) >= 2 and len(english) > len(other)


def parse_json_text(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def extract_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()
    chunks: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


def call_json(
    env: dict[str, str], model: str, instructions: str, prompt: str,
    max_output_tokens: int, retries: int, temperature: float | None = None,
) -> tuple[dict, dict]:
    url = env["FOUNDRY_BASE_URL"].rstrip("/") + "/responses"
    payload: dict[str, object] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "api-key": env["FOUNDRY_API_KEY"],
        })
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body = json.load(response)
            return parse_json_text(extract_text(body)), body.get("usage", {})
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError,
                json.JSONDecodeError, ValueError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"{model}: {error}") from None
            time.sleep(min(2 ** attempt + random.random(), 20))
    raise AssertionError("unreachable")


def load_products(data_dir: Path) -> tuple[dict[str, dict], dict[str, int]]:
    counts_payload = json.loads((data_dir / "products.json").read_text(encoding="utf-8"))
    counts = {row["parent_asin"]: int(row["review_count"]) for row in counts_payload["products"]}
    metadata: dict[str, dict] = {}
    with gzip.open(data_dir / "metadata.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            metadata[row["parent_asin"]] = row
    return metadata, counts


def candidate_products(
    metadata: dict[str, dict], counts: dict[str, int], needed: int,
    min_review_count: int = 30, excluded_parents: set[str] | None = None,
) -> list[str]:
    excluded_parents = excluded_parents or set()
    groups: dict[str, list[str]] = defaultdict(list)
    for parent, product in metadata.items():
        if parent in excluded_parents:
            continue
        if product.get("main_category") != "Industrial & Scientific" or counts.get(parent, 0) < min_review_count:
            continue
        categories = product.get("categories") or []
        group = str(categories[1] if len(categories) > 1 else "Other")
        if not normalize_text(product.get("title")):
            continue
        groups[group].append(parent)
    for values in groups.values():
        values.sort(key=lambda parent: (stable_rank(parent), parent))
    names = sorted(groups, key=lambda name: stable_rank("group:" + name))
    selected: list[str] = []
    while len(selected) < needed:
        changed = False
        for name in names:
            if groups[name]:
                selected.append(groups[name].pop(0))
                changed = True
                if len(selected) == needed:
                    break
        if not changed:
            break
    if len(selected) < needed:
        raise RuntimeError(f"Only {len(selected)} eligible candidate products")
    return selected


def load_candidate_reviews(data_dir: Path, candidates: set[str]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(data_dir / "reviews.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            review = json.loads(line)
            parent = str(review.get("parent_asin") or "")
            if parent in candidates:
                rows[parent].append(review)
    return rows


def assign_candidates(
    candidates: list[str], targets: list[str], metadata: dict[str, dict],
    reviews: dict[str, list[dict]],
) -> tuple[list[str], list[str]]:
    """Assign a plausible product to each target, retaining unused timeout replacements."""
    def rank(parent: str, target: str) -> tuple:
        rows = reviews.get(parent, [])
        ratings = [float(row.get("rating") or 0) for row in rows]
        average = sum(ratings) / len(ratings) if ratings else 0
        negative = sum(value <= 2 for value in ratings) / len(ratings) if ratings else 1
        product = metadata[parent]
        richness = len(product.get("features") or []) + len(product.get("description") or [])
        if target == "BUY":
            return (-average, negative, -richness, stable_rank(parent))
        if target == "CONDITIONAL_BUY":
            return (abs(average - 4.15), abs(negative - 0.12), -richness, stable_rank(parent))
        if target == "DO_NOT_BUY":
            return (average, -negative, -richness, stable_rank(parent))
        return (richness, abs(average - 3.8), -negative, stable_rank(parent))

    remaining = set(candidates)
    assigned: list[str] = []
    for target in targets:
        parent = min(remaining, key=lambda value: rank(value, target))
        assigned.append(parent)
        remaining.remove(parent)
    spares = sorted(remaining, key=lambda value: stable_rank("spare:" + value))
    return assigned, spares


def dominant_variant(reviews: list[dict], minimum: int) -> tuple[str, list[dict]]:
    by_asin: dict[str, list[dict]] = defaultdict(list)
    for review in reviews:
        asin = str(review.get("asin") or "").strip()
        if asin:
            by_asin[asin].append(review)
    if not by_asin:
        return "unknown", reviews
    asin, rows = max(by_asin.items(), key=lambda item: (len(item[1]), item[0]))
    return (asin, rows) if len(rows) >= minimum else ("mixed", reviews)


def normalized_product(product: dict) -> dict:
    details = product.get("details") or {}
    product_fields: dict[str, object] = {}
    package_or_ambiguous: dict[str, object] = {}
    for key, value in details.items():
        if isinstance(value, (dict, list)) or len(str(value)) > 500:
            continue
        lowered = str(key).lower()
        target = package_or_ambiguous if "package" in lowered or key == "Item Weight" else product_fields
        target[str(key)] = value
    return {
        "catalog_identity": {
            "title": normalize_text(product.get("title")),
            "brand_or_store": product.get("store"),
            "categories": (product.get("categories") or [])[:6],
        },
        "seller_catalog_claims": {
            "features": [normalize_text(value)[:500] for value in (product.get("features") or [])[:10]],
            "description": [normalize_text(value)[:700] for value in (product.get("description") or [])[:4]],
            "technical_details": product_fields,
        },
        "package_or_ambiguous_fields": package_or_ambiguous,
        "catalog_rating_summary": {
            "average_rating": product.get("average_rating"),
            "rating_count": product.get("rating_number"),
        },
        "field_warning": (
            "Catalog fields are seller/listing claims. Package dimensions and Item Weight must not "
            "be treated as individual product dimensions or unit weight without explicit confirmation."
        ),
    }


def review_score(review: dict) -> tuple[int, int, int]:
    return (
        int(review.get("helpful_vote") or 0),
        1 if review.get("verified_purchase") else 0,
        len(normalize_text(review.get("text"))),
    )


def select_evidence(reviews: list[dict], limit: int, char_budget: int) -> list[dict]:
    valid: list[dict] = []
    seen: set[str] = set()
    for review in reviews:
        text = normalize_text(review.get("text"))
        fingerprint = hashlib.sha256(text.lower().encode()).hexdigest()
        if len(text.split()) < 8 or not looks_english(text) or fingerprint in seen:
            continue
        seen.add(fingerprint)
        valid.append(review)
    buckets = {
        rating: sorted(
            (row for row in valid if int(float(row.get("rating") or 0)) == rating),
            key=review_score,
            reverse=True,
        )
        for rating in range(1, 6)
    }
    ordered: list[dict] = []
    for rating in range(1, 6):
        if buckets[rating]:
            ordered.append(buckets[rating].pop(0))
    ordered.extend(sorted(
        (row for bucket in buckets.values() for row in bucket), key=review_score, reverse=True
    ))
    selected: list[dict] = []
    chars = 0
    for row in ordered:
        if len(selected) >= limit:
            break
        copy = dict(row)
        copy["text"] = normalize_text(row.get("text"))[:900]
        estimate = len(copy["text"]) + len(normalize_text(copy.get("title"))) + 120
        if selected and chars + estimate > char_budget:
            continue
        selected.append(copy)
        chars += estimate
    return selected


def review_context(all_reviews: list[dict], selected: list[dict], variant_asin: str) -> dict:
    ratings = Counter(int(float(row.get("rating") or 0)) for row in all_reviews)
    evidence = []
    source_ids = []
    for index, row in enumerate(selected, start=1):
        evidence.append({
            "id": f"R{index}", "variant_asin": row.get("asin"), "rating": row.get("rating"),
            "verified_purchase": row.get("verified_purchase"),
            "helpful_votes": row.get("helpful_vote", 0),
            "title": normalize_text(row.get("title")), "text": normalize_text(row.get("text")),
        })
        source_ids.append(hashlib.sha256(
            "\x1f".join(str(row.get(key) or "").strip() for key in
                         ("user_id", "parent_asin", "asin", "timestamp", "rating", "title", "text")).encode()
        ).hexdigest())
    return {
        "variant_asin": variant_asin,
        "complete_variant_review_statistics": {
            "reviews": len(all_reviews),
            "ratings": {str(value): ratings.get(value, 0) for value in range(1, 6)},
            "verified_purchase_reviews": sum(bool(row.get("verified_purchase")) for row in all_reviews),
        },
        "selection_note": (
            "Selected evidence favors useful and diverse reports and is not a frequency estimate. "
            "Only complete statistics may support prevalence claims."
        ),
        "selected_reviews": evidence,
        "source_review_ids": source_ids,
    }


def validate_user_result(value: dict, target: str) -> None:
    required = {
        "item_needed", "intended_application", "operating_context", "mandatory_requirements",
        "preferred_requirements", "explicitly_unknown", "hidden_decision_basis", "target_result",
    }
    if set(value) != required or value.get("target_result") != target:
        raise ValueError("Invalid user-generation schema or target")
    if not all(isinstance(value.get(key), str) and value[key].strip()
               for key in ("item_needed", "intended_application", "hidden_decision_basis")):
        raise ValueError("Required user-generation text is empty")
    if not isinstance(value["operating_context"], dict):
        raise ValueError("operating_context must be an object")
    if not isinstance(value["mandatory_requirements"], list) or not value["mandatory_requirements"]:
        raise ValueError("mandatory_requirements must be a nonempty array")
    for item in value["mandatory_requirements"]:
        if not isinstance(item, dict) or set(item) != {"requirement", "reason"}:
            raise ValueError("Invalid mandatory requirement")
    for key in ("preferred_requirements", "explicitly_unknown"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"{key} must be a string array")


def user_author_prompt(
    target: str, product: dict, reviews: dict, feedback: list[str]
) -> str:
    target_rule = {
        "BUY": (
            "Choose a modest, ordinary use that the catalog and multiple relevant reports support. "
            "Use only two or three non-absolute mandatory requirements. Do not require certifications, "
            "guaranteed lifetime, exact tolerances, exact delivery quantity, or universal defect-free performance."
        ),
        "CONDITIONAL_BUY": (
            "Choose a modest use whose core suitability is supported. Add exactly one bounded, practical "
            "condition such as confirming the selected size/variant or accepting a documented limitation. "
            "The condition must not determine basic functional or safety compatibility."
        ),
        "DO_NOT_BUY": (
            "Use only two or three realistic requirements. At least one must be directly contradicted by "
            "a clear, relevant user report or catalog fact. Do not turn one report into a prevalence claim."
        ),
        "INSUFFICIENT_EVIDENCE": (
            "Use only two or three realistic requirements. Exactly one essential compatibility or safety "
            "fact must be genuinely absent or materially contradictory, and no other requirement may have "
            "a demonstrated failure that would force DO_NOT_BUY."
        ),
    }[target]
    feedback_text = json.dumps(feedback, ensure_ascii=False) if feedback else "[]"
    return (
        "Create one realistic procurement request for the product family represented below. You may "
        "use product and review evidence to ensure the scenario is answerable, but the visible request "
        "must sound like an independent buyer need and must not mention the target label, reviews, "
        "ratings, Amazon, the seller, or the candidate listing. Do not copy package dimensions, item "
        "weight, malformed fields, marketing language, or exact listing values unless independently "
        "necessary and plausible. Do not invent certifications. Requirements must be internally "
        "consistent, technically relevant, and specific enough to audit. A review can support a reported "
        "experience, but not a guarantee for every unit. Never make a seller claim or a single review carry "
        "an absolute requirement. `explicitly_unknown` is only for buyer context the requester omitted "
        "(for example substrate, size, or environment), never for missing product evidence.\n\n"
        f"Hidden target result: {target}. {target_rule}\n\n"
        "Return JSON only with exactly this shape: "
        '{"item_needed":"string","intended_application":"string","operating_context":{},'
        '"mandatory_requirements":[{"requirement":"string","reason":"string"}],'
        '"preferred_requirements":["string"],"explicitly_unknown":["string"],'
        '"hidden_decision_basis":"string","target_result":"string"}.\n\n'
        f"Feedback from a previous validation attempt:\n{feedback_text}\n\n"
        f"Normalized product data:\n{json.dumps(product, ensure_ascii=False)}\n\n"
        f"Original review context:\n{json.dumps(reviews, ensure_ascii=False)}"
    )


def format_user(request: dict, product: dict, reviews: dict) -> str:
    lines = [
        "Procurement request:",
        f"Item needed: {normalize_text(request['item_needed'])}",
        f"Intended application: {normalize_text(request['intended_application'])}",
    ]
    context = {key: value for key, value in request["operating_context"].items() if value not in (None, "", [])}
    if context:
        lines.extend(["", "Operating context:", json.dumps(context, ensure_ascii=False)])
    lines.extend(["", "Mandatory requirements:"])
    lines.extend(
        f"- {normalize_text(item['requirement'])} — {normalize_text(item['reason'])}"
        for item in request["mandatory_requirements"]
    )
    if request["preferred_requirements"]:
        lines.extend(["", "Preferred requirements:"])
        lines.extend(f"- {normalize_text(item)}" for item in request["preferred_requirements"])
    if request["explicitly_unknown"]:
        lines.extend(["", "Information not provided by the requester:"])
        lines.extend(f"- {normalize_text(item)}" for item in request["explicitly_unknown"])
    visible_reviews = {key: value for key, value in reviews.items() if key != "source_review_ids"}
    lines.extend([
        "", "Normalized product data:", json.dumps(product, ensure_ascii=False),
        "", "Original Amazon review context:", json.dumps(visible_reviews, ensure_ascii=False),
        "", "Assess whether this specific product is suitable for the procurement request. Use only "
        "the supplied information. Treat catalog content as seller claims and reviews as user reports. "
        "Do not infer frequency from the selected evidence subset.",
    ])
    return "\n".join(lines)


def validate_answer(value: dict, evidence_count: int) -> None:
    required = {"result", "rationale", "missing_information", "pros", "risks_and_limitations", "cited_reviews"}
    if set(value) != required or value.get("result") not in RESULTS:
        raise ValueError("Invalid assistant response schema")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("Assistant rationale is empty")
    for key in ("missing_information", "pros", "risks_and_limitations", "cited_reviews"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"{key} must be a string array")
    allowed = {f"R{index}" for index in range(1, evidence_count + 1)}
    if any(reference not in allowed for reference in value["cited_reviews"]):
        raise ValueError("Assistant cited an unknown review")


def assistant_prompt(user_message: str, rules: str, feedback: list[str]) -> str:
    return (
        "Answer the complete user message as one high-quality SFT assistant response. Follow the "
        "dataset rules exactly. Do not see or infer a desired label; choose the correct result from "
        "the supplied request and evidence. Seller/catalog claims are not independently verified. "
        "Variant-specific review evidence must not be generalized. A selected evidence subset does "
        "not establish prevalence. Phrase catalog facts as seller/listing claims and review facts as "
        "individual or grouped reports. Do not convert a report into a universal defect, add a causal "
        "consequence, or claim an exact threshold unless the evidence states it. Do not repeat the same "
        "point across rationale, missing information, and risks. Mention missing information only when it "
        "could change the decision. Return JSON only with exactly this shape: "
        '{"result":"BUY|CONDITIONAL_BUY|DO_NOT_BUY|INSUFFICIENT_EVIDENCE",'
        '"rationale":"string","missing_information":["string"],"pros":["string"],'
        '"risks_and_limitations":["string"],"cited_reviews":["R1"]}. '
        "Use zero to four concise items per list; do not invent content to fill a quota.\n\n"
        f"Previous validator feedback:\n{json.dumps(feedback, ensure_ascii=False)}\n\n"
        f"Dataset rules:\n{rules}\n\nSystem message:\n{SYSTEM_MESSAGE}\n\nUser message:\n{user_message}"
    )


def format_answer(value: dict) -> str:
    lines = [
        "Purchase recommendation:",
        f"- Result: {value['result']}",
        f"- Rationale: {normalize_text(value['rationale'])}",
    ]
    if value["missing_information"]:
        lines.extend(["", "Missing information:"])
        lines.extend(f"- {normalize_text(item)}" for item in value["missing_information"])
    lines.extend(["", "Pros:"])
    lines.extend(f"- {normalize_text(item)}" for item in value["pros"])
    if not value["pros"]:
        lines.append("- No material advantage is established by the supplied evidence.")
    lines.extend(["", "Risks and limitations:"])
    lines.extend(f"- {normalize_text(item)}" for item in value["risks_and_limitations"])
    if not value["risks_and_limitations"]:
        lines.append("- No specific material risk is established by the supplied evidence.")
    return "\n".join(lines)


def validate_judgment(value: dict) -> None:
    required = {
        "verdict", "system_score", "user_score", "assistant_score", "decision_correct",
        "grounded", "sft_ready", "issues", "recommended_changes",
    }
    if set(value) != required or value.get("verdict") not in {"APPROVE", "REVISE", "REJECT"}:
        raise ValueError("Invalid judge response schema")
    for key in ("system_score", "user_score", "assistant_score"):
        if not isinstance(value[key], int) or not 1 <= value[key] <= 5:
            raise ValueError(f"Invalid {key}")
    for key in ("decision_correct", "grounded", "sft_ready"):
        if not isinstance(value[key], bool):
            raise ValueError(f"Invalid {key}")
    for key in ("issues", "recommended_changes"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"Invalid {key}")


def judge_prompt(
    target: str, product: dict, reviews: dict, messages: list[dict], rules: str
) -> str:
    return (
        "Audit one complete SFT conversation against the original Amazon Reviews 2023 evidence. "
        "Validate system, user, and assistant separately. Check internal consistency and realism of "
        "the buyer request; package-versus-product field misuse; relevant mandatory requirements; "
        "variant contamination; catalog-claim attribution; grounding of every material statement; "
        "decision precedence; label correctness; concision; and absence of leakage or invented facts. "
        "The hidden target is an authoring goal, not proof that the label is correct. Reject it if the "
        "evidence does not support that target. Score strictly. APPROVE only when the conversation can "
        "enter SFT unchanged. Return JSON only with exactly this shape: "
        '{"verdict":"APPROVE|REVISE|REJECT","system_score":1,"user_score":1,'
        '"assistant_score":1,"decision_correct":true,"grounded":true,"sft_ready":true,'
        '"issues":["string"],"recommended_changes":["string"]}.\n\n'
        f"Hidden target result:\n{target}\n\nDataset rules:\n{rules}\n\n"
        f"Normalized original product data:\n{json.dumps(product, ensure_ascii=False)}\n\n"
        f"Original review context:\n{json.dumps({k:v for k,v in reviews.items() if k != 'source_review_ids'}, ensure_ascii=False)}\n\n"
        f"Conversation:\n{json.dumps(messages, ensure_ascii=False)}"
    )


def passes_gate(judgment: dict, target: str, answer: dict) -> bool:
    return all((
        judgment["verdict"] == "APPROVE",
        judgment["sft_ready"], judgment["grounded"], judgment["decision_correct"],
        judgment["system_score"] >= 4,
        judgment["user_score"] == 5,
        judgment["assistant_score"] >= 4,
        answer["result"] == target,
        not judgment["issues"],
    ))


def process_one(
    index: int, parent: str, target: str, product_raw: dict, all_parent_reviews: list[dict],
    args: argparse.Namespace, env: dict[str, str], rules: str,
) -> dict:
    variant_asin, variant_reviews = dominant_variant(all_parent_reviews, args.min_variant_reviews)
    selected = select_evidence(variant_reviews, args.max_reviews, args.max_evidence_chars)
    if len(selected) < 6:
        raise RuntimeError("fewer than six informative English reviews")
    product = normalized_product(product_raw)
    reviews = review_context(variant_reviews, selected, variant_asin)
    feedback: list[str] = []
    history: list[dict] = []
    final: dict | None = None
    for attempt in range(1, args.attempts + 1):
        request, user_usage = call_json(
            env, args.user_model,
            "You create realistic, auditable technical procurement requests. Return valid JSON only.",
            user_author_prompt(target, product, reviews, feedback), 1300, args.retries, 0.2,
        )
        validate_user_result(request, target)
        user_message = format_user(request, product, reviews)
        answer, assistant_usage = call_json(
            env, args.assistant_model,
            "You are a rigorous purchasing copilot and SFT response author. Return valid JSON only.",
            assistant_prompt(user_message, rules, feedback), 1400, args.retries, 0.1,
        )
        validate_answer(answer, len(reviews["selected_reviews"]))
        assistant_message = format_answer(answer)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        judgment, judge_usage = call_json(
            env, args.judge_model,
            "You are an independent, strict SFT dataset auditor. Return valid JSON only.",
            judge_prompt(target, product, reviews, messages, rules), 1500, args.retries, None,
        )
        validate_judgment(judgment)
        passed = passes_gate(judgment, target, answer)
        history.append({
            "attempt": attempt, "request": request, "answer": answer, "judgment": judgment,
            "usage": {"user": user_usage, "assistant": assistant_usage, "judge": judge_usage},
            "passed": passed,
        })
        final = {"messages": messages, "request": request, "answer": answer, "judgment": judgment}
        if passed:
            break
        feedback = judgment["issues"] + judgment["recommended_changes"]
    assert final is not None
    status = "approved" if history[-1]["passed"] else (
        "rejected" if final["judgment"]["verdict"] == "REJECT" else "pending_human_review"
    )
    example_id = hashlib.sha256(f"{parent}:{target}:{PIPELINE_VERSION}".encode()).hexdigest()
    example = {
        "messages": final["messages"],
        "metadata": {
            "example_id": example_id, "parent_asin": parent, "variant_asin": variant_asin,
            "target_result": target, "actual_result": final["answer"]["result"],
            "source_review_ids": reviews["source_review_ids"],
            "source_reviews_available": len(variant_reviews),
            "source_reviews_selected": len(selected),
            "review_status": status, "language": "en", "pipeline_version": PIPELINE_VERSION,
            "models": {"user": args.user_model, "assistant": args.assistant_model, "judge": args.judge_model},
            "attempts": len(history), "judge": final["judgment"],
        },
    }
    return {"index": index, "example": example, "history": history, "normalized_product": product,
            "review_context": {key: value for key, value in reviews.items() if key != "source_review_ids"}}


def process_slot(
    index: int, target: str, parents: list[str], metadata: dict[str, dict],
    all_reviews: dict[str, list[dict]], args: argparse.Namespace,
    env: dict[str, str], rules: str,
) -> dict:
    failures: list[dict] = []
    for parent in parents:
        try:
            row = process_one(
                index, parent, target, metadata[parent], all_reviews[parent], args, env, rules,
            )
            row["candidate_failures"] = failures
            return row
        except Exception as error:
            failures.append({"parent_asin": parent, "error": str(error)})
    raise RuntimeError("all candidate attempts failed: " + json.dumps(failures, ensure_ascii=False))


def write_review(path: Path, rows: list[dict]) -> None:
    lines = ["# Validated three-model SFT pilot", ""]
    for row in rows:
        example = row["example"]
        meta = example["metadata"]
        judge = meta["judge"]
        lines.extend([
            f"## {row['index']}. {meta['parent_asin']} — {meta['target_result']}", "",
            f"- Status: `{meta['review_status']}`",
            f"- Actual result: `{meta['actual_result']}`",
            f"- Attempts: {meta['attempts']}",
            f"- Scores: system {judge['system_score']}/5, user {judge['user_score']}/5, assistant {judge['assistant_score']}/5",
            f"- Judge verdict: `{judge['verdict']}`", "",
            "### User", "", "```text", example["messages"][1]["content"], "```", "",
            "### Assistant", "", example["messages"][2]["content"], "",
            "### Judge issues", "",
        ])
        lines.extend(f"- {issue}" for issue in judge["issues"] or ["None"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.examples < 1:
        raise ValueError("--examples must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "master": output / "examples_master.jsonl",
        "approved": output / "training.jsonl",
        "pending": output / "pending_human_review.jsonl",
        "rejected": output / "rejected.jsonl",
        "runs": output / "generation_runs.jsonl",
        "review": output / "REVIEW.md",
        "manifest": output / "manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    env = load_env()
    rules = RULES_FILE.read_text(encoding="utf-8").strip()
    metadata, counts = load_products(args.data_dir)
    # Keep spare candidates so an ineligible dominant variant does not abort the run.
    candidates = candidate_products(metadata, counts, args.examples * 20)
    all_reviews = load_candidate_reviews(args.data_dir, set(candidates))
    targets = [args.target_result] * args.examples if args.target_result else [
        RESULTS[index % len(RESULTS)] for index in range(args.examples)
    ]
    assigned, spares = assign_candidates(candidates, targets, metadata, all_reviews)
    work = []
    for offset, (parent, target) in enumerate(zip(assigned, targets)):
        replacements = spares[offset * 2:(offset + 1) * 2]
        work.append((offset + 1, [parent, *replacements], target))
    rows: list[dict] = []
    errors: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_slot, index, target, parents, metadata, all_reviews, args, env, rules,
            ): (index, parents[0], target)
            for index, parents, target in work
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index, parent, target = futures[future]
            try:
                row = future.result()
                rows.append(row)
                with LOCK:
                    with paths["runs"].open("a", encoding="utf-8") as destination:
                        destination.write(json.dumps({"status": "ok", **row}, ensure_ascii=False) + "\n")
            except Exception as error:
                failure = {"status": "error", "index": index, "parent_asin": parent,
                           "target_result": target, "error": str(error)}
                errors.append(failure)
                with LOCK:
                    with paths["runs"].open("a", encoding="utf-8") as destination:
                        destination.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"progress={completed}/{len(work)} completed={len(rows)} errors={len(errors)}", flush=True)
    rows.sort(key=lambda row: row["index"])
    with paths["master"].open("w", encoding="utf-8") as master, \
         paths["approved"].open("w", encoding="utf-8") as approved, \
         paths["pending"].open("w", encoding="utf-8") as pending, \
         paths["rejected"].open("w", encoding="utf-8") as rejected:
        for row in rows:
            example = row["example"]
            master.write(json.dumps(example, ensure_ascii=False) + "\n")
            status = example["metadata"]["review_status"]
            if status == "approved":
                approved.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + "\n")
            elif status == "rejected":
                rejected.write(json.dumps(example, ensure_ascii=False) + "\n")
            else:
                pending.write(json.dumps(example, ensure_ascii=False) + "\n")
    write_review(paths["review"], rows)
    statuses = Counter(row["example"]["metadata"]["review_status"] for row in rows)
    actual = Counter(row["example"]["metadata"]["actual_result"] for row in rows)
    target_counts = Counter(row["example"]["metadata"]["target_result"] for row in rows)
    judge_scores = {
        key: round(sum(row["example"]["metadata"]["judge"][key] for row in rows) / len(rows), 2)
        if rows else 0
        for key in ("system_score", "user_score", "assistant_score")
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "models": {"user": args.user_model, "assistant": args.assistant_model, "judge": args.judge_model},
        "requested_examples": args.examples,
        "completed_examples": len(rows), "errors": errors,
        "target_results": dict(sorted(target_counts.items())),
        "actual_results": dict(sorted(actual.items())),
        "statuses": dict(sorted(statuses.items())), "average_judge_scores": judge_scores,
        "gate": {"system_min": 4, "user_required": 5, "assistant_min": 4,
                 "decision_correct": True, "grounded": True, "issues_required": 0},
        "source": "McAuley-Lab/Amazon-Reviews-2023 / Industrial_and_Scientific",
        "files": {key: path.name for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if len(rows) == args.examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
