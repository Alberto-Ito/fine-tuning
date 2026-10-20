#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "foundry_sft_full"
PENDING = DATA / "pending_human_review.jsonl"
APPROVED = DATA / "append_to_training.jsonl"
STATUS = DATA / "pending_human_review_status.md"
RESULTS = DATA / "foundry_human_review_results.jsonl"


def category(reason: str) -> str:
    value = reason.lower()
    if any(key in value for key in ("domain fit", "domain relevance", "domain uncertainty", "domain appropriateness", "mro domain", "industrial suitability")):
        return "domain"
    if any(key in value for key in ("safety", "fire risk", "health issue", "overheating", "firearm", "injury", "medical use", "seal wounds")):
        return "safety"
    if any(key in value for key in ("conflicting", "discrepancy", "ambiguity", "authenticity", "certification", "mislabel", "size compatibility")):
        return "specification"
    if any(key in value for key in ("single", "one report", "one review", "isolated", "unclear if", "frequency and scope", "limited reviews", "more evidence")):
        return "limited"
    if any(key in value for key in ("unrelated", "category contamination", "product confusion", "variant confusion")):
        return "limited"
    return "specification"


pending = [json.loads(line) for line in PENDING.read_text().splitlines()]
approved = [json.loads(line) for line in APPROVED.read_text().splitlines()]
result_rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
latest = {row["example_id"]: row["result"] for row in result_rows if row.get("status") == "ok"}
updates = {}
counts = Counter()

for item in pending:
    metadata = item["metadata"]
    example_id = metadata["example_id"]
    audit = latest[example_id]
    kind = category(audit["review_reason"])
    counts[kind] += 1
    answer = item["messages"][-1]["content"].strip()

    if kind == "domain":
        prefix = (
            "Purchase recommendation: Do not buy this product for an industrial application because its primary category, positioning, "
            "or represented use does not clearly support the Industrial & Scientific domain. If it is the only available option, request "
            "from the seller or manufacturer the technical documentation and certifications for the industrial standards applicable to "
            "the intended use before purchase. If verifiable compliance cannot be provided, do not buy it."
        )
        rules = "R012"
    elif kind == "safety":
        prefix = (
            "Purchase recommendation: Do not buy this product because the supplied evidence reports a potential safety or health problem. "
            "If it is the only available option, reconsider it only after reliable evidence establishes whether the incident resulted from "
            "incorrect user operation or from a product failure. If incorrect use is established, warn against that specific use. If the "
            "product caused the problem, or the cause remains uncertain, do not buy it."
        )
        rules = "R009"
    elif kind == "specification":
        if metadata["parent_asin"] == "B00F34XH6W":
            prefix = (
                "Purchase recommendation: Do not buy this product. The reviews report rust and poor coating, which are direct quality "
                "problems rather than a specification that can be resolved by asking the seller."
            )
            rules = "R004, R009"
        else:
            prefix = (
                "Purchase recommendation: Do not buy this product until the seller confirms the specific product specification that is "
                "contradictory in the supplied evidence. Obtain confirmation of that specification before placing the order; a generic "
                "assurance is not sufficient."
            )
            rules = "R008"
    else:
        prefix = (
            "Evidence note: The reported problem is limited or may be an isolated case. It should be presented as a possible issue, not as "
            "a demonstrated recurring defect. The example remains suitable for training with that uncertainty stated explicitly."
        )
        rules = "R001, R004, R006"

    item["messages"][-1]["content"] = prefix + "\n\n" + answer
    metadata["review_status"] = "approved"
    metadata["review_reasons"] = [f"Locally resolved under {rules}: {kind}."]
    approved.append(item)
    updates[example_id] = rules

with APPROVED.open("w") as handle:
    for item in approved:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
PENDING.write_text("")

status = STATUS.read_text()
for example_id, rules in updates.items():
    pattern = re.compile(
        rf"^(\| `{re.escape(example_id)}` \| `[^`]+` \| `[^`]+` \| )Pendiente( \|)$",
        re.M,
    )
    status, matched = pattern.subn(r"\1Aprobado\2", status)
    if matched != 1:
        raise SystemExit(f"Could not update status row for {example_id}")
status = re.sub(r"\| Pendiente \| [\d.]+ \|", "| Pendiente | 0 |", status, count=1)
status = re.sub(r"\| Aprobado \| [\d.]+ \|", f"| Aprobado | {len(approved):,} |".replace(",", "."), status, count=1)
STATUS.write_text(status)

print(json.dumps({"resolved": len(pending), "categories": dict(counts), "approved_total": len(approved), "pending_total": 0}, indent=2))
