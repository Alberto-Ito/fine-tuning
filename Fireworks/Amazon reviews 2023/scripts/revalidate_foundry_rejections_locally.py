#!/usr/bin/env python3
import json
import re
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "foundry_sft_full"
TRAINING = DATA / "append_to_training.jsonl"
FOUNDRY_RESULTS = DATA / "foundry_training_validation_results.jsonl"
OUTPUT = DATA / "local_revalidation_of_foundry_rejections.jsonl"
REJECTED = DATA / "rejected.jsonl"
STATUS = DATA / "pending_human_review_status.md"

# Products whose purpose/evidence is intrinsically outside the MRO or industrial-
# support dataset. This list was produced by local human-style inspection, not by
# another model call. Product failures and no-buy recommendations are excluded.
OFF_DOMAIN_ASINS = {
    "B01824W4Y0",  # consumer audio DAC
    "B01N6V7L5K",  # beauty/personal-care dispenser
    "B079757WYT",  # dowsing/divination set
    "B08LMHZ4K1",  # iridology/health device
    "B07NDPN9W5",  # peroxide marketed for personal-health cure
    "B07VRQWJV6",  # cosmetic PRP treatment tubes
    "B01FG74GPG",  # food/beverage flavor essence
    "B078WXZVXP",  # dowsing/reiki rods
    "B00ZWUX302",  # consumer shopping-cart liner
    "B01J480GXY",  # educational mineral specimen
    "B004Y4ZB2K",  # magnetic-therapy belt
    "B077T53GKK",  # consumer real-estate sign
}


def product_name(item):
    match = re.search(r'\\?"name\\?": \\?"([^"\\]+)', item["messages"][1]["content"])
    return match.group(1) if match else ""


parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

training = [json.loads(line) for line in TRAINING.read_text().splitlines()]
rows = [json.loads(line) for line in FOUNDRY_RESULTS.read_text().splitlines()]
latest_ok = {row["example_id"]: row["result"] for row in rows if row.get("status") == "ok"}

output = []
for item in training:
    example_id = item["metadata"]["example_id"]
    audit = latest_ok[example_id]
    if audit["recommended_status"] != "rejected":
        continue
    asin = item["metadata"]["parent_asin"]
    if asin in OFF_DOMAIN_ASINS:
        decision = "rejected"
        reason = (
            "The product or the use represented by the evidence is outside the target "
            "MRO/industrial-support domain, so the example is not suitable for training (R002)."
        )
        rules = ["R002"]
    else:
        decision = "approved"
        reason = (
            "The example is suitable for training. Product failures, safety concerns, objective "
            "discrepancies, excessive price, quantity problems, or a recommendation not to buy "
            "are useful negative-purchase training content and are not grounds for dataset rejection. "
            "The corrected answer should be used when the existing answer is inaccurate."
        )
        rules = audit["applicable_rules"]
    output.append({
        "example_id": example_id,
        "parent_asin": asin,
        "product_name": product_name(item),
        "foundry_recommendation": "rejected",
        "local_decision": decision,
        "applicable_rules": rules,
        "local_reason": reason,
        "corrected_answer": audit["corrected_answer"],
    })

with OUTPUT.open("w") as handle:
    for row in output:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

print(json.dumps({
    "reviewed": len(output),
    "decisions": dict(Counter(row["local_decision"] for row in output)),
    "rejected_asins": len({row["parent_asin"] for row in output if row["local_decision"] == "rejected"}),
    "output": str(OUTPUT),
}, ensure_ascii=False, indent=2))

if args.apply:
    reject_ids = {row["example_id"] for row in output if row["local_decision"] == "rejected"}
    existing_rejected = [json.loads(line) for line in REJECTED.read_text().splitlines()]
    existing_ids = {item["metadata"]["example_id"] for item in existing_rejected}
    if reject_ids & existing_ids:
        raise SystemExit("Refusing to apply: at least one target ID is already in rejected.jsonl")
    moved, kept = [], []
    for item in training:
        example_id = item["metadata"]["example_id"]
        if example_id in reject_ids:
            item["metadata"]["review_status"] = "rejected"
            item["metadata"]["review_reasons"] = [
                "R002: product or represented use is outside the target MRO/industrial-support domain."
            ]
            moved.append(item)
        else:
            kept.append(item)
    if len(moved) != len(reject_ids):
        raise SystemExit(f"Refusing to apply: found {len(moved)}/{len(reject_ids)} target IDs")

    with TRAINING.open("w") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with REJECTED.open("a") as handle:
        for item in moved:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    status = STATUS.read_text()
    for example_id in reject_ids:
        pattern = re.compile(
            rf"^(\| `{re.escape(example_id)}` \| `[^`]+` \| `[^`]+` \| )Aprobado( \|)$",
            re.M,
        )
        status, count = pattern.subn(r"\1Rechazado\2", status)
        if count != 1:
            raise SystemExit(f"Could not update status table for {example_id}")
    approved_total = len(kept)
    rejected_total = len(existing_rejected) + len(moved)
    status = re.sub(r"\| Aprobado \| [\d.]+ \|", f"| Aprobado | {approved_total:,} |".replace(",", "."), status, count=1)
    status = re.sub(r"\| Rechazado \| [\d.]+ \|", f"| Rechazado | {rejected_total:,} |".replace(",", "."), status, count=1)
    STATUS.write_text(status)
    print(json.dumps({"moved": len(moved), "approved_total": approved_total, "rejected_total": rejected_total}))
