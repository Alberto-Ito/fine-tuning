#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "foundry_sft_full"
TRAINING = DATA / "append_to_training.jsonl"

NO_BUY_ASINS = {
    "B000FMYQGW", "B00E3UPN6Y", "B07NSL4CCY", "B081KTXBT6",
    "B01AD9P0CA", "B06XPDRGFB", "B07VT891P9", "B07Z7J6LXD",
    "B08L3G6PQH", "B07K7Q84FJ", "B0050K3FVQ", "B08PL179H6",
}
CAPACITY_WARNING_ASIN = "B07Y21R2KS"

NO_BUY_TEXT = (
    "Purchase recommendation: Do not buy this product based on the supplied review evidence. "
    "The reported problems are sufficiently serious or recurrent that a more reliable alternative should be preferred. "
    "This recommendation concerns the product purchase; the example remains suitable for training."
)
CAPACITY_WARNING = (
    "Purchase recommendation: Do not buy this product, including as a fallback option. The jack is advertised with a 20-ton lifting "
    "capacity, but user reports indicate failure even below 10 tons. This objective discrepancy leaves no review-supported safe working "
    "capacity, and failure during lifting can create a serious safety hazard. Choose lifting equipment with a verified capacity instead."
)
OLD_CAPACITY_WARNING = (
    "Safety-critical capacity warning: Do not rely on the advertised 20-ton lifting capacity. "
    "User reports indicate that the jack may fail to lift even loads below 10 tons. Because an unverified lifting limit can cause "
    "injury or property damage, do not use this jack for safety-critical lifting, do not work beneath a supported load, and prefer "
    "properly rated and independently certified lifting equipment."
)

items = [json.loads(line) for line in TRAINING.read_text().splitlines()]
changed = []
for item in items:
    asin = item["metadata"]["parent_asin"]
    answer = item["messages"][-1]["content"].strip()
    if asin in NO_BUY_ASINS:
        if not answer.startswith(NO_BUY_TEXT):
            item["messages"][-1]["content"] = NO_BUY_TEXT + "\n\n" + answer
            changed.append((item["metadata"]["example_id"], asin, "no_buy"))
    elif asin == CAPACITY_WARNING_ASIN:
        if answer.startswith(OLD_CAPACITY_WARNING + "\n\n"):
            answer = answer[len(OLD_CAPACITY_WARNING) + 2:]
        previous_warning = (
            "Safety-critical capacity warning: Do not rely on the advertised 20-ton lifting capacity. "
            "User reports indicate that the jack may fail to lift even loads below 10 tons. Because an unverified lifting limit can cause "
            "injury or property damage, do not use this jack for safety-critical lifting and prefer lifting equipment with a verified capacity."
        )
        if answer.startswith(previous_warning + "\n\n"):
            answer = answer[len(previous_warning) + 2:]
        task = item["metadata"]["task"]
        task_text = {
            "pros_and_cons": (
                "Catalog advantages include a low-profile design and a nominal 20-ton rating. However, reviews report that the jack may "
                "fail to lift even loads below 10 tons and may arrive without essential handles or hoses. Because the evidence does not "
                "establish any safe working capacity, the catalog advantages do not support purchase or use."
            ),
            "use_cases": (
                "The catalog describes lifting, jacking, and spreading applications, but the reviews do not establish a safe usable load. "
                "A reported failure below 10 tons contradicts the advertised 20-ton capacity. Therefore, no lifting use case can be "
                "recommended from this evidence, including light-duty or fallback use."
            ),
            "common_problems": (
                "The central problem is an objective and safety-critical capacity discrepancy: the product is advertised for 20 tons, "
                "while reviews report failure below 10 tons. Reviews also report missing handles and hoses. The unknown safe working limit "
                "makes the jack unsuitable for reliance in lifting operations."
            ),
            "purchase_recommendation": (
                "Do not purchase this jack. Do not treat it as a fallback or assume that using a lighter load makes it safe, because the "
                "reviews do not establish a verified working capacity. The reported failure below 10 tons materially contradicts the "
                "advertised 20-ton rating."
            ),
        }.get(task, "The reviews do not establish a safe working capacity for this jack.")
        item["messages"][-1]["content"] = CAPACITY_WARNING + "\n\n" + task_text
        changed.append((item["metadata"]["example_id"], asin, "capacity_warning"))

with TRAINING.open("w") as handle:
    for item in items:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

print(json.dumps({
    "changed": len(changed),
    "no_buy": sum(kind == "no_buy" for _, _, kind in changed),
    "capacity_warning": sum(kind == "capacity_warning" for _, _, kind in changed),
    "asins": sorted({asin for _, asin, _ in changed}),
}, ensure_ascii=False, indent=2))
