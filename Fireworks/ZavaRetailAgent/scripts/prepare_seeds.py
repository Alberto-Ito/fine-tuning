from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    CONFIG,
    INTENT_TOOL_SEQUENCES,
    REFERENCE,
    ROOT,
    SEEDS,
    ensure_directories,
    load_iteration_config,
    read_json,
    write_jsonl,
)


def compact_context(
    db: dict[str, Any], user_id: str, intent: str, rng: random.Random
) -> dict[str, Any]:
    user = db["users"][user_id]
    order_ids = list(user.get("orders", []))
    orders = [
        db["orders"][order_id] for order_id in order_ids if order_id in db["orders"]
    ]

    if intent == "modify_pending_order_payment":
        matching = [
            order
            for order in orders
            if order.get("status") == "pending"
            and valid_alternative_payment_methods(user, order)
        ]
    elif intent in {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
    }:
        matching = [order for order in orders if order.get("status") == "pending"]
    elif intent in {"return_delivered_order", "exchange_delivered_order_items"}:
        matching = [order for order in orders if order.get("status") == "delivered"]
    else:
        matching = orders

    selected_order = (
        rng.choice(matching) if matching else (rng.choice(orders) if orders else None)
    )
    products: dict[str, Any] = {}
    if selected_order:
        for item in selected_order.get("items", []):
            product_id = item.get("product_id")
            if product_id in db["products"]:
                products[product_id] = db["products"][product_id]

    other_user_id = next(
        (candidate for candidate in db["users"] if candidate != user_id), None
    )
    context = {
        "authenticated_user_id": user_id,
        "user": user,
        "selected_order": selected_order,
        "related_products": products,
    }
    if intent == "policy_refusal_other_user" and other_user_id:
        other = db["users"][other_user_id]
        context["other_user_request"] = {
            "user_id": other_user_id,
            "name": other.get("name"),
            "email": other.get("email"),
        }
    return context


def valid_alternative_payment_methods(
    user: dict[str, Any], order: dict[str, Any]
) -> list[str]:
    original_ids = {
        entry.get("payment_method_id")
        for entry in order.get("payment_history", [])
        if entry.get("transaction_type") == "payment"
    }
    order_total = sum(item.get("price", 0) for item in order.get("items", []))
    alternatives: list[str] = []
    for payment_method_id, payment_method in user.get("payment_methods", {}).items():
        if payment_method_id in original_ids:
            continue
        if (
            payment_method.get("source") == "gift_card"
            and payment_method.get("balance", 0) < order_total
        ):
            continue
        alternatives.append(payment_method_id)
    return alternatives


def eligible_users(db: dict[str, Any], intent: str) -> list[str]:
    eligible: list[str] = []
    for user_id, user in db["users"].items():
        orders = [db["orders"].get(order_id, {}) for order_id in user.get("orders", [])]
        statuses = {order.get("status") for order in orders}
        if intent == "modify_pending_order_payment" and not any(
            order.get("status") == "pending"
            and valid_alternative_payment_methods(user, order)
            for order in orders
        ):
            continue
        if (
            intent
            in {
                "cancel_pending_order",
                "modify_pending_order_address",
                "modify_pending_order_items",
            }
            and "pending" not in statuses
        ):
            continue
        if (
            intent in {"return_delivered_order", "exchange_delivered_order_items"}
            and "delivered" not in statuses
        ):
            continue
        eligible.append(user_id)
    return eligible


def build_seeds(
    count: int, seed: int, intents: list[str], language: str, iteration: str
) -> list[dict[str, Any]]:
    db = read_json(REFERENCE / "db.json")
    rng = random.Random(seed)
    pools = {intent: eligible_users(db, intent) for intent in intents}
    records: list[dict[str, Any]] = []
    intent_counts: defaultdict[str, int] = defaultdict(int)

    for index in range(count):
        intent = intents[index % len(intents)]
        pool = pools[intent]
        if not pool:
            raise RuntimeError(f"No eligible users found for intent {intent}")
        user_id = rng.choice(pool)
        intent_counts[intent] += 1
        context = compact_context(db, user_id, intent, rng)
        records.append(
            {
                "seed_id": f"iter{iteration}-{index + 1:05d}",
                "intent": intent,
                "variation": intent_counts[intent],
                "language": language,
                "required_tool_sequence_json": json.dumps(
                    INTENT_TOOL_SEQUENCES[intent], separators=(",", ":")
                ),
                "context_json": json.dumps(
                    context, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    rng.shuffle(records)
    return records


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(CONFIG / "iteration_01.json"))
    known_args, _ = config_parser.parse_known_args()
    config = load_iteration_config(Path(known_args.config))
    parser = argparse.ArgumentParser(
        description="Build grounded scenario seeds from the Zava database.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--count", type=int, default=config["target_generated_examples"]
    )
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument(
        "--output", default=str(SEEDS / f"{config['_iteration_name']}.jsonl")
    )
    args = parser.parse_args()
    ensure_directories()
    records = build_seeds(
        args.count,
        args.seed,
        config["intents"],
        config["language"],
        config["_iteration_id"],
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    written = write_jsonl(output_path, records)
    print(f"Wrote {written} grounded scenario seeds to {args.output}")


if __name__ == "__main__":
    main()
