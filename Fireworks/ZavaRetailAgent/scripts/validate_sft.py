from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    CONFIG,
    CONSEQUENTIAL_TOOLS,
    FINAL,
    GENERATED,
    INTENT_TOOL_SEQUENCES,
    REFERENCE,
    REPORTS,
    SOURCE,
    ensure_directories,
    load_iteration_config,
    read_json,
    read_jsonl,
    write_jsonl,
)


def signature(record: dict[str, Any]) -> str:
    normalized = []
    for message in record.get("messages", []):
        normalized.append(
            {
                "role": message.get("role"),
                "content": " ".join((message.get("content") or "").lower().split()),
                "tool_calls": message.get("tool_calls"),
            }
        )
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_record(
    record: dict[str, Any], tool_names: set[str], db: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return ["messages must be a list with at least 3 entries"]
    if messages[0].get("role") != "system":
        errors.append("first message is not system")

    outstanding: set[str] = set()
    known_ids = set(db["users"]) | set(db["orders"]) | set(db["products"])
    for product in db["products"].values():
        known_ids.update(product.get("variants", {}))
    for user in db["users"].values():
        known_ids.update(user.get("payment_methods", {}))

    for index, message in enumerate(messages):
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            errors.append(f"message {index}: tool_calls on non-assistant role")
        if len(calls) > 1:
            errors.append(f"message {index}: more than one tool call")
        if calls and message.get("content") not in (None, ""):
            errors.append(f"message {index}: assistant mixes content and tool call")
        for call in calls:
            call_id = call.get("id")
            function = call.get("function", {})
            if not call_id:
                errors.append(f"message {index}: missing tool call id")
            else:
                outstanding.add(call_id)
            if function.get("name") not in tool_names:
                errors.append(f"message {index}: unknown tool {function.get('name')!r}")
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                errors.append(f"message {index}: tool arguments are not JSON")
                continue
            for key, value in arguments.items():
                if (
                    key.endswith("_id")
                    and isinstance(value, str)
                    and value not in known_ids
                ):
                    errors.append(f"message {index}: ungrounded {key}={value}")
                if key.endswith("_ids") and isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item not in known_ids:
                            errors.append(
                                f"message {index}: ungrounded {key} item={item}"
                            )
        if role == "tool":
            if not isinstance(message.get("content"), str) or not message["content"]:
                errors.append(f"message {index}: tool result content is missing")
            tool_call_id = message.get("tool_call_id")
            if tool_call_id not in outstanding:
                errors.append(f"message {index}: unmatched tool_call_id")
            else:
                outstanding.remove(tool_call_id)
    final_calls = {
        call.get("id")
        for call in (messages[-1].get("tool_calls") or [])
        if messages[-1].get("role") == "assistant" and call.get("id")
    }
    # The upstream Zava SFT records intentionally stop after the final action call;
    # the execution result is not part of the supervised target.
    unexpected_outstanding = outstanding - final_calls
    if unexpected_outstanding:
        errors.append(f"missing tool results for {sorted(unexpected_outstanding)}")
    return errors


def validate_intent_workflow(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intent = record.get("metadata", {}).get("intent")
    required = INTENT_TOOL_SEQUENCES.get(intent)
    if required is None:
        return [f"unknown intent workflow: {intent!r}"]

    messages = record.get("messages", [])
    calls: list[tuple[int, str]] = []
    for message_index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            calls.append((message_index, call.get("function", {}).get("name", "")))
    call_names = [name for _, name in calls]

    cursor = 0
    for required_name in required:
        try:
            cursor = call_names.index(required_name, cursor) + 1
        except ValueError:
            errors.append(
                f"intent {intent}: missing or out-of-order required tool {required_name}; "
                f"observed sequence={call_names}"
            )
            break

    if call_names and call_names[0] not in {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
    }:
        errors.append(f"intent {intent}: first tool does not authenticate the user")

    confirmation_terms = {
        "yes",
        "confirm",
        "confirmed",
        "proceed",
        "go ahead",
        "do it",
        "correct",
    }
    for message_index, tool_name in calls:
        if tool_name not in CONSEQUENTIAL_TOOLS:
            continue
        previous_user = next(
            (
                message.get("content", "").lower()
                for message in reversed(messages[:message_index])
                if message.get("role") == "user"
            ),
            "",
        )
        if not any(term in previous_user for term in confirmation_terms):
            errors.append(
                f"message {message_index}: consequential tool {tool_name} lacks explicit user confirmation"
            )
    return errors


def validate_action_semantics(record: dict[str, Any], db: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intent = record.get("metadata", {}).get("intent")
    calls = [
        call
        for message in record.get("messages", [])
        for call in message.get("tool_calls") or []
    ]
    action_call = next(
        (
            call
            for call in reversed(calls)
            if call.get("function", {}).get("name") in CONSEQUENTIAL_TOOLS
        ),
        None,
    )
    if action_call is None:
        return errors

    function = action_call.get("function", {})
    try:
        arguments = json.loads(function.get("arguments", "{}"))
    except json.JSONDecodeError:
        return errors

    order_id = arguments.get("order_id")
    order = db["orders"].get(order_id) if isinstance(order_id, str) else None
    if order is not None and intent in {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    } and order.get("status") != "pending":
        errors.append(
            f"intent {intent}: action targets non-pending order {order_id}"
        )

    if function.get("name") == "modify_pending_order_payment" and order is not None:
        payment_method_id = arguments.get("payment_method_id")
        original_ids = {
            entry.get("payment_method_id")
            for entry in order.get("payment_history", [])
            if entry.get("transaction_type") == "payment"
        }
        if payment_method_id in original_ids:
            errors.append(
                "modify_pending_order_payment reuses the original payment method"
            )
        user = db["users"].get(order.get("user_id"), {})
        payment_method = user.get("payment_methods", {}).get(payment_method_id)
        if payment_method is None:
            errors.append(
                "modify_pending_order_payment uses a method not owned by the order user"
            )
        elif payment_method.get("source") == "gift_card":
            order_total = sum(
                item.get("price", 0) for item in order.get("items", [])
            )
            if payment_method.get("balance", 0) < order_total:
                errors.append(
                    "modify_pending_order_payment uses a gift card with insufficient balance"
                )
    return errors


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(CONFIG / "iteration_01.json"))
    known_args, _ = config_parser.parse_known_args()
    config = load_iteration_config(Path(known_args.config))
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, split, and merge generated Zava SFT data.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--input", default=str(GENERATED / f"{config['_iteration_name']}_raw.jsonl")
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=config["validation_fraction"]
    )
    parser.add_argument("--seed", type=int, default=config["seed"])
    args = parser.parse_args()
    ensure_directories()

    tools = read_json(REFERENCE / "retail_tools.json")
    tool_names = {tool["function"]["name"] for tool in tools}
    db = read_json(REFERENCE / "db.json")
    originals = read_jsonl(SOURCE / "sft_train.jsonl")
    generated = read_jsonl(Path(args.input))
    seen = {signature(record) for record in originals}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    intent_counts: Counter[str] = Counter()

    for record in generated:
        errors = validate_record(record, tool_names, db)
        errors.extend(validate_intent_workflow(record))
        errors.extend(validate_action_semantics(record, db))
        sig = signature(record)
        if sig in seen:
            errors.append("exact normalized duplicate")
        if errors:
            rejected.append({"errors": errors, "record": record})
            continue
        seen.add(sig)
        intent_counts[record.get("metadata", {}).get("intent", "unknown")] += 1
        accepted.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    validation_count = round(len(accepted) * args.validation_fraction)
    if args.validation_fraction > 0 and len(accepted) > 1:
        validation_count = min(len(accepted) - 1, max(1, validation_count))
    generated_validation = accepted[:validation_count]
    generated_train = accepted[validation_count:]

    # Metadata is useful for QA but is removed from the fine-tuning payload.
    clean_train = [
        {key: value for key, value in record.items() if key != "metadata"}
        for record in generated_train
    ]
    clean_validation = [
        {key: value for key, value in record.items() if key != "metadata"}
        for record in generated_validation
    ]
    combined = [*originals, *clean_train]

    output_prefix = config["_iteration_name"]
    write_jsonl(FINAL / f"{output_prefix}_sft_generated_train.jsonl", clean_train)
    write_jsonl(
        FINAL / f"{output_prefix}_sft_generated_validation.jsonl", clean_validation
    )
    write_jsonl(FINAL / f"{output_prefix}_sft_train_extended.jsonl", combined)
    rejected_path = REPORTS / f"{config['_iteration_name']}_rejected.jsonl"
    validation_report_path = REPORTS / f"{config['_iteration_name']}_validation.json"
    write_jsonl(rejected_path, rejected)
    report = {
        "original_train_examples": len(originals),
        "generated_input_examples": len(generated),
        "generated_accepted": len(accepted),
        "generated_rejected": len(rejected),
        "generated_train": len(clean_train),
        "generated_validation": len(clean_validation),
        "combined_train_examples": len(combined),
        "accepted_by_intent": dict(sorted(intent_counts.items())),
        "official_test_set_modified": False,
    }
    validation_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if rejected:
        raise SystemExit(
            f"Some generated records were rejected; inspect {rejected_path}"
        )


if __name__ == "__main__":
    main()
