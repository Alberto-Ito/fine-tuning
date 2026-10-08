from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
REFERENCE = ROOT / "reference"
SOURCE = ROOT / "data" / "source"
SEEDS = ROOT / "data" / "seeds"
GENERATED = ROOT / "data" / "generated"
FINAL = ROOT / "data" / "final"
REPORTS = ROOT / "reports"

INTENT_TOOL_SEQUENCES: dict[str, list[str]] = {
    "order_status": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
    ],
    "cancel_pending_order": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "cancel_pending_order",
    ],
    "modify_pending_order_address": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "modify_pending_order_address",
    ],
    "modify_pending_order_payment": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "modify_pending_order_payment",
    ],
    "modify_pending_order_items": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "modify_pending_order_items",
    ],
    "return_delivered_order": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "return_delivered_order_items",
    ],
    "exchange_delivered_order_items": [
        "find_user_id_by_email",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "exchange_delivered_order_items",
    ],
    "update_profile_address": [
        "find_user_id_by_email",
        "get_user_details",
        "modify_user_address",
    ],
    "product_information": [
        "find_user_id_by_email",
        "get_user_details",
        "get_product_details",
    ],
    "policy_refusal_other_user": [
        "find_user_id_by_email",
        "get_user_details",
    ],
}

CONSEQUENTIAL_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def ensure_directories() -> None:
    for path in (SEEDS, GENERATED, FINAL, REPORTS, ROOT / "artifacts"):
        path.mkdir(parents=True, exist_ok=True)


def load_environment() -> None:
    load_dotenv(ROOT / ".env")


def require_environment(*names: str) -> dict[str, str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill it in."
        )
    return {name: os.environ[name] for name in names}


def normalize_foundry_base_url(value: str) -> str:
    """Return the OpenAI-compatible API root, not a concrete operation URL."""
    url = value.rstrip("/")
    for suffix in ("/responses", "/chat/completions"):
        url = url.removesuffix(suffix)
    if not url.endswith("/openai/v1"):
        raise ValueError(
            "FOUNDRY_BASE_URL must end with /openai/v1, /openai/v1/responses, "
            "or /openai/v1/chat/completions"
        )
    return url


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_iteration_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG / "iteration_01.json"
    config = read_json(config_path)
    prefix = "iteration_"
    if not config_path.stem.startswith(prefix) or not config_path.stem[len(prefix) :]:
        raise ValueError(
            f"Iteration config filename must follow iteration_<id>.json: {config_path}"
        )
    config["_iteration_id"] = config_path.stem[len(prefix) :]
    config["_iteration_name"] = config_path.stem
    required = {
        "seed",
        "target_generated_examples",
        "validation_fraction",
        "language",
        "max_parallel_requests",
        "max_output_tokens",
        "temperature",
        "intents",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(
            f"Missing required keys in {config_path}: {', '.join(missing)}"
        )
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count
