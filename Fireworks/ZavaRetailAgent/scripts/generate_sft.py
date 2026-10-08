from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

# Data Designer reads this setting during import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_DESIGNER_HOME", str(PROJECT_ROOT / ".data-designer"))

import data_designer.config as dd
import numpy as np
import pandas as pd
from common import (
    CONFIG,
    GENERATED,
    REFERENCE,
    ROOT,
    SEEDS,
    ensure_directories,
    load_environment,
    load_iteration_config,
    normalize_foundry_base_url,
    read_json,
    read_jsonl,
    require_environment,
    write_jsonl,
)
from data_designer.interface import DataDesigner
from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"]
    content: str


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["tool"]
    tool_call_id: str
    content: str


ConversationMessage = UserMessage | AssistantMessage | ToolMessage


class GeneratedConversation(BaseModel):
    messages: list[ConversationMessage] = Field(min_length=2)


GENERATION_PROMPT = """
Create one high-quality supervised fine-tuning conversation for the Zava retail agent.

Scenario seed:
- seed_id: {{ seed_id }}
- target intent: {{ intent }}
- variation number: {{ variation }}
- language: {{ language }}
- required tool sequence: {{ required_tool_sequence_json }}
- authoritative database context: {{ context_json }}

Requirements:
1. Return messages only for the user/assistant/tool portion. Do not include the system message; it is added later.
2. Follow the supplied retail policy exactly.
3. Ground every user, order, item, product, payment method, and tool result in context_json. Never invent identifiers.
4. Authenticate before disclosing account data or acting. Prefer email; name plus ZIP is the fallback.
5. Use only tools in the supplied tool catalog. Emit at most one tool call per assistant message.
6. A tool-calling assistant message must not also contain natural-language content.
7. Every non-final tool call must be followed by a matching tool message with the exact tool_call_id. The conversation may end on an assistant tool call, matching the source Zava SFT format.
8. Obtain explicit confirmation before consequential changes.
9. For policy_refusal_other_user, demonstrate the correct refusal without exposing the other user's private data.
10. Make the wording and dialogue trajectory meaningfully different from template-like repetitions.
11. Finish at a natural stopping point. Match the source dataset convention of ending immediately after the final consequential tool call when appropriate.
12. Execute every tool in required_tool_sequence, in that exact order. Additional get_product_details calls are allowed only when needed, but never skip a required step.
13. The first tool must authenticate the user. After authentication, call get_user_details before disclosing account information.
14. Before an order action, call get_order_details even when the order data appears in context_json; the context is grounding data, not permission to skip the observable tool workflow.
15. Every tool message must contain the exact JSON or scalar result supported by context_json. Never emit an empty tool result.
16. Immediately before a consequential action call, the assistant must summarize the action and the user must explicitly confirm it.
17. For modify_pending_order_payment, choose an existing payment method different from the order's original method. A gift card is valid only if its balance covers the entire order total. If context_json has no valid alternative, do not fabricate one.
""".strip()


def build_configuration(
    seed_path: Path, model: str, iteration_config: dict[str, Any]
) -> dd.DataDesignerConfigBuilder:
    policy = (REFERENCE / "policy.md").read_text(encoding="utf-8")
    tools = (REFERENCE / "retail_tools.json").read_text(encoding="utf-8")
    system_prompt = (
        "You create grounded training examples, not customer-facing answers. "
        "Do not reveal hidden reasoning.\n\n"
        "AUTHORITATIVE RETAIL POLICY:\n"
        + policy
        + "\n\nAUTHORITATIVE TOOL CATALOG:\n"
        + tools
    )
    model_config = dd.ModelConfig(
        alias="foundry-generator",
        model=model,
        provider="microsoft-foundry",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=iteration_config["temperature"],
            top_p=0.95,
            max_tokens=iteration_config["max_output_tokens"],
            max_parallel_requests=iteration_config["max_parallel_requests"],
            timeout=180,
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="generated_conversation",
            prompt=GENERATION_PROMPT,
            system_prompt=system_prompt,
            model_alias="foundry-generator",
            output_format=GeneratedConversation,
        )
    )
    return builder


def normalize_generated(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)

    def to_python(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return [to_python(child) for child in item.tolist()]
        if isinstance(item, dict):
            return {
                key: to_python(child)
                for key, child in item.items()
                if child is not None
            }
        if isinstance(item, list):
            return [to_python(child) for child in item]
        if hasattr(item, "item"):
            return item.item()
        return item

    normalized = to_python(value)
    if not isinstance(normalized, dict) or not isinstance(
        normalized.get("messages"), list
    ):
        raise TypeError("Data Designer returned an unexpected structured result")
    # Foundry occasionally includes a redundant narration beside a tool call even
    # when explicitly instructed not to. The source Zava format supervises only
    # the call in that turn, so canonicalize it before validation and export.
    for message in normalized["messages"]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message.pop("content", None)
    validated = GeneratedConversation.model_validate(normalized)
    return validated.model_dump(exclude_none=True)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(CONFIG / "iteration_01.json"))
    known_args, _ = config_parser.parse_known_args()
    iteration_config = load_iteration_config(Path(known_args.config))
    parser = argparse.ArgumentParser(
        description="Generate grounded Zava SFT examples through Microsoft Foundry.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--count",
        type=int,
        default=iteration_config["target_generated_examples"],
        help="Number of examples to request.",
    )
    parser.add_argument(
        "--seed-file",
        default=str(SEEDS / f"{iteration_config['_iteration_name']}.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(GENERATED / f"{iteration_config['_iteration_name']}_raw.jsonl"),
    )
    parser.add_argument(
        "--dataset-name", default=f"zava-{iteration_config['_iteration_name']}"
    )
    parser.add_argument(
        "--artifact-parquet",
        help="Reuse an already generated Data Designer Parquet file without calling Foundry.",
    )
    parser.add_argument(
        "--seed-id",
        action="append",
        dest="seed_ids",
        help=(
            "Generate only this seed ID. May be repeated. Existing output records "
            "with those seed IDs are replaced while all other records are preserved."
        ),
    )
    args = parser.parse_args()

    ensure_directories()
    load_environment()
    env = require_environment(
        "FOUNDRY_BASE_URL", "FOUNDRY_MODEL_DEPLOYMENT", "FOUNDRY_API_KEY"
    )
    seed_path = Path(args.seed_file).resolve()
    if not seed_path.exists():
        raise SystemExit(
            f"Seed file not found: {seed_path}. Run scripts/prepare_seeds.py first."
        )

    selected_seed_ids = set(args.seed_ids or [])
    if selected_seed_ids:
        seed_records = read_jsonl(seed_path)
        selected_seeds = [
            record
            for record in seed_records
            if record.get("seed_id") in selected_seed_ids
        ]
        found_seed_ids = {record.get("seed_id") for record in selected_seeds}
        missing_seed_ids = sorted(selected_seed_ids - found_seed_ids)
        if missing_seed_ids:
            raise SystemExit(f"Unknown seed IDs: {', '.join(missing_seed_ids)}")
        seed_path = ROOT / "artifacts" / f"{iteration_config['_iteration_name']}_retry_seeds.jsonl"
        write_jsonl(seed_path, selected_seeds)
        args.count = len(selected_seeds)

    if args.artifact_parquet:
        dataframe = pd.read_parquet(args.artifact_parquet)
    else:
        provider = dd.ModelProvider(
            name="microsoft-foundry",
            endpoint=normalize_foundry_base_url(env["FOUNDRY_BASE_URL"]),
            provider_type="openai",
            api_key="FOUNDRY_API_KEY",
        )
        builder = build_configuration(
            seed_path, env["FOUNDRY_MODEL_DEPLOYMENT"], iteration_config
        )
        designer = DataDesigner(
            artifact_path=ROOT / "artifacts",
            model_providers=[provider],
        )
        results = designer.create(
            builder,
            num_records=args.count,
            dataset_name=args.dataset_name,
        )
        dataframe = results.load_dataset()
    policy = (REFERENCE / "policy.md").read_text(encoding="utf-8")
    tools = read_json(REFERENCE / "retail_tools.json")

    output_records = []
    for _, row in dataframe.iterrows():
        generated = normalize_generated(row["generated_conversation"])
        output_records.append(
            {
                "messages": [
                    {"role": "system", "content": policy},
                    *generated["messages"],
                ],
                "tools": tools,
                "metadata": {
                    "source": "synthetic",
                    "generator": "nemo-data-designer",
                    "foundry_deployment": env["FOUNDRY_MODEL_DEPLOYMENT"],
                    "seed_id": row["seed_id"],
                    "intent": row["intent"],
                    "iteration": iteration_config["_iteration_id"],
                },
            }
        )
    output_path = Path(args.output)
    if selected_seed_ids and output_path.exists():
        retained_records = [
            record
            for record in read_jsonl(output_path)
            if record.get("metadata", {}).get("seed_id") not in selected_seed_ids
        ]
        output_records = [*retained_records, *output_records]
        output_records.sort(key=lambda record: record["metadata"]["seed_id"])
    written = write_jsonl(output_path, output_records)
    print(f"Generated {written} raw SFT examples at {args.output}")


if __name__ == "__main__":
    main()
