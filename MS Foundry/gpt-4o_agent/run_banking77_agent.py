#!/usr/bin/env python3
"""Evaluate the persisted Foundry banking77-agent on all Banking77 test cases.

Each API request contains only the customer text as user input. Intent
instructions and labels are intentionally left to the persisted agent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = (
    "https://alberto-ito-poc-resource.services.ai.azure.com/api/projects/"
    "alberto-ito-poc/agents/banking77-agent/endpoint/protocols/openai/responses"
)
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "OLD_Qwen3.5-0.8B/outputs/banking77/predictions/two_epochs/test_predictions.jsonl"
)
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.getenv("FOUNDRY_AGENT_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--api-version", default="v1")
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Only for smoke tests")
    return parser.parse_args()


def endpoint_with_api_version(endpoint: str, api_version: str) -> str:
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query))
    query["api-version"] = api_version
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_api_key() -> str:
    key = os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set FOUNDRY_API_KEY (or AZURE_OPENAI_API_KEY) in the environment.")
    return key


def load_test_data(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: dict[int, dict[str, Any]] = {}
    labels_by_id: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["index"])
            label_id = int(row["expected_label_id"])
            rows[index] = {"text": row["text"], "label": label_id}
            labels_by_id[label_id] = row["expected_label"]
    if sorted(rows) != list(range(len(rows))):
        raise ValueError(f"Dataset indices in {path} are not contiguous")
    if sorted(labels_by_id) != list(range(len(labels_by_id))):
        raise ValueError(f"Label ids in {path} are not contiguous")
    return [rows[index] for index in range(len(rows))], [labels_by_id[i] for i in range(len(labels_by_id))]


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    completed[int(row["index"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"Warning: ignored malformed output line {line_number}")
    return completed


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def normalize_prediction(raw: str, labels: list[str]) -> str | None:
    value = raw.strip().strip("`\"'").splitlines()[0].strip() if raw.strip() else ""
    if value in labels:
        return value
    lowered = value.lower().replace(" ", "_").replace("-", "_").rstrip(". ,;:")
    exact = {label.lower(): label for label in labels}
    if lowered in exact:
        return exact[lowered]
    padded = f" {re.sub(r'[^a-z0-9_?]+', ' ', lowered)} "
    matches = [label for label in labels if f" {label.lower()} " in padded]
    return max(matches, key=len) if matches else None


def request_prediction(
    *, endpoint: str, api_key: str, text: str, labels: list[str],
    timeout: float, max_retries: int,
) -> tuple[str | None, str, dict[str, Any], float, str, dict[str, Any], str]:
    # Deliberately only `input`: all classification instructions live in the agent.
    body = json.dumps({"input": [{"role": "user", "content": text}]}).encode("utf-8")
    last_error = ""
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        request = Request(endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        })
        try:
            with urlopen(request, timeout=timeout) as http_response:
                response = json.loads(http_response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            raw = extract_output_text(response)
            predicted = normalize_prediction(raw, labels)
            return (predicted, raw, response.get("usage", {}), latency,
                    response.get("id", ""), response.get("agent_reference", {}),
                    response.get("model", ""))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 409, 429} and exc.code < 500:
                raise RuntimeError(last_error) from exc
            retry_after = exc.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2 ** attempt)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            delay = min(60.0, 2 ** attempt)
        if attempt < max_retries:
            time.sleep(delay + random.random())
    raise RuntimeError(f"request failed after {max_retries + 1} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    endpoint = endpoint_with_api_version(args.endpoint, args.api_version)
    test, labels = load_test_data(args.dataset_jsonl)
    total = min(len(test), args.limit) if args.limit else len(test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    completed = load_completed(predictions_path)
    pending = [index for index in range(total) if index not in completed]
    print(f"Dataset: Banking77 test ({total} cases, {len(labels)} labels)")
    print(f"Agent: banking77-agent v2; completed: {len(completed)}; pending: {len(pending)}")

    def evaluate(index: int) -> dict[str, Any]:
        item = test[index]
        predicted, raw, usage, latency, response_id, agent_reference, reported_model = request_prediction(
            endpoint=endpoint, api_key=api_key, text=item["text"], labels=labels,
            timeout=args.timeout, max_retries=args.max_retries,
        )
        expected_id = int(item["label"])
        return {
            "index": index,
            "text": item["text"],
            "expected_label_id": expected_id,
            "expected_label": labels[expected_id],
            "predicted_label_id": labels.index(predicted) if predicted is not None else None,
            "predicted_label": predicted if predicted is not None else raw.strip(),
            "valid_label": predicted is not None,
            "correct": predicted == labels[expected_id],
            "latency_seconds": latency,
            "usage": usage,
            "response_id": response_id,
            "raw_output": raw,
            "agent_reference": agent_reference,
            "reported_model": reported_model,
            "request_scope": "user_input_only",
            "status": "ok",
        }

    done = len(completed)
    with predictions_path.open("a", encoding="utf-8", buffering=1) as prediction_file, \
            errors_path.open("a", encoding="utf-8", buffering=1) as error_file, \
            ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                row = future.result()
                with WRITE_LOCK:
                    prediction_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[index] = row
                done += 1
                if done % 25 == 0 or done == total:
                    correct = sum(bool(row["correct"]) for row in completed.values())
                    print(f"Progress: {done}/{total} ({done / total:.1%}); running accuracy={correct / done:.4f}", flush=True)
            except Exception as exc:
                with WRITE_LOCK:
                    error_file.write(json.dumps({"index": index, "error": str(exc),
                                                 "timestamp": time.time()}, ensure_ascii=False) + "\n")
                print(f"Failed index {index}: {exc}", flush=True)

    missing = [index for index in range(total) if index not in completed]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} cases failed. Rerun to retry them.")
    print("All cases completed. Run compute_metrics.py to generate the reports.")


if __name__ == "__main__":
    main()
