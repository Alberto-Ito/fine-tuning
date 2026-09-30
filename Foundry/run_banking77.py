#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible Microsoft Foundry deployment on Banking77.

The runner is resumable: each successful response is appended immediately to
JSONL, and subsequent executions skip completed dataset indices.
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
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://alberto-ito-poc-resource.services.ai.azure.com/openai/v1/responses"
DEFAULT_MODEL = "gpt-5.6-luna"
PRINT_LOCK = threading.Lock()
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "OLD_Qwen3.5-0.8B/outputs/banking77/predictions/two_epochs/test_predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.getenv("FOUNDRY_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("FOUNDRY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Only for smoke tests")
    return parser.parse_args()


def get_api_key() -> str:
    key = os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set FOUNDRY_API_KEY (or AZURE_OPENAI_API_KEY) in the environment.")
    return key


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


def load_test_data(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the canonical test inputs already used by both local Qwen runs."""
    rows: dict[int, dict[str, Any]] = {}
    labels_by_id: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["index"])
            expected_id = int(row["expected_label_id"])
            rows[index] = {"text": row["text"], "label": expected_id}
            labels_by_id[expected_id] = row["expected_label"]
    expected_indices = list(range(len(rows)))
    if sorted(rows) != expected_indices:
        raise ValueError(f"Dataset indices in {path} are not contiguous")
    expected_label_ids = list(range(len(labels_by_id)))
    if sorted(labels_by_id) != expected_label_ids:
        raise ValueError(f"Label ids in {path} are not contiguous")
    return [rows[index] for index in expected_indices], [labels_by_id[i] for i in expected_label_ids]


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
    # Accept a label embedded in a short explanatory response, choosing the
    # longest match first to avoid matching a substring of another label.
    padded = f" {re.sub(r'[^a-z0-9_]+', ' ', lowered)} "
    matches = [label for label in labels if f" {label.lower()} " in padded]
    return max(matches, key=len) if matches else None


def request_prediction(
    *, endpoint: str, api_key: str, model: str, system_prompt: str,
    text: str, labels: list[str], timeout: float, max_retries: int,
) -> tuple[str, str, dict[str, Any], float, str]:
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": text,
        "max_output_tokens": 128,
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        request = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "api-key": api_key,
            },
        )
        try:
            with urlopen(request, timeout=timeout) as http_response:
                response = json.loads(http_response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            raw = extract_output_text(response)
            predicted = normalize_prediction(raw, labels)
            if predicted is None:
                diagnostic = {
                    "status": response.get("status"),
                    "incomplete_details": response.get("incomplete_details"),
                    "usage": response.get("usage"),
                    "output": response.get("output"),
                }
                raise ValueError(f"unparseable model output: {raw!r}; response={diagnostic!r}")
            return predicted, raw, response.get("usage", {}), latency, response.get("id", "")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {detail}"
            retryable = exc.code in {408, 409, 429} or exc.code >= 500
            if not retryable:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"

    test, labels = load_test_data(args.dataset_jsonl)
    total = min(len(test), args.limit) if args.limit else len(test)
    completed = load_completed(predictions_path)
    pending = [index for index in range(total) if index not in completed]
    label_list = ", ".join(labels)
    system_prompt = (
        "Classify the banking customer query into exactly one Banking77 intent. "
        "Reply with only the exact intent label, with no explanation. Valid labels: " + label_list
    )
    print(f"Dataset: Banking77 test ({total} cases, {len(labels)} labels)")
    print(f"Model: {args.model}; completed: {len(completed)}; pending: {len(pending)}")

    def evaluate(index: int) -> dict[str, Any]:
        item = test[index]
        predicted, raw, usage, latency, response_id = request_prediction(
            endpoint=args.endpoint, api_key=api_key, model=args.model,
            system_prompt=system_prompt, text=item["text"], labels=labels,
            timeout=args.timeout, max_retries=args.max_retries,
        )
        expected_id = int(item["label"])
        return {
            "index": index,
            "text": item["text"],
            "expected_label_id": expected_id,
            "expected_label": labels[expected_id],
            "predicted_label_id": labels.index(predicted),
            "predicted_label": predicted,
            "correct": predicted == labels[expected_id],
            "latency_seconds": latency,
            "usage": usage,
            "response_id": response_id,
            "raw_output": raw,
            "model": args.model,
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
                with PRINT_LOCK:
                    prediction_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[index] = row
                done += 1
                if done % 25 == 0 or done == total:
                    correct = sum(bool(row["correct"]) for row in completed.values())
                    print(f"Progress: {done}/{total} ({done / total:.1%}); running accuracy={correct / done:.4f}", flush=True)
            except Exception as exc:  # retain failures for diagnosis; rerun retries them
                error = {"index": index, "error": str(exc), "timestamp": time.time()}
                with PRINT_LOCK:
                    error_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(f"Failed index {index}: {exc}", flush=True)

    missing = [index for index in range(total) if index not in completed]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} cases failed. Rerun the same command to retry them.")
    print("All cases completed. Run compute_metrics.py to generate the reports.")


if __name__ == "__main__":
    main()
