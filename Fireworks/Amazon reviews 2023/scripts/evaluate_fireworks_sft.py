#!/usr/bin/env python3
"""Build a leakage-safe test set and evaluate Fireworks chat models on it."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASKS = ("pros_and_cons", "use_cases", "common_problems", "purchase_recommendation")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.I)
RECOMMEND_RE = re.compile(r"\b(?:do not buy|don't buy|not recommended|recommend(?:ed|ation)?|buy|purchase|avoid|suitable|unsuitable)\b", re.I)
UNCERTAINTY_RE = re.compile(r"\b(?:may|might|could|reported|according to|evidence|uncertain|limited|one review|some users?)\b", re.I)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_sft_evaluation(source: Path, output: Path) -> dict[str, Any]:
    """Convert system/user/assistant SFT rows into benchmark input/reference rows."""
    prepared = []
    for index, row in enumerate(read_jsonl(source), 1):
        messages = row.get("messages", [])
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            raise ValueError(f"Expected assistant reference in row {index} of {source}")
        prompt_messages = messages[:-1]
        if not any(message.get("role") == "user" for message in prompt_messages):
            raise ValueError(f"Missing user message in row {index} of {source}")
        prepared.append({
            "id": f"evaluation-{index:04d}",
            "task": "purchase_recommendation",
            "messages": prompt_messages,
            "reference": messages[-1].get("content", ""),
            "source_file": str(source),
            "source_row": index,
        })
    write_jsonl(output, prepared)
    manifest = {
        "method": "SFT evaluation rows converted to prompt/reference benchmark rows",
        "source": str(source),
        "total": len(prepared),
        "task_counts": {"purchase_recommendation": len(prepared)},
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_test_set(train_master: Path, candidate_master: Path, output: Path, per_task: int, seed: int) -> dict[str, Any]:
    train_products = {
        row.get("metadata", {}).get("parent_asin")
        for row in read_jsonl(train_master)
        if row.get("metadata", {}).get("parent_asin")
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    seen_ids: set[str] = set()
    for row in read_jsonl(candidate_master):
        meta = row.get("metadata", {})
        task = meta.get("task")
        example_id = meta.get("example_id")
        if meta.get("review_status") != "approved":
            rejected["not_approved"] += 1
            continue
        if meta.get("parent_asin") in train_products:
            rejected["training_product"] += 1
            continue
        if task not in TASKS or not example_id or example_id in seen_ids:
            rejected["invalid_or_duplicate"] += 1
            continue
        messages = row.get("messages", [])
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            rejected["invalid_messages"] += 1
            continue
        seen_ids.add(example_id)
        candidates[task].append(row)

    selected = []
    for task in TASKS:
        pool = sorted(candidates[task], key=lambda r: r["metadata"]["example_id"])
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(pool)
        if len(pool) < per_task:
            raise ValueError(f"Only {len(pool)} eligible examples for {task}; requested {per_task}")
        for row in pool[:per_task]:
            messages = row["messages"]
            selected.append({
                "id": row["metadata"]["example_id"],
                "task": task,
                "parent_asin": row["metadata"]["parent_asin"],
                "messages": messages[:-1],
                "reference": messages[-1]["content"],
                "source_review_count": row["metadata"].get("source_review_count"),
            })
    selected.sort(key=lambda r: (r["task"], r["id"]))
    write_jsonl(output, selected)
    manifest = {
        "method": "approved teacher answers; strict parent_asin exclusion against SFT training candidates",
        "seed": seed,
        "per_task": per_task,
        "total": len(selected),
        "unique_test_products": len({r["parent_asin"] for r in selected}),
        "training_products_excluded": len(train_products),
        "task_counts": dict(Counter(r["task"] for r in selected)),
        "rejected_candidate_counts": dict(rejected),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def call_model(api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int, retries: int, reasoning_effort: str | None) -> tuple[str, dict]:
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"], body.get("usage", {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"Fireworks HTTP {exc.code}: {body[:500]}") from exc
            if attempt == retries:
                raise RuntimeError(f"Fireworks HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Fireworks request failed: {exc}") from exc
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Fireworks request failed after {retries + 1} attempts")


def run_model(test_path: Path, output: Path, model: str, workers: int, max_tokens: int, limit: int | None, reasoning_effort: str | None, resume: bool) -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in the environment; it is never written to disk.")
    rows = list(read_jsonl(test_path))
    if limit:
        rows = rows[:limit]
    retained: dict[str, dict[str, Any]] = {}
    if resume and output.exists():
        retained = {row["id"]: row for row in read_jsonl(output) if not row.get("error")}
        rows = [row for row in rows if row["id"] not in retained]
        print(f"Resuming: retained {len(retained)} successful rows; retrying {len(rows)} rows", flush=True)

    def execute(row):
        started = time.monotonic()
        try:
            prediction, usage = call_model(api_key, model, row["messages"], max_tokens, 120, 4, reasoning_effort)
            return {**row, "model": model, "prediction": prediction, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3), "error": None}
        except Exception as exc:  # keep partial runs inspectable
            return {**row, "model": model, "prediction": "", "usage": {}, "latency_seconds": round(time.monotonic() - started, 3), "error": str(exc)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, row): row["id"] for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            print(f"[{index}/{len(rows)}] {futures[future]}", flush=True)
    results.extend(retained.values())
    results.sort(key=lambda r: (r["task"], r["id"]))
    write_jsonl(output, results)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(reference: str, prediction: str) -> float:
    ref, pred = Counter(tokens(reference)), Counter(tokens(prediction))
    overlap = sum((ref & pred).values())
    if not ref or not pred or not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref, pred = tokens(reference), tokens(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_length(ref, pred)
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def extract_user_evidence(messages: list[dict[str, str]]) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") in {"system", "user"})


def unsupported_token_rate(evidence: str, prediction: str) -> float:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "be", "with", "that", "this", "it", "as", "on", "by", "from", "should", "product", "users", "reviews"}
    source = set(tokens(evidence))
    meaningful = [t for t in tokens(prediction) if len(t) > 3 and t not in stop]
    return sum(t not in source for t in meaningful) / len(meaningful) if meaningful else 1.0


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    prediction, reference = row.get("prediction", ""), row["reference"]
    evidence = extract_user_evidence(row["messages"])
    pred_tokens = tokens(prediction)
    return {
        "id": row["id"], "task": row["task"], "error": row.get("error"),
        "token_f1": token_f1(reference, prediction),
        "rouge_l_f1": rouge_l_f1(reference, prediction),
        "unsupported_token_rate": unsupported_token_rate(evidence, prediction),
        "length_words": len(pred_tokens),
        "empty": not bool(pred_tokens),
        "has_recommendation_language": bool(RECOMMEND_RE.search(prediction)),
        "has_uncertainty_language": bool(UNCERTAINTY_RE.search(prediction)),
    }


def summarize(predictions: list[Path], output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"models": {}}
    for path in predictions:
        rows = list(read_jsonl(path))
        metrics = [row_metrics(row) for row in rows]
        model = rows[0].get("model", path.stem) if rows else path.stem
        successful = [m for m in metrics if not m["error"]]
        def aggregate(items):
            return {
                "n": len(items),
                "token_f1": round(statistics.mean(m["token_f1"] for m in items), 4) if items else None,
                "rouge_l_f1": round(statistics.mean(m["rouge_l_f1"] for m in items), 4) if items else None,
                "unsupported_token_rate": round(statistics.mean(m["unsupported_token_rate"] for m in items), 4) if items else None,
                "mean_length_words": round(statistics.mean(m["length_words"] for m in items), 1) if items else None,
                "recommendation_language_rate": round(statistics.mean(m["has_recommendation_language"] for m in items), 4) if items else None,
                "uncertainty_language_rate": round(statistics.mean(m["has_uncertainty_language"] for m in items), 4) if items else None,
            }
        report["models"][model] = {
            "overall": aggregate(successful),
            "by_task": {task: aggregate([m for m in successful if m["task"] == task]) for task in TASKS},
            "errors": [m for m in metrics if m["error"]],
        }
    models = list(report["models"])
    if len(models) == 2:
        first, second = models
        report["comparison"] = {
            "first": first, "second": second,
            "delta_second_minus_first": {
                key: round(report["models"][second]["overall"][key] - report["models"][first]["overall"][key], 4)
                for key in ("token_f1", "rouge_l_f1", "unsupported_token_rate")
                if report["models"][first]["overall"][key] is not None and report["models"][second]["overall"][key] is not None
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--train-master", type=Path, required=True)
    build.add_argument("--candidate-master", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--per-task", type=int, default=50)
    build.add_argument("--seed", type=int, default=20261014)
    run = sub.add_parser("run")
    run.add_argument("--test", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-tokens", type=int, default=500)
    run.add_argument("--limit", type=int)
    run.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    run.add_argument("--resume", action="store_true", help="Retain successful rows in an existing output and retry errors")
    summary = sub.add_parser("summarize")
    summary.add_argument("--predictions", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_sft_evaluation(args.source, args.output), indent=2))
    elif args.command == "build":
        print(json.dumps(build_test_set(args.train_master, args.candidate_master, args.output, args.per_task, args.seed), indent=2))
    elif args.command == "run":
        run_model(args.test, args.output, args.model, args.workers, args.max_tokens, args.limit, args.reasoning_effort, args.resume)
    else:
        print(json.dumps(summarize(args.predictions, args.output), indent=2))


if __name__ == "__main__":
    main()
