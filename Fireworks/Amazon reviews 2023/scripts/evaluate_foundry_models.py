#!/usr/bin/env python3
"""Run Azure AI Foundry Responses models and judge three benchmark outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DECISIONS = ["BUY", "CONDITIONAL_BUY", "DO_NOT_BUY", "INSUFFICIENT_EVIDENCE", "UNDETERMINED"]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_text(body: dict[str, Any]) -> str:
    if body.get("output_text"):
        return body["output_text"]
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(content.get("text", ""))
    return "".join(parts)


def post_responses(endpoint: str, api_key: str, payload: dict[str, Any], retries: int = 6) -> tuple[str, dict, dict]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "api-key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            return response_text(body), body.get("usage", {}), body
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"Foundry HTTP {exc.code}: {message[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Foundry request failed: {exc}") from exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("Foundry request failed")


def run_generation(test: Path, output: Path, endpoint: str, model: str, workers: int, resume: bool, limit: int | None, max_output_tokens: int) -> None:
    key = os.environ.get("AZURE_AI_API_KEY")
    if not key:
        raise SystemExit("Set AZURE_AI_API_KEY; the key is never written to disk.")
    rows = list(read_jsonl(test))
    if limit:
        rows = rows[:limit]
    retained = {}
    if resume and output.exists():
        retained = {row["id"]: row for row in read_jsonl(output) if not row.get("error")}
        rows = [row for row in rows if row["id"] not in retained]
        print(f"Resuming: retained {len(retained)}; retrying {len(rows)}", flush=True)

    def execute(row):
        started = time.monotonic()
        payload = {
            "model": model,
            "input": row["messages"],
            "max_output_tokens": max_output_tokens,
            "temperature": 0,
            "store": False,
        }
        try:
            prediction, usage, _ = post_responses(endpoint, key, payload)
            return {**row, "model": model, "prediction": prediction, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3), "error": None}
        except Exception as exc:
            return {**row, "model": model, "prediction": "", "usage": {}, "latency_seconds": round(time.monotonic() - started, 3), "error": str(exc)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, row): row["id"] for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            print(f"[{index}/{len(rows)}] {futures[future]}", flush=True)
    results.extend(retained.values())
    results.sort(key=lambda row: (row["task"], row["id"]))
    write_jsonl(output, results)


def score_schema() -> dict[str, Any]:
    evaluation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "inferred_decision": {"type": "string", "enum": DECISIONS},
            "decision_correct": {"type": "boolean"},
            "groundedness": {"type": "integer", "minimum": 0, "maximum": 100},
            "evidence_coverage": {"type": "integer", "minimum": 0, "maximum": 100},
            "rule_compliance": {"type": "integer", "minimum": 0, "maximum": 100},
            "severity_calibration": {"type": "integer", "minimum": 0, "maximum": 100},
            "catalog_user_separation": {"type": "integer", "minimum": 0, "maximum": 100},
            "actionability": {"type": "integer", "minimum": 0, "maximum": 100},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "critical_errors": {"type": "array", "items": {"type": "string"}},
            "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "pass": {"type": "boolean"},
        },
        "required": ["inferred_decision", "decision_correct", "groundedness", "evidence_coverage", "rule_compliance", "severity_calibration", "catalog_user_separation", "actionability", "unsupported_claims", "critical_errors", "overall_score", "pass"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reference_decision": {"type": "string", "enum": DECISIONS},
            "A": evaluation,
            "B": evaluation,
            "C": evaluation,
            "ranking": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "C"]}, "minItems": 3, "maxItems": 3},
            "best_answer_reason": {"type": "string"},
        },
        "required": ["reference_decision", "A", "B", "C", "ranking", "best_answer_reason"],
    }


JUDGE_INSTRUCTIONS = """You are a strict evaluator of a technical purchasing copilot. Judge each candidate independently from the supplied product catalog facts, review evidence, task, reference answer, and rules. Do not reward stylistic similarity by itself. A claim is grounded only when it follows from the supplied evidence. Distinguish catalog facts from user reports. Never treat one isolated report as recurring. Penalize invented specifications, safety guarantees, unsupported compatibility, disproportionate severity, omitted critical evidence, or a purchase decision that conflicts with the evidence.

Infer one decision for the reference and every candidate: BUY, CONDITIONAL_BUY, DO_NOT_BUY, INSUFFICIENT_EVIDENCE, or UNDETERMINED. decision_correct means the candidate decision is substantively consistent with the reference decision and evidence, allowing equivalent wording. A critical error is a reversed purchase decision, invented/omitted safety issue, hard-requirement violation, or materially unsupported claim that could change the purchase.

Scores are 0-100. overall_score weights groundedness 30%, decision correctness 25%, rule compliance 20%, evidence coverage 15%, actionability 10%; severity and catalog/user separation should constrain the relevant component. If any critical error exists, pass must be false and overall_score must not exceed 49. Otherwise pass requires overall_score >= 75, groundedness >= 80, and rule_compliance >= 80. Rank A/B/C best to worst; include all three exactly once. Return only schema-valid JSON."""


def run_judge(test: Path, prediction_files: list[Path], output: Path, endpoint: str, model: str, workers: int, resume: bool, limit: int | None) -> None:
    key = os.environ.get("AZURE_AI_API_KEY")
    if not key:
        raise SystemExit("Set AZURE_AI_API_KEY; the key is never written to disk.")
    tests = {row["id"]: row for row in read_jsonl(test)}
    sources = []
    for path in prediction_files:
        rows = {row["id"]: row for row in read_jsonl(path)}
        if set(rows) != set(tests):
            raise ValueError(f"Prediction IDs do not match test set: {path}")
        if any(row.get("error") for row in rows.values()):
            raise ValueError(f"Predictions contain errors: {path}")
        sources.append(rows)
    retained = {}
    pending_ids = list(tests)
    if limit:
        pending_ids = pending_ids[:limit]
    if resume and output.exists():
        retained = {row["id"]: row for row in read_jsonl(output) if not row.get("error")}
        pending_ids = [item_id for item_id in pending_ids if item_id not in retained]
        print(f"Resuming judge: retained {len(retained)}; retrying {len(pending_ids)}", flush=True)

    def execute(item_id):
        test_row = tests[item_id]
        order = list(range(3))
        random.Random(f"judge:{item_id}").shuffle(order)
        aliases = ["A", "B", "C"]
        alias_to_model = {alias: sources[source_index][item_id]["model"] for alias, source_index in zip(aliases, order)}
        candidates = "\n\n".join(f"Candidate {alias}:\n{sources[source_index][item_id]['prediction']}" for alias, source_index in zip(aliases, order))
        context = "\n\n".join(message.get("content", "") for message in test_row["messages"])
        user_input = f"TASK AND EVIDENCE:\n{context}\n\nREFERENCE ANSWER:\n{test_row['reference']}\n\n{candidates}"
        payload = {
            "model": model,
            "instructions": JUDGE_INSTRUCTIONS,
            "input": user_input,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 1800,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "purchasing_copilot_judgment", "strict": True, "schema": score_schema()}},
        }
        started = time.monotonic()
        try:
            raw, usage, _ = post_responses(endpoint, key, payload)
            judgment = json.loads(raw)
            return {"id": item_id, "task": test_row["task"], "judge_model": model, "alias_to_model": alias_to_model, "judgment": judgment, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3), "error": None}
        except Exception as exc:
            return {"id": item_id, "task": test_row["task"], "judge_model": model, "alias_to_model": alias_to_model, "judgment": None, "usage": {}, "latency_seconds": round(time.monotonic() - started, 3), "error": str(exc)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, item_id): item_id for item_id in pending_ids}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            print(f"[judge {index}/{len(pending_ids)}] {futures[future]}", flush=True)
    results.extend(retained.values())
    results.sort(key=lambda row: (row["task"], row["id"]))
    write_jsonl(output, results)


def summarize_judgments(judgments: Path, output: Path) -> dict[str, Any]:
    rows = list(read_jsonl(judgments))
    errors = [row for row in rows if row.get("error")]
    model_rows = defaultdict(list)
    wins = Counter()
    reference_decisions = Counter()
    for row in rows:
        if row.get("error"):
            continue
        judgment = row["judgment"]
        reference_decisions[judgment["reference_decision"]] += 1
        winner_alias = judgment["ranking"][0]
        wins[row["alias_to_model"][winner_alias]] += 1
        for alias, model in row["alias_to_model"].items():
            model_rows[model].append({**judgment[alias], "task": row["task"]})

    def aggregate(items):
        numeric = ["groundedness", "evidence_coverage", "rule_compliance", "severity_calibration", "catalog_user_separation", "actionability", "overall_score"]
        result = {"n": len(items)}
        result.update({field: round(statistics.mean(item[field] for item in items), 2) for field in numeric})
        result.update({
            "decision_accuracy": round(statistics.mean(item["decision_correct"] for item in items), 4),
            "pass_rate": round(statistics.mean(item["pass"] for item in items), 4),
            "critical_error_rate": round(statistics.mean(bool(item["critical_errors"]) for item in items), 4),
            "unsupported_claim_rate": round(statistics.mean(bool(item["unsupported_claims"]) for item in items), 4),
        })
        return result

    summary = {
        "judge": rows[0].get("judge_model") if rows else None,
        "cases": len(rows),
        "successful_judgments": len(rows) - len(errors),
        "errors": errors,
        "reference_decisions": dict(reference_decisions),
        "models": {},
    }
    for model, items in model_rows.items():
        summary["models"][model] = {
            "overall": aggregate(items),
            "wins": wins[model],
            "win_rate": round(wins[model] / len(items), 4) if items else None,
            "by_task": {task: aggregate([item for item in items if item["task"] == task]) for task in sorted({item["task"] for item in items})},
            "decision_distribution": dict(Counter(item["inferred_decision"] for item in items)),
        }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--test", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--endpoint", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--workers", type=int, default=4)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--max-output-tokens", type=int, default=500)
    judge = sub.add_parser("judge")
    judge.add_argument("--test", type=Path, required=True)
    judge.add_argument("--predictions", type=Path, nargs=3, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--endpoint", required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--workers", type=int, default=3)
    judge.add_argument("--resume", action="store_true")
    judge.add_argument("--limit", type=int)
    summary = sub.add_parser("summarize")
    summary.add_argument("--judgments", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        run_generation(args.test, args.output, args.endpoint, args.model, args.workers, args.resume, args.limit, args.max_output_tokens)
    elif args.command == "judge":
        run_judge(args.test, args.predictions, args.output, args.endpoint, args.model, args.workers, args.resume, args.limit)
    else:
        print(json.dumps(summarize_judgments(args.judgments, args.output), indent=2))


if __name__ == "__main__":
    main()
