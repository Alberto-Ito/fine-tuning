#!/usr/bin/env python3
"""Compute agent metrics and a consolidated Banking77 comparison."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINE_TUNING = ROOT.parents[1]
OUTPUT = ROOT / "outputs"
DATASET = FINE_TUNING / "OLD_Qwen3.5-0.8B/outputs/banking77/predictions/two_epochs/test_predictions.jsonl"
DIRECT_METRICS = ROOT.parent / "gpt-5.6-luna/outputs/test_metrics.json"


def load_metadata(path: Path) -> tuple[int, list[str]]:
    indices, labels = set(), {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            indices.add(int(row["index"]))
            labels[int(row["expected_label_id"])] = row["expected_label"]
    if indices != set(range(len(indices))) or set(labels) != set(range(len(labels))):
        raise ValueError(f"Non-contiguous dataset metadata in {path}")
    return len(indices), [labels[index] for index in range(len(labels))]


def main() -> None:
    test_size, labels = load_metadata(DATASET)
    rows_by_index = {}
    with (OUTPUT / "test_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                rows_by_index[int(row["index"])] = row
    if len(rows_by_index) != test_size:
        raise SystemExit(f"Expected {test_size} predictions, found {len(rows_by_index)}")

    # The final column captures agent outputs outside the allowed label set.
    confusion = [[0 for _ in range(len(labels) + 1)] for _ in labels]
    for index in range(test_size):
        row = rows_by_index[index]
        predicted_id = row.get("predicted_label_id")
        column = int(predicted_id) if predicted_id is not None else len(labels)
        confusion[int(row["expected_label_id"])][column] += 1
    per_class = []
    for label_id, label in enumerate(labels):
        tp = confusion[label_id][label_id]
        support = sum(confusion[label_id])
        predicted_count = sum(row[label_id] for row in confusion)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label_id": label_id, "label": label, "precision": precision,
                          "recall": recall, "f1": f1, "support": support})

    accuracy = sum(confusion[i][i] for i in range(len(labels))) / test_size
    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class)
    weighted_f1 = sum(row["f1"] * row["support"] for row in per_class) / test_size
    latencies = sorted(float(row["latency_seconds"]) for row in rows_by_index.values())
    numeric_usage_keys = {key for row in rows_by_index.values()
                          for key, value in row.get("usage", {}).items()
                          if isinstance(value, (int, float))}
    usage = {key: sum(int(row.get("usage", {}).get(key, 0) or 0)
                      for row in rows_by_index.values()) for key in sorted(numeric_usage_keys)}
    agent_versions = Counter(
        (row.get("agent_reference", {}).get("name"), row.get("agent_reference", {}).get("version"))
        for row in rows_by_index.values()
    )
    metrics = {
        "agent": "banking77-agent",
        "agent_version": "2",
        "dataset": "PolyAI/banking77",
        "split": "test",
        "examples": test_size,
        "request_scope": "user_input_only",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "latency_seconds_mean": sum(latencies) / test_size,
        "latency_seconds_p50": latencies[int(0.50 * (test_size - 1))],
        "latency_seconds_p95": latencies[int(0.95 * (test_size - 1))],
        "usage": usage,
        "agent_references": {f"{name}:{version}": count for (name, version), count in agent_versions.items()},
        "invalid_label_outputs": sum(row[-1] for row in confusion),
    }
    (OUTPUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "test_metrics_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader(); writer.writerows(per_class)
    with (OUTPUT / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted", *labels, "__invalid_label__"])
        for label, values in zip(labels, confusion):
            writer.writerow([label, *values])

    direct = json.loads(DIRECT_METRICS.read_text(encoding="utf-8"))
    comparison = [
        ("Qwen3-0.6B LoRA (2 epochs)", 0.8506493506493507, 0.8508984605730113,
         0.8508984605730114, 38_812, 0, 38_812, "local classifier"),
        ("Qwen3.5-0.8B LoRA (2 epochs)", 0.8688311688311688, 0.8688787846760082,
         0.8688787846760081, 38_779, 0, 38_779, "local classifier"),
        ("gpt-5.6-luna (direct)", direct["accuracy"], direct["macro_f1"], direct["weighted_f1"],
         direct["usage"]["input_tokens"], direct["usage"]["output_tokens"],
         direct["usage"]["total_tokens"], "client prompt + labels"),
        ("banking77-agent v2", accuracy, macro_f1, weighted_f1,
         usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0),
         "user input only; server-side instructions"),
    ]
    lines = [
        "# Banking77 — consolidated test comparison", "",
        "All results cover the same 3,080-example test split and 77 labels.", "",
        "| Model / agent | Accuracy | Macro-F1 | Weighted-F1 | Input tokens | Output tokens | Total tokens | Request configuration |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, acc, macro, weighted, inputs, outputs, total_tokens, config in comparison:
        lines.append(f"| {model} | {acc:.4f} | {macro:.4f} | {weighted:.4f} | "
                     f"{inputs:,} | {outputs:,} | {total_tokens:,} | {config} |")
    lines.extend([
        "", "## Main findings", "",
        f"- The persisted agent trails direct GPT-5.6 Luna by {(direct['accuracy'] - accuracy) * 100:.2f} percentage points in accuracy and {(direct['macro_f1'] - macro_f1) * 100:.2f} points in macro-F1.",
        f"- The agent uses {usage.get('total_tokens', 0) / direct['usage']['total_tokens']:.2f}x the API-reported tokens of the direct run because its longer persisted instructions are included server-side on every request.",
        f"- Qwen3.5-0.8B has the strongest result: {(0.8688311688311688 - accuracy) * 100:.2f} accuracy points and {(0.8688787846760082 - macro_f1) * 100:.2f} macro-F1 points above the agent.",
        "- The dedicated agent endpoint's validation response reported the underlying model field as `gpt-4o`, despite the experiment directory name. The evaluated serving target is therefore identified primarily by its immutable agent reference: `banking77-agent:2`.",
        "", "## Agent run", "",
        f"- Completed predictions: {test_size}/{test_size}",
        "- Agent reference: `banking77-agent`, version `2`",
        "- Client-supplied content: only the individual customer message as user input",
        f"- Outputs outside the 77-label set: {metrics['invalid_label_outputs']}",
        f"- Mean latency: {metrics['latency_seconds_mean']:.3f} seconds",
        f"- Latency p50/p95: {metrics['latency_seconds_p50']:.3f} / {metrics['latency_seconds_p95']:.3f} seconds",
        "", "## Token methodology", "",
        "- Qwen token counts were calculated over all test texts using each local tokenizer. The classifiers return logits, so output tokens are zero.",
        "- Direct GPT usage is reported by the Responses API and includes the client-supplied compact instructions and label list on every request.",
        "- Agent usage is reported by the dedicated agent endpoint. The client sent no instructions, labels, or dataset context; input usage includes the persisted server-side agent instructions.",
        "- Token figures cover test inference only and exclude Qwen fine-tuning.", "",
    ])
    (ROOT / "banking77_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
