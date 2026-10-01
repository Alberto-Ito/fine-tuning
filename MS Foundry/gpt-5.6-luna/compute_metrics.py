#!/usr/bin/env python3
"""Compute Banking77 metrics and comparison tables from Foundry JSONL output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DEFAULT_DATASET = (
    ROOT.parent / "OLD_Qwen3.5-0.8B/outputs/banking77/predictions/two_epochs/test_predictions.jsonl"
)


def load_dataset_metadata(path: Path) -> tuple[int, list[str]]:
    indices = set()
    labels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            indices.add(int(row["index"]))
            labels[int(row["expected_label_id"])] = row["expected_label"]
    if indices != set(range(len(indices))) or set(labels) != set(range(len(labels))):
        raise ValueError(f"Non-contiguous dataset metadata in {path}")
    return len(indices), [labels[index] for index in range(len(labels))]


def main() -> None:
    test_size, labels = load_dataset_metadata(DEFAULT_DATASET)
    rows_by_index = {}
    with (OUTPUT / "test_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                rows_by_index[int(row["index"])] = row
    if len(rows_by_index) != test_size:
        raise SystemExit(f"Expected {test_size} predictions, found {len(rows_by_index)}")

    confusion = [[0 for _ in labels] for _ in labels]
    for index in range(test_size):
        row = rows_by_index[index]
        confusion[int(row["expected_label_id"])][int(row["predicted_label_id"])] += 1

    per_class = []
    for label_id, label in enumerate(labels):
        tp = confusion[label_id][label_id]
        support = sum(confusion[label_id])
        predicted_count = sum(matrix_row[label_id] for matrix_row in confusion)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label_id": label_id, "label": label, "precision": precision,
                          "recall": recall, "f1": f1, "support": support})

    total = test_size
    accuracy = sum(confusion[i][i] for i in range(len(labels))) / total
    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class)
    weighted_f1 = sum(row["f1"] * row["support"] for row in per_class) / total
    latencies = sorted(float(row["latency_seconds"]) for row in rows_by_index.values())
    usage_keys = {key for row in rows_by_index.values() for key, value in row.get("usage", {}).items()
                  if isinstance(value, (int, float))}
    usage = {key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in rows_by_index.values())
             for key in sorted(usage_keys)}
    metrics = {
        "model": next(iter(rows_by_index.values()))["model"],
        "dataset": "PolyAI/banking77",
        "split": "test",
        "examples": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "latency_seconds_mean": sum(latencies) / total,
        "latency_seconds_p50": latencies[int(0.50 * (total - 1))],
        "latency_seconds_p95": latencies[int(0.95 * (total - 1))],
        "usage": usage,
    }
    (OUTPUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "test_metrics_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    with (OUTPUT / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted", *labels])
        for label, values in zip(labels, confusion):
            writer.writerow([label, *values])

    comparison = [
        ("Qwen3-0.6B LoRA (2 epochs)", 0.8506493506493507, 0.8508984605730113,
         0.8508984605730114, 38_812, 0, 38_812),
        ("Qwen3.5-0.8B LoRA (2 epochs)", 0.8688311688311688, 0.8688787846760082,
         0.8688787846760081, 38_779, 0, 38_779),
        (metrics["model"], accuracy, macro_f1, weighted_f1,
         usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0)),
    ]
    lines = [
        "# Banking77 test comparison", "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Input tokens | Output tokens | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, acc, macro, weighted, input_tokens, output_tokens, total_tokens in comparison:
        lines.append(
            f"| {model} | {acc:.4f} | {macro:.4f} | {weighted:.4f} | "
            f"{input_tokens:,} | {output_tokens:,} | {total_tokens:,} |"
        )
    lines.extend([
        "", f"Foundry predictions: {total}/{total} completed.", "",
        "Token methodology:", "",
        "- Qwen input tokens were recalculated over all 3,080 test texts with each checkpoint's "
        "local tokenizer, special tokens enabled, truncation at 128, and no padding. These "
        "sequence-classification models return logits rather than generated tokens, so output tokens are zero.",
        "- GPT-5.6 Luna values are the API-reported usage totals. Each independent request includes "
        "the classification instructions and the full list of 77 valid labels, in addition to the test text.",
        "- These figures cover test inference only; they do not include Qwen fine-tuning tokens.", "",
    ])
    (ROOT / "banking77_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
