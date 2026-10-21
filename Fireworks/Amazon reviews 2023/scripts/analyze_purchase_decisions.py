#!/usr/bin/env python3
"""Extract purchase labels and report deterministic classification metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABELS = ("BUY", "CONDITIONAL_BUY", "DO_NOT_BUY", "INSUFFICIENT_EVIDENCE")
LABEL_RE = re.compile(r"(?<![A-Z_])(INSUFFICIENT_EVIDENCE|CONDITIONAL_BUY|DO_NOT_BUY|BUY)(?![A-Z_])", re.I)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def label(text: str) -> str | None:
    match = LABEL_RE.search(text.upper())
    return match.group(1).upper() if match else None


def metrics(reference: dict[str, str | None], predictions: Path) -> tuple[str, dict]:
    prediction_rows = list(rows(predictions))
    model = prediction_rows[0].get("model", predictions.stem)
    predicted = {row["id"]: label(row.get("prediction", "")) for row in prediction_rows}
    confusion = {actual: {guess: 0 for guess in (*LABELS, "MISSING")} for actual in LABELS}
    for item_id, actual in reference.items():
        if actual in LABELS:
            confusion[actual][predicted.get(item_id) or "MISSING"] += 1
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[item][item] for item in LABELS)
    per_class = {}
    for item in LABELS:
        tp = confusion[item][item]
        fp = sum(confusion[actual][item] for actual in LABELS if actual != item)
        fn = sum(value for guess, value in confusion[item].items() if guess != item)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[item].values())
        per_class[item] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": support}
    macro_f1 = sum(value["f1"] for value in per_class.values()) / len(LABELS)
    weighted_f1 = sum(value["f1"] * value["support"] for value in per_class.values()) / total
    return model, {
        "n": total,
        "accuracy": round(correct / total, 4) if total else None,
        "macro_f1": round(macro_f1, 4) if total else None,
        "weighted_f1": round(weighted_f1, 4) if total else None,
        "missing_labels": sum(1 for value in predicted.values() if value is None),
        "prediction_distribution": dict(Counter(value or "MISSING" for value in predicted.values())),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = {row["id"]: label(row["reference"]) for row in rows(args.test)}
    report = {
        "labels": list(LABELS),
        "reference_distribution": dict(Counter(value or "MISSING" for value in reference.values())),
        "models": {},
    }
    for path in args.predictions:
        model, result = metrics(reference, path)
        report["models"][model] = result
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
