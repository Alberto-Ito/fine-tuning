import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import classification_metrics


def write_test_artifacts(test_dataset, result, label_names, paths, filename):
    logits, labels = result.predictions, result.label_ids
    predictions = np.argmax(logits, axis=-1)
    details = classification_metrics(predictions, labels, label_names)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    prediction_path = paths["predictions"] / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for index, example in enumerate(test_dataset):
            expected, predicted = int(labels[index]), int(predictions[index])
            stream.write(json.dumps({"index": index, "text": example["text"],
                "expected_label_id": expected, "expected_label": label_names[expected],
                "predicted_label_id": predicted, "predicted_label": label_names[predicted],
                "confidence": float(confidences[index]), "correct": expected == predicted},
                ensure_ascii=False) + "\n")
    aggregate = {key: details[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    aggregate["loss"] = float(result.metrics["test_loss"])
    (paths["metrics"] / "test_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (paths["metrics"] / "test_metrics_per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label_id", "label", "precision", "recall", "f1", "support"])
        writer.writeheader(); writer.writerows(details["per_class"])
    with (paths["reports"] / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["expected\\predicted", *label_names])
        for label, row in zip(label_names, details["confusion_matrix"]): writer.writerow([label, *row])
    return aggregate

