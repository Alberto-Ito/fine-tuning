import numpy as np


def classification_metrics(predictions, labels, label_names):
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    for expected, predicted in zip(labels, predictions):
        confusion[int(expected), int(predicted)] += 1
    per_class = []
    for label_id, label in enumerate(label_names):
        tp = int(confusion[label_id, label_id])
        support = int(confusion[label_id, :].sum())
        predicted_count = int(confusion[:, label_id].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label_id": label_id, "label": label, "precision": precision,
                          "recall": recall, "f1": f1, "support": support})
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class, "confusion_matrix": confusion.tolist(),
    }


def compute_metrics(result, label_names):
    details = classification_metrics(np.argmax(result[0], axis=-1), result[1], label_names)
    return {key: details[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}

