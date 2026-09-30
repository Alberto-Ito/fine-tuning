"""Re-evaluate a trained Banking77 LoRA adapter without retraining it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from train_banking77 import classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "banking77_1epoch.yaml",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "banking77" / "final_model" / "one_epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_yaml(args.config.resolve())
    dataset_config = load_yaml(PROJECT_ROOT / experiment["dataset_config"])
    metrics_dir = PROJECT_ROOT / experiment["metrics_dir"]
    report_path = PROJECT_ROOT / experiment["report_dir"] / "banking77_one_epoch.md"

    raw = load_dataset("csv", data_files=dataset_config["data_files"])
    raw = raw.rename_column(
        dataset_config["source_label_column"], dataset_config["label_column"]
    ).class_encode_column(dataset_config["label_column"])
    test_dataset = raw[dataset_config["test_split"]]
    label_feature = test_dataset.features[dataset_config["label_column"]]
    label_names = list(label_feature.names)
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.adapter.resolve(), use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        experiment["model_name"],
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.adapter.resolve())

    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        encoded = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=int(experiment["max_length"]),
            padding="max_length",
        )
        encoded["labels"] = batch[label_column]
        return encoded

    tokenized = test_dataset.map(
        tokenize, batched=True, remove_columns=test_dataset.column_names
    )

    def compute_metrics(result) -> Dict[str, float]:
        logits, labels = result
        predictions = np.argmax(logits, axis=-1)
        details = classification_metrics(predictions, labels, label_names)
        return {
            key: details[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(metrics_dir / "evaluation_tmp"),
            per_device_eval_batch_size=int(experiment["per_device_eval_batch_size"]),
            report_to="none",
            dataloader_num_workers=0,
        ),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    result = trainer.predict(tokenized, metric_key_prefix="test")
    updated = {
        "loss": float(result.metrics["test_loss"]),
        "accuracy": float(result.metrics["test_accuracy"]),
        "macro_precision": float(result.metrics["test_macro_precision"]),
        "macro_recall": float(result.metrics["test_macro_recall"]),
        "macro_f1": float(result.metrics["test_macro_f1"]),
    }
    metrics_path = metrics_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = report_path.read_text(encoding="utf-8")
    marker = "- Test accuracy:"
    loss_line = f"- Test loss: {updated['loss']:.4f}\n"
    if "- Test loss:" not in report:
        report = report.replace(marker, loss_line + marker)
    report_path.write_text(report, encoding="utf-8")

    summary_path = metrics_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads((metrics_dir / "trainer_log_history.json").read_text(encoding="utf-8"))
    logged_train_losses = [float(record["loss"]) for record in history if "loss" in record]
    summary["test_metrics"] = updated
    summary["losses"] = {
        "train_average": float(summary["train_metrics"]["train_loss"]),
        "train_final_logged": logged_train_losses[-1],
        "validation": float(summary["validation_metrics"]["validation_loss"]),
        "test": updated["loss"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(updated, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
