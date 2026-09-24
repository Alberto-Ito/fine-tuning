"""Train Qwen3-0.6B for Banking77 classification with HF Trainer and PEFT LoRA."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psutil
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "banking77_pilot.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def macro_f1(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    scores = []
    for label_id in range(num_labels):
        predicted = predictions == label_id
        expected = labels == label_id
        true_positive = int(np.logical_and(predicted, expected).sum())
        false_positive = int(np.logical_and(predicted, np.logical_not(expected)).sum())
        false_negative = int(np.logical_and(np.logical_not(predicted), expected).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


class ResourceCallback(TrainerCallback):
    """Record process and MPS memory alongside Trainer events."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.started_at = time.perf_counter()
        self.process = psutil.Process()

    def _memory(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "rss_gb": round(self.process.memory_info().rss / 1024**3, 3),
        }
        if torch.backends.mps.is_available():
            result["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1024**3, 3)
            result["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1024**3, 3)
        return result

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {
            "step": state.global_step,
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 2),
            **self._memory(),
            **logs,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = args.config.resolve()
    experiment = load_yaml(experiment_path)
    dataset_config = load_yaml(resolve_project_path(experiment["dataset_config"]))

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; this training run requires Apple Silicon MPS.")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    seed = int(experiment["seed"])
    set_seed(seed)

    if dataset_config.get("loader") == "csv":
        raw = load_dataset("csv", data_files=dataset_config["data_files"])
        source_label = dataset_config.get("source_label_column")
        if source_label and source_label != dataset_config["label_column"]:
            raw = raw.rename_column(source_label, dataset_config["label_column"])
        raw = raw.class_encode_column(dataset_config["label_column"])
    else:
        raw = load_dataset(dataset_config["huggingface_id"])
    split = raw[dataset_config["train_split"]].train_test_split(
        test_size=float(dataset_config["validation_fraction"]),
        seed=int(dataset_config["seed"]),
        stratify_by_column=dataset_config["label_column"],
    )
    datasets = DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw[dataset_config["test_split"]],
    )

    label_feature = datasets["train"].features[dataset_config["label_column"]]
    label_names = list(label_feature.names)
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(experiment["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    tokenized = datasets.map(tokenize, batched=True, remove_columns=datasets["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        experiment["model_name"],
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(experiment["lora_rank"]),
        lora_alpha=int(experiment["lora_alpha"]),
        lora_dropout=float(experiment["lora_dropout"]),
        bias="none",
        target_modules=list(experiment["lora_target_modules"]),
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = resolve_project_path(experiment["output_dir"])
    final_model_dir = resolve_project_path(experiment["final_model_dir"])
    logging_dir = resolve_project_path(experiment["logging_dir"])
    metrics_dir = resolve_project_path(experiment["metrics_dir"])
    for directory in (output_dir, final_model_dir, logging_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    resource_log = metrics_dir / "resources.jsonl"
    resource_log.unlink(missing_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=experiment["run_name"],
        max_steps=int(experiment["max_steps"]),
        num_train_epochs=float(experiment["num_train_epochs"]),
        per_device_train_batch_size=int(experiment["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(experiment["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(experiment["gradient_accumulation_steps"]),
        learning_rate=float(experiment["learning_rate"]),
        weight_decay=float(experiment["weight_decay"]),
        warmup_ratio=float(experiment["warmup_ratio"]),
        lr_scheduler_type=experiment["lr_scheduler_type"],
        eval_strategy="steps",
        eval_steps=int(experiment["eval_steps"]),
        save_strategy="steps",
        save_steps=int(experiment["save_steps"]),
        save_total_limit=int(experiment["save_total_limit"]),
        logging_strategy="steps",
        logging_steps=int(experiment["logging_steps"]),
        logging_dir=str(logging_dir),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    def compute_metrics(result) -> Dict[str, float]:
        logits, labels = result
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float((predictions == labels).mean()),
            "macro_f1": macro_f1(predictions, labels, len(label_names)),
        }

    callback = ResourceCallback(resource_log)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    started_at = time.perf_counter()
    train_result = trainer.train()
    elapsed_seconds = time.perf_counter() - started_at
    validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    evaluation_seconds_during_train = sum(
        float(record.get("eval_runtime", 0.0)) for record in trainer.state.log_history
    )
    optimization_seconds = max(elapsed_seconds - evaluation_seconds_during_train, 0.0)

    summary = {
        "run_name": experiment["run_name"],
        "model": experiment["model_name"],
        "method": "LoRA",
        "device": "mps",
        "dtype": "bfloat16",
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "test_examples": len(tokenized["test"]),
        "num_labels": len(label_names),
        "global_steps": trainer.state.global_step,
        "effective_batch_size": (
            int(experiment["per_device_train_batch_size"])
            * int(experiment["gradient_accumulation_steps"])
        ),
        "examples_seen_approx": (
            trainer.state.global_step
            * int(experiment["per_device_train_batch_size"])
            * int(experiment["gradient_accumulation_steps"])
        ),
        "train_loop_seconds_including_evaluation": round(elapsed_seconds, 2),
        "evaluation_seconds_during_train": round(evaluation_seconds_during_train, 2),
        "estimated_optimization_seconds": round(optimization_seconds, 2),
        "estimated_optimization_seconds_per_step": round(
            optimization_seconds / max(trainer.state.global_step, 1), 3
        ),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "macos": platform.mac_ver()[0],
        },
    }
    with (metrics_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (metrics_dir / "trainer_log_history.json").open("w", encoding="utf-8") as stream:
        json.dump(trainer.state.log_history, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
