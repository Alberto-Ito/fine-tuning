"""Fine-tune Qwen3.5-0.8B on Banking77 and record quality/operational metrics."""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/banking77_2epochs.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classification_metrics(predictions, labels, label_names) -> Dict[str, Any]:
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
        per_class.append({
            "label_id": label_id, "label": label, "precision": precision,
            "recall": recall, "f1": f1, "support": support,
        })
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def memory_snapshot(process: psutil.Process) -> Dict[str, float]:
    result = {"rss_gb": process.memory_info().rss / 1024**3}
    if torch.backends.mps.is_available():
        result.update({
            "mps_allocated_gb": torch.mps.current_allocated_memory() / 1024**3,
            "mps_driver_gb": torch.mps.driver_allocated_memory() / 1024**3,
        })
    return result


class ResourceCallback(TrainerCallback):
    """Persist resource samples and retain observed memory peaks."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.started_at = time.perf_counter()
        self.process = psutil.Process()
        self.peaks: Dict[str, float] = {}

    def sample(self, **extra) -> Dict[str, Any]:
        memory = memory_snapshot(self.process)
        for key, value in memory.items():
            self.peaks[key] = max(self.peaks.get(key, 0.0), value)
        record = {
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 3),
            **{key: round(value, 3) for key, value in memory.items()},
            **extra,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.sample(phase="trainer", step=state.global_step, **logs)


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_inference(model, dataset, batch_size: int, warmup_batches: int,
                        resource_callback: ResourceCallback) -> Dict[str, float]:
    """Measure batched classification inference over the test split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator, num_workers=0)
    device = next(model.parameters()).device
    model.eval()
    durations, request_tokens = [], []
    measured_requests = measured_tokens = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            token_counts = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
            synchronize()
            started = time.perf_counter()
            model(**batch)
            synchronize()
            elapsed = time.perf_counter() - started
            resource_callback.sample(phase="benchmark", batch=index, batch_seconds=elapsed)
            if index >= warmup_batches:
                durations.append(elapsed)
                request_tokens.extend(int(value) for value in token_counts)
                measured_requests += len(token_counts)
                measured_tokens += sum(token_counts)
    total_seconds = sum(durations)
    if not durations or not measured_requests:
        raise RuntimeError("Operational benchmark did not collect measured batches")
    return {
        "scope": "batched test inference after warmup",
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "measured_batches": len(durations),
        "measured_requests": measured_requests,
        "input_tokens": measured_tokens,
        "input_tokens_per_request_mean": float(np.mean(request_tokens)),
        "input_tokens_per_request_p95": float(np.percentile(request_tokens, 95)),
        "requests_per_second": measured_requests / total_seconds,
        "input_tokens_per_second": measured_tokens / total_seconds,
        "batch_latency_ms_p50": float(np.percentile(durations, 50) * 1000),
        "batch_latency_ms_p95": float(np.percentile(durations, 95) * 1000),
        "benchmark_seconds": total_seconds,
    }


def write_test_artifacts(test_dataset, result, label_names, predictions_dir,
                         metrics_dir, report_dir, confusion_filename):
    logits, labels = result.predictions, result.label_ids
    predictions = np.argmax(logits, axis=-1)
    details = classification_metrics(predictions, labels, label_names)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    with (predictions_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, example in enumerate(test_dataset):
            expected, predicted = int(labels[index]), int(predictions[index])
            stream.write(json.dumps({
                "index": index, "text": example["text"],
                "expected_label_id": expected, "expected_label": label_names[expected],
                "predicted_label_id": predicted, "predicted_label": label_names[predicted],
                "confidence": float(confidences[index]), "correct": expected == predicted,
            }, ensure_ascii=False) + "\n")
    aggregate = {key: details[key] for key in
                 ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    aggregate["loss"] = float(result.metrics["test_loss"])
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (metrics_dir / "test_metrics_per_class.csv").open(
        "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "label_id", "label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(details["per_class"])
    with (report_dir / confusion_filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["expected\\predicted", *label_names])
        for label, row in zip(label_names, details["confusion_matrix"]):
            writer.writerow([label, *row])
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_yaml(args.config.resolve())
    dataset_config = load_yaml(resolve_project_path(experiment["dataset_config"]))
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; this training run requires Apple Silicon MPS.")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    seed = int(experiment["seed"])
    set_seed(seed)

    raw = load_dataset("csv", data_files=dataset_config["data_files"])
    raw = raw.rename_column(dataset_config["source_label_column"],
                            dataset_config["label_column"])
    raw = raw.class_encode_column(dataset_config["label_column"])
    split = raw[dataset_config["train_split"]].train_test_split(
        test_size=float(dataset_config["validation_fraction"]), seed=dataset_config["seed"],
        stratify_by_column=dataset_config["label_column"])
    datasets = DatasetDict(train=split["train"], validation=split["test"],
                           test=raw[dataset_config["test_split"]])
    label_names = list(datasets["train"].features[dataset_config["label_column"]].names)
    id2label = dict(enumerate(label_names))
    label2id = {label: index for index, label in id2label.items()}

    model_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(experiment["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        experiment["model_name"], num_labels=len(label_names), id2label=id2label,
        label2id=label2id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model_load_seconds = time.perf_counter() - model_started
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    text_column, label_column = dataset_config["text_column"], dataset_config["label_column"]
    def tokenize(batch):
        encoded = tokenizer(batch[text_column], truncation=True,
                            max_length=int(experiment["max_length"]), padding="max_length")
        encoded["labels"] = batch[label_column]
        return encoded
    tokenized = datasets.map(tokenize, batched=True,
                             remove_columns=datasets["train"].column_names)

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=int(experiment["lora_rank"]),
        lora_alpha=int(experiment["lora_alpha"]),
        lora_dropout=float(experiment["lora_dropout"]), bias="none",
        target_modules=list(experiment["lora_target_modules"]), modules_to_save=["score"]))
    model.print_trainable_parameters()

    paths = {key: resolve_project_path(experiment[key]) for key in (
        "output_dir", "final_model_dir", "logging_dir", "metrics_dir",
        "predictions_dir", "report_dir")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    resource_log = paths["metrics_dir"] / "resources.jsonl"
    resource_log.unlink(missing_ok=True)
    callback = ResourceCallback(resource_log)
    callback.sample(phase="model_loaded")

    training_args = TrainingArguments(
        output_dir=str(paths["output_dir"]), run_name=experiment["run_name"],
        max_steps=int(experiment["max_steps"]),
        num_train_epochs=float(experiment["num_train_epochs"]),
        per_device_train_batch_size=int(experiment["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(experiment["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(experiment["gradient_accumulation_steps"]),
        learning_rate=float(experiment["learning_rate"]),
        weight_decay=float(experiment["weight_decay"]),
        # Transformers 5 accepts a fractional value in warmup_steps.
        warmup_steps=float(experiment["warmup_ratio"]),
        lr_scheduler_type=experiment["lr_scheduler_type"],
        eval_strategy="steps", eval_steps=int(experiment["eval_steps"]),
        save_strategy="steps", save_steps=int(experiment["save_steps"]),
        save_total_limit=int(experiment["save_total_limit"]),
        logging_strategy="steps", logging_steps=int(experiment["logging_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        optim="adamw_torch", fp16=False, bf16=False, report_to="none", seed=seed,
        data_seed=seed, dataloader_num_workers=0, remove_unused_columns=True,
        label_names=["labels"])

    def compute_metrics(result):
        logits, labels = result
        details = classification_metrics(np.argmax(logits, axis=-1), labels, label_names)
        return {key: details[key] for key in
                ("accuracy", "macro_precision", "macro_recall", "macro_f1")}

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"], processing_class=tokenizer,
                      compute_metrics=compute_metrics, callbacks=[callback])
    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    validation_metrics = trainer.evaluate(tokenized["validation"],
                                          metric_key_prefix="validation")
    test_result = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_metrics = write_test_artifacts(
        datasets["test"], test_result, label_names, paths["predictions_dir"],
        paths["metrics_dir"], paths["report_dir"],
        experiment["confusion_matrix_filename"])
    operational = benchmark_inference(
        trainer.model, tokenized["test"], int(experiment["benchmark_batch_size"]),
        int(experiment["benchmark_warmup_batches"]), callback)
    operational["model_load_seconds"] = model_load_seconds
    operational["peak_memory_gb"] = {key: round(value, 3)
                                     for key, value in callback.peaks.items()}
    (paths["metrics_dir"] / "operational_metrics.json").write_text(
        json.dumps(operational, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    trainer.save_model(str(paths["final_model_dir"]))
    tokenizer.save_pretrained(str(paths["final_model_dir"]))
    history = trainer.state.log_history
    losses = [float(row["loss"]) for row in history if "loss" in row]
    summary = {
        "run_name": experiment["run_name"], "model": experiment["model_name"],
        "method": "LoRA", "device": "mps", "dtype": "bfloat16",
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "test_examples": len(tokenized["test"]), "num_labels": len(label_names),
        "global_steps": trainer.state.global_step,
        "effective_batch_size": int(experiment["per_device_train_batch_size"]) *
                                int(experiment["gradient_accumulation_steps"]),
        "train_loop_seconds_including_evaluation": train_seconds,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "operational_metrics": operational,
        "losses": {"train_average": float(train_result.training_loss),
                   "train_final_logged": losses[-1],
                   "validation": float(validation_metrics["validation_loss"]),
                   "test": float(test_metrics["loss"])},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "macos": platform.mac_ver()[0]},
    }
    (paths["metrics_dir"] / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (paths["metrics_dir"] / "trainer_log_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    peak = operational["peak_memory_gb"]
    report = [experiment["report_title"], "",
              f"- Base model: `{experiment['model_name']}`",
              f"- Best checkpoint: `{trainer.state.best_model_checkpoint}`",
              f"- Completed steps: {trainer.state.global_step}",
              f"- Test loss: {test_metrics['loss']:.4f}",
              f"- Test accuracy: {test_metrics['accuracy']:.4f}",
              f"- Test macro-F1: {test_metrics['macro_f1']:.4f}", "",
              "## Operational metrics", "",
              f"- Input tokens/request (mean): {operational['input_tokens_per_request_mean']:.2f}",
              f"- Requests/s: {operational['requests_per_second']:.2f}",
              f"- Input tokens/s: {operational['input_tokens_per_second']:.2f}",
              f"- Batch latency p50/p95: {operational['batch_latency_ms_p50']:.2f} / {operational['batch_latency_ms_p95']:.2f} ms",
              f"- Model load time: {model_load_seconds:.2f} s",
              f"- Peak process RSS: {peak.get('rss_gb', 0):.3f} GB",
              f"- Peak MPS allocated/driver: {peak.get('mps_allocated_gb', 0):.3f} / {peak.get('mps_driver_gb', 0):.3f} GB", "",
              f"Full metrics: `{experiment['metrics_dir']}/run_summary.json`."]
    (paths["report_dir"] / experiment["report_filename"]).write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
