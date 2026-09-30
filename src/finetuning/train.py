"""Train any configured sequence-classification model with LoRA."""

import argparse
import json
import os
import time

import torch
from transformers import Trainer, TrainingArguments, set_seed

from .common.artifacts import write_test_artifacts
from .common.callbacks import ResourceCallback
from .common.config import load_yaml, project_path, resolve_config_path
from .common.data import prepare_datasets
from .common.evaluation import compute_metrics
from .common.models import load_classifier, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = project_path(args.config)
    experiment = load_yaml(config_path)
    dataset = load_yaml(resolve_config_path(experiment["dataset_config"], config_path))
    set_seed(int(experiment.get("seed", 42)))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    tokenizer = load_tokenizer(experiment["model_name"])
    datasets, tokenized, label_names = prepare_datasets(dataset, tokenizer, int(experiment["max_length"]))
    model = load_classifier(experiment, label_names, tokenizer)
    paths = {
        "output": project_path(experiment.get("output_dir", "outputs/checkpoints")),
        "final_model": project_path(experiment.get("final_model_dir", "outputs/model")),
        "logging": project_path(experiment.get("logging_dir", "outputs/logs")),
        "metrics": project_path(experiment.get("metrics_dir", "outputs/metrics")),
        "predictions": project_path(experiment.get("predictions_dir", "outputs/predictions")),
        "reports": project_path(experiment.get("report_dir", "reports")),
    }
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    callback = ResourceCallback(paths["metrics"] / "resources.jsonl")
    callback.sample(phase="model_loaded")
    training = {"output_dir": str(paths["output"]), "run_name": experiment["run_name"],
        "max_steps": int(experiment.get("max_steps", -1)), "num_train_epochs": float(experiment["num_train_epochs"]),
        "per_device_train_batch_size": int(experiment["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(experiment["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(experiment["gradient_accumulation_steps"]),
        "learning_rate": float(experiment["learning_rate"]), "weight_decay": float(experiment["weight_decay"]),
        "warmup_ratio": float(experiment.get("warmup_ratio", 0.0)), "lr_scheduler_type": experiment.get("lr_scheduler_type", "cosine"),
        "eval_strategy": "steps", "eval_steps": int(experiment["eval_steps"]), "save_strategy": "steps",
        "save_steps": int(experiment["save_steps"]), "save_total_limit": int(experiment["save_total_limit"]),
        "logging_strategy": "steps", "logging_steps": int(experiment["logging_steps"]), "logging_dir": str(paths["logging"]),
        "load_best_model_at_end": True, "metric_for_best_model": "macro_f1", "greater_is_better": True,
        "optim": "adamw_torch", "fp16": False, "bf16": False, "report_to": "none", "seed": int(experiment.get("seed", 42)),
        "data_seed": int(experiment.get("seed", 42)), "dataloader_num_workers": 0, "remove_unused_columns": True,
        "label_names": ["labels"]}
    trainer = Trainer(model=model, args=TrainingArguments(**training), train_dataset=tokenized["train"],
                      eval_dataset=tokenized["validation"], processing_class=tokenizer,
                      compute_metrics=lambda result: compute_metrics(result, label_names), callbacks=[callback])
    started = time.perf_counter(); result = trainer.train(); elapsed = time.perf_counter() - started
    validation = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    test_result = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test = write_test_artifacts(datasets["test"], test_result, label_names, paths,
                                experiment.get("confusion_matrix_filename", "confusion_matrix.csv"))
    trainer.save_model(str(paths["final_model"])); tokenizer.save_pretrained(str(paths["final_model"]))
    summary = {"run_name": experiment["run_name"], "model": experiment["model_name"], "method": "LoRA",
        "train_examples": len(tokenized["train"]), "validation_examples": len(tokenized["validation"]),
        "test_examples": len(tokenized["test"]), "global_steps": trainer.state.global_step,
        "train_loop_seconds": elapsed, "train_metrics": result.metrics, "validation_metrics": validation, "test_metrics": test}
    (paths["metrics"] / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (paths["metrics"] / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
