"""Evaluate a saved LoRA adapter on the configured test split."""

import argparse
import json

import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from .common.config import load_yaml, project_path, resolve_config_path
from .common.data import load_raw_dataset
from .common.evaluation import compute_metrics
from .common.models import load_tokenizer
from .common.models import load_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--adapter", required=True)
    args = parser.parse_args(); config_path = project_path(args.config)
    experiment = load_yaml(config_path); dataset_config = load_yaml(resolve_config_path(experiment["dataset_config"], config_path))
    raw = load_raw_dataset(dataset_config); test = raw[dataset_config["test_split"]]
    tokenizer = load_tokenizer(experiment["model_name"])
    label_names = list(test.features[dataset_config["label_column"]].names)
    model = load_classifier(experiment, label_names, tokenizer, apply_lora=False)
    model = PeftModel.from_pretrained(model, project_path(args.adapter))
    text, label = dataset_config["text_column"], dataset_config["label_column"]
    tokenized = test.map(lambda batch: {**tokenizer(batch[text], truncation=True, max_length=int(experiment["max_length"]), padding="max_length"), "labels": batch[label]}, batched=True, remove_columns=test.column_names)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="/tmp/finetuning-evaluate", per_device_eval_batch_size=int(experiment["per_device_eval_batch_size"]), report_to="none"), processing_class=tokenizer, compute_metrics=lambda result: compute_metrics(result, label_names))
    result = trainer.predict(tokenized, metric_key_prefix="test")
    metrics = {key.removeprefix("test_"): float(value) for key, value in result.metrics.items() if key.startswith("test_")}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
