"""LoRA/SFT training with Hugging Face Transformers, PEFT, PyTorch, and MPS."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_STEPS = 4000


class ChatDataset(Dataset):
    """Tokenized chat data whose loss only covers the assistant response."""

    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows: list[dict[str, list[int]]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                messages = record["messages"]
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True
                )
                full_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise ValueError(f"{path}:{line_number}: chat template prefix mismatch")
                full_ids = full_ids[:max_length]
                prompt_length = min(len(prompt_ids), len(full_ids))
                labels = [-100] * prompt_length + full_ids[prompt_length:]
                if not any(label != -100 for label in labels):
                    raise ValueError(f"{path}:{line_number}: assistant answer was truncated")
                self.rows.append(
                    {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class ChatCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        def tensors(key: str) -> list[torch.Tensor]:
            return [torch.tensor(row[key], dtype=torch.long) for row in features]

        return {
            "input_ids": pad_sequence(tensors("input_ids"), batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(tensors("attention_mask"), batch_first=True, padding_value=0),
            "labels": pad_sequence(tensors("labels"), batch_first=True, padding_value=-100),
        }


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            with self.destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, **logs}, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=HERE / "adapters-4000")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.max_steps > MAX_STEPS:
        raise SystemExit(f"--max-steps must be between 1 and {MAX_STEPS}; this run must stop at 4000")
    if args.batch_size != 4:
        raise SystemExit("This reproduction requires --batch-size 4")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable. Run on Apple Silicon with an MPS-enabled PyTorch build.")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = ChatDataset(DATA / "train.jsonl", tokenizer, args.max_seq_length)
    valid_data = ChatDataset(DATA / "valid.jsonl", tokenizer, args.max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    layer_count = model.config.num_hidden_layers
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=160,  # alpha/r = 20, matching the MLX LoRA scale
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(layer_count - 4, layer_count)),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output_dir = args.output_dir.resolve()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=None,
        logging_strategy="steps",
        logging_steps=50,
        seed=0,
        data_seed=0,
        # Accelerate 1.10 rejects its CUDA-oriented fp16 autocast flag on MPS.
        # The model itself is already loaded in float16 above.
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    metrics_file = output_dir / "metrics.jsonl"
    if not args.resume_from_checkpoint:
        metrics_file.unlink(missing_ok=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=ChatCollator(tokenizer.pad_token_id),
        callbacks=[JsonlMetricsCallback(metrics_file)],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    final_eval = trainer.evaluate()
    summary = {
        "model": args.model,
        "steps": trainer.state.global_step,
        "batch_size": args.batch_size,
        "examples_seen": trainer.state.global_step * args.batch_size,
        "epochs": trainer.state.global_step * args.batch_size / len(train_data),
        "train_loss": result.training_loss,
        "eval_loss": final_eval["eval_loss"],
        "eval_perplexity": math.exp(final_eval["eval_loss"]),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
