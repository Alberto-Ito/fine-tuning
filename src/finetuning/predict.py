"""Run predictions with a saved LoRA adapter."""

import argparse
import json

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

from .common.config import load_yaml, project_path
from .common.models import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True); parser.add_argument("--text", nargs="+"); parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args(); adapter = project_path(args.model)
    tokenizer = load_tokenizer(str(adapter))
    base_name = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    adapter_weights = load_file(str(adapter / "adapter_model.safetensors"))
    score_weights = next(value for key, value in adapter_weights.items() if "score" in key and value.ndim == 2)
    num_labels = int(score_weights.shape[0])
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=num_labels, dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, adapter); model.eval()
    if args.text:
        texts = args.text
    elif args.input:
        with open(args.input, encoding="utf-8") as stream:
            texts = [json.loads(line)["text"] for line in stream]
    else:
        raise SystemExit("Provide --text or --input")
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        predictions = model(**encoded).logits.argmax(dim=-1).tolist()
    rows = [{"text": text, "label_id": label} for text, label in zip(texts, predictions)]
    rendered = json.dumps(rows, indent=2, ensure_ascii=False)
    if args.output: project_path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else: print(rendered)


if __name__ == "__main__":
    main()
