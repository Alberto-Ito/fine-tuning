"""Measure exact JSON sentiment accuracy for a Hugging Face PEFT checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = {"positivo", "negativo", "neutro"}


def parse_answer(text: str) -> str | None:
    try:
        answer = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(answer, dict) and set(answer) == {"sentimiento"}:
        label = answer["sentimiento"]
        return label if isinstance(label, str) and label in LABELS else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--checkpoint", type=Path, help="PEFT checkpoint directory")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.baseline and args.checkpoint is None:
        parser.error("--checkpoint is required unless --baseline is used")
    if args.baseline and args.checkpoint is not None:
        parser.error("--checkpoint cannot be combined with --baseline")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("mps")
    model = base if args.baseline else PeftModel.from_pretrained(base, str(args.checkpoint))
    model.eval()

    correct = invalid = evaluated = 0
    confusion: Counter[tuple[str, str]] = Counter()
    with (DATA / "test.jsonl").open(encoding="utf-8") as dataset, torch.inference_mode():
        for line in dataset:
            if evaluated >= args.limit:
                break
            messages = json.loads(line)["messages"]
            truth = parse_answer(messages[-1]["content"])
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            predicted = parse_answer(generated)
            evaluated += 1
            correct += predicted == truth
            invalid += predicted is None
            confusion[truth, predicted or "inválido"] += 1
            print(f"{evaluated:02d}. esperado={truth}, obtenido={predicted or 'inválido'}; {generated!r}")

    print(f"Precisión exacta: {correct}/{evaluated} = {correct / evaluated:.1%}")
    print(f"Respuestas JSON inválidas: {invalid}")
    print("Confusión (esperado -> obtenido):")
    for (truth, predicted), count in sorted(confusion.items()):
        print(f"  {truth} -> {predicted}: {count}")


if __name__ == "__main__":
    main()
