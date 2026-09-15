"""Measure exact JSON sentiment accuracy on the held-out MLX-LM test split."""

import argparse
import json
from collections import Counter
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


HERE = Path(__file__).resolve().parent
MODEL = "mlx-community/llama2-13b-qnt4bit"
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
    parser.add_argument("--baseline", action="store_true", help="Evaluate without the trained adapter")
    parser.add_argument("--limit", type=int, default=60, help="Number of held-out examples to run")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    adapter = None if args.baseline else str(HERE / "adapters")
    model, tokenizer = load(MODEL, adapter_path=adapter)
    correct = 0
    invalid = 0
    confusion: Counter[tuple[str, str]] = Counter()
    evaluated = 0
    with (HERE / "data" / "test.jsonl").open(encoding="utf-8") as dataset:
        for line in dataset:
            if evaluated >= args.limit:
                break
            messages = json.loads(line)["messages"]
            truth = parse_answer(messages[-1]["content"])
            if truth is None:
                raise ValueError("Invalid assistant answer in test dataset")
            prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
            generated = generate(
                model, tokenizer, prompt=prompt, max_tokens=32,
                sampler=make_sampler(temp=0.0), verbose=False,
            )
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
