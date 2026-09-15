"""Convert the Fireworks RFT sentiment dataset to MLX-LM chat SFT data."""

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
OUTPUT = HERE / "data"
LABELS = {"positivo", "negativo", "neutro"}
SPLITS = {
    "train": [SOURCE / "rft-train-10000.jsonl"],
    "valid": [SOURCE / "rft-eval.jsonl"],
    "test": [SOURCE / "rft-eval-sarcasm.jsonl", SOURCE / "rft-eval-short.jsonl"],
}


def convert(record: dict, location: str) -> dict:
    messages = record.get("messages")
    label = record.get("ground_truth")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location}: messages must be a nonempty list")
    if label not in LABELS:
        raise ValueError(f"{location}: invalid ground_truth {label!r}")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError(f"{location}: expected system/user prompt messages")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{location}: every message needs text content")
    answer = json.dumps({"sentimiento": label}, ensure_ascii=False, separators=(",", ":"))
    return {"messages": [*messages, {"role": "assistant", "content": answer}]}


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    seen_prompts: dict[str, str] = {}
    for split, files in SPLITS.items():
        counts: Counter[str] = Counter()
        destination = OUTPUT / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for source in files:
                with source.open(encoding="utf-8") as input_file:
                    for line_number, line in enumerate(input_file, 1):
                        if not line.strip():
                            raise ValueError(f"{source}:{line_number}: empty row")
                        location = f"{source}:{line_number}"
                        record = json.loads(line)
                        converted = convert(record, location)
                        prompt = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
                        previous = seen_prompts.get(prompt)
                        if previous is not None and previous != split:
                            raise ValueError(f"{location}: prompt also appears in {previous}")
                        seen_prompts[prompt] = split
                        counts[record["ground_truth"]] += 1
                        output.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")))
                        output.write("\n")
        print(f"{destination}: {sum(counts.values())} rows, {dict(counts)}")


if __name__ == "__main__":
    main()
