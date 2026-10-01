#!/usr/bin/env python3
"""Evaluate Qwen3-0.6B zero-shot on the complete Banking77 test split."""
from pathlib import Path
import json, re
from collections import Counter
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"

def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS no está disponible; se cancela para no ejecutar la evaluación en CPU.")
    device = torch.device("mps")
    ds = load_dataset("PolyAI/banking77", split="test")
    labels = ds.features["label"].names
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True).to(device)
    model.eval()
    label_text = "\n".join(f"{i}: {name}" for i, name in enumerate(labels))
    y, pred, rows = [], [], []; input_tokens = output_tokens = 0
    for n, item in enumerate(ds):
        prompt = ("Classify the banking customer message into exactly one label. "
                  "Return only the numeric label ID.\nLabels:\n" + label_text +
                  "\nMessage: " + item["text"] + "\nLabel ID:")
        encoded = tok(prompt, return_tensors="pt").to(device)
        input_tokens += int(encoded.input_ids.shape[1])
        with torch.inference_mode():
            out = model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        new_ids = out[0, encoded.input_ids.shape[1]:]
        answer = tok.decode(new_ids, skip_special_tokens=True).strip()
        output_tokens += len(new_ids)
        m = re.search(r"\b(\d{1,2})\b", answer)
        p = int(m.group(1)) if m and int(m.group(1)) < len(labels) else -1
        y.append(int(item["label"])); pred.append(p)
        rows.append({"text": item["text"], "gold": labels[int(item["label"])], "prediction": labels[p] if p >= 0 else None, "raw_output": answer})
        if (n + 1) % 100 == 0: print(f"{n+1}/{len(ds)}", flush=True)
    valid = [p if p >= 0 else len(labels) for p in pred]
    metrics = {"model": MODEL_ID, "dataset": "PolyAI/banking77", "examples": len(ds), "accuracy": accuracy_score(y, valid), "macro_f1": f1_score(y, valid, average="macro", labels=list(range(len(labels))), zero_division=0), "weighted_f1": f1_score(y, valid, average="weighted", labels=list(range(len(labels))), zero_division=0), "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "invalid_outputs": sum(p < 0 for p in pred)}
    OUT.mkdir(exist_ok=True); (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); (OUT / "predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
