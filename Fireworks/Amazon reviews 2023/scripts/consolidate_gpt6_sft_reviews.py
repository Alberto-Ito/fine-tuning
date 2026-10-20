#!/usr/bin/env python3
"""Consolidate GPT-6 SFT review batches into auditable JSONL and Markdown."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rich_sft_pilot_50_v4"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    examples = read_jsonl(DATA / "examples_master.jsonl")
    reviews: list[dict] = []
    for path in sorted(DATA.glob("gpt6_review_batch_*.jsonl")):
        reviews.extend(read_jsonl(path))
    reviews.sort(key=lambda row: row["index"])
    if len(examples) != 50 or len(reviews) != 50:
        raise RuntimeError(f"Expected 50 examples and reviews; found {len(examples)} and {len(reviews)}")
    for index, (example, review) in enumerate(zip(examples, reviews), start=1):
        if review["index"] != index or review["parent_asin"] != example["metadata"]["parent_asin"]:
            raise RuntimeError(f"Review alignment failed at index {index}")

    combined = DATA / "gpt6_sft_review_all.jsonl"
    combined.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviews),
        encoding="utf-8",
    )
    verdicts = Counter(row["verdict"] for row in reviews)
    summary = {
        "review_model": "gpt-6-sol",
        "examples_reviewed": len(reviews),
        "verdicts": dict(sorted(verdicts.items())),
        "sft_ready": sum(bool(row["sft_ready"]) for row in reviews),
        "decision_correct": sum(bool(row["decision_correct"]) for row in reviews),
        "average_scores": {
            key: round(sum(row[key] for row in reviews) / len(reviews), 2)
            for key in ("system_score", "user_score", "assistant_score")
        },
        "files": {
            "examples": "examples_master.jsonl",
            "detailed_reviews": combined.name,
            "report": "GPT6_SFT_REVIEW.md",
        },
    }
    (DATA / "gpt6_sft_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# GPT-6 SFT review — rich pilot v4",
        "",
        "Review model: `gpt-6-sol`.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Examples reviewed | {len(reviews)} |",
        f"| APPROVE | {verdicts['APPROVE']} |",
        f"| REVISE | {verdicts['REVISE']} |",
        f"| REJECT | {verdicts['REJECT']} |",
        f"| SFT-ready without changes | {summary['sft_ready']} |",
        f"| Correct purchase decision | {summary['decision_correct']} |",
        f"| Average system score | {summary['average_scores']['system_score']} / 5 |",
        f"| Average user score | {summary['average_scores']['user_score']} / 5 |",
        f"| Average assistant score | {summary['average_scores']['assistant_score']} / 5 |",
        "",
        "## Per-example results",
        "",
        "| # | parent_asin | Verdict | SFT ready | System | User | Assistant | Decision | Main finding |",
        "|---:|---|---|:---:|---:|---:|---:|:---:|---|",
    ]
    for row in reviews:
        finding = row["issues"][0] if row["issues"] else "No material issue found"
        lines.append(
            f"| {row['index']} | `{safe_cell(row['parent_asin'])}` | {row['verdict']} | "
            f"{'Yes' if row['sft_ready'] else 'No'} | {row['system_score']} | {row['user_score']} | "
            f"{row['assistant_score']} | {'Correct' if row['decision_correct'] else 'Incorrect'} | "
            f"{safe_cell(finding)} |"
        )
    lines.extend([
        "",
        "## Detailed findings",
        "",
        "The complete issue lists, rationales, and recommended changes are stored in "
        "`gpt6_sft_review_all.jsonl`.",
        "",
    ])
    (DATA / "GPT6_SFT_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
