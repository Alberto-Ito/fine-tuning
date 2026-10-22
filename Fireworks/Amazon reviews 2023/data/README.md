# Local data

This directory contains the Amazon Reviews 2023 source subsets, generation
artifacts, historical experiments, and the current canonical SFT dataset for
`Industrial_and_Scientific`.

Status below reflects the files on **2026-10-21**. For machine-readable counts
and batch provenance, use `dataset/manifest.json`.

## Canonical dataset

`dataset/` is the only current combined dataset. Training and evaluation are
product-disjoint, and each `parent_asin` occurs only once in
`examples_master.jsonl`.

| Metric | Count |
|---|---:|
| Canonical examples / unique products | 1,872 |
| Approved | 1,284 |
| Training examples | 830 |
| Evaluation examples | 121 |
| Backup examples | 520 |
| Pending human review | 532 |
| Rejected | 56 |

Only approved examples are exported to training or evaluation. The backup
split retains every example excluded by the active result-routing policy,
including approved results assigned to neither target as well as pending and
rejected cases.

### Approved result distribution by split

| Result | Training | Evaluation |
|---|---:|---:|
| `BUY` | 128 | 22 |
| `CONDITIONAL_BUY` | 211 | 24 |
| `DO_NOT_BUY` | 300 | 51 |
| `INSUFFICIENT_EVIDENCE` | 191 | 24 |
| **Total** | **830** | **121** |

The latest automatic generation policy sends approved `BUY` examples to
evaluation and approved `CONDITIONAL_BUY` examples to training. A subsequent
focused review added 26 corrected, pure `BUY` examples to training. Do not infer
split distributions from `generation_batches[*].actual_results`: batch result
counts include approved, pending, and rejected examples.

### Canonical files

| File | Purpose |
|---|---|
| `dataset/examples_master.jsonl` | Canonical 1,872-row ledger with messages, product identity, source-review hashes, models, judge output, status, provenance, and split assignment. |
| `dataset/training.jsonl` | 830 approved, messages-only SFT records, including 26 `BUY` examples added by focused review. |
| `dataset/evaluation.jsonl` | 121 approved, messages-only evaluation records, including 22 `BUY` examples and disjoint from training by product and conversation. |
| `dataset/backup.jsonl` | 520 full examples excluded from the target result routes, including non-approved target responses and all other results from routed batches. |
| `dataset/pending_human_review.jsonl` | 532 full examples that did not pass the automatic gate unchanged. |
| `dataset/rejected.jsonl` | 56 full examples rejected by the judge or human-review state. |
| `dataset/manifest.json` | Counts, result distributions, model mix, source batches, offsets, timings, and generation history. |
| `dataset/human_review_decisions.jsonl` | The 26 applied `BUY` approvals produced by the focused review of the pending set. |
| `dataset/human_review_decisions.example.jsonl` | Example format for recording manual decisions. It is not an applied decision file. |

The messages-only exports contain exactly three roles in order: `system`,
`user`, and `assistant`. The master file is the source of truth; derived exports
must be rebuilt from it rather than edited independently.

## Active source data

### `industrial_and_scientific_5k_products/`

This is the active Amazon Reviews 2023 source layer used by the current SFT
pipeline.

- 5,000 selected products: 4,000 core plus 1,000 deterministic long-tail.
- 533,879 original review rows.
- Every matching source review is retained: there is no per-product cap, text
  filter, rating filter, or deduplication pass.
- Every selected product has metadata and at least three reviews.
- `reviews.jsonl.gz`, `metadata.jsonl.gz`, and `products.json` are source
  artifacts; `manifest.json` documents the selection and counts.

The compressed files use Git LFS. If a `.gz` file is only a small text pointer,
restore it with `git lfs checkout` before running generation.

### `industrial_and_scientific_150k/`

This is the earlier curated sample, not the source for the current canonical
dataset. It contains 150,000 selected reviews across 14,000 products using a
core-plus-long-tail strategy with per-product sampling. `enriched_reviews` is a
joined intermediate and is not an SFT dataset.

## Current generation pipeline

The current single-pass pipeline uses:

1. `gpt-4.1-mini` to inspect product/catalog/review evidence, select the most
   defensible result, and author the procurement request.
2. `gpt-6-sol` to generate the assistant response once.
3. A separate `gpt-6-sol` invocation to audit system, user, assistant, result
   precedence, grounding, attribution, and SFT readiness.

Generation is batched, but each product receives one effective authoring pass,
one response pass, and one judge pass. A second transport attempt is permitted
only after timeout, invalid JSON, or another technical failure; there is no
feedback-driven answer regeneration.

The first 50 canonical examples used `gpt-4o` as assistant. The remaining 1,822
used `gpt-6-sol`. Exact model and batch provenance is stored per example and in
`dataset/manifest.json`.

The automatic generation routing policy sends approved `BUY` examples to
evaluation and approved `CONDITIONAL_BUY` examples to training. The later human
review added 26 corrected `BUY` examples to training. Every other generated
result, plus all pending and rejected examples, is routed to `backup.jsonl`.

Product generation normally requires at least 30 source reviews. After those
1,822 eligible products were exhausted, the latest 50-example batch used an
explicit minimum of 25 reviews, excluded all previously used products, and
still required at least four usable reviews from the dominant variant.

## Directory inventory

### Active and canonical

- `dataset/`: combined master dataset and current exports.
- `industrial_and_scientific_5k_products/`: active complete-review source.
- `generation_staging/`: restartable per-batch outputs already represented in
  the canonical manifest. These are audit/recovery artifacts, not another
  dataset to concatenate.

### Single-pass milestones

These directories preserve the early batches that seeded the canonical
dataset. Do not append them again; their examples are already represented in
`dataset/`.

- `validated_sft_single_pass_50/`: first 50, with `gpt-4o` assistant; none
  passed the strict automatic gate.
- `validated_sft_single_pass_50_gpt6_assistant_next50/`: next 50 with
  `gpt-6-sol` assistant; 38 approved.
- `validated_sft_single_pass_200_gpt6_assistant_next200/`: next 200; 152
  approved.

### Historical or experimental outputs

- `rich_sft_pilot_50/` through `rich_sft_pilot_50_v4/`: successive 50-example
  prompt/template experiments and GPT-6 review artifacts.
- `validated_sft_pilot_10/` and `validated_sft_pilot_10_v2/`: three-model
  multi-attempt prototypes superseded by the single-pass pipeline.
- `sft_pilot_aqara/` and `sqlite_pilot_10/`: early product and SQLite pilots.
- `foundry_generation_smoke_4/` and `_v2/`: endpoint/pipeline smoke tests.
- `foundry_sft_10000/`: earlier GPT-4.1-mini teacher run; its manifest reports
  9,348 master examples, 6,817 approved, and 2,531 pending after final checks.
- `foundry_sft_full/`: earlier large run; its manifest reports 47,708 master
  examples, 39,852 approved, and 7,856 pending. It is not merged into the
  current canonical dataset.
- `industrial_and_scientific_150k/`: earlier capped source sample described
  above.

Historical directories are retained for comparison and provenance. They must
not be combined with `dataset/` unless duplicate products and example lineage
are explicitly reconciled.

## Human review

Record proposed decisions in a separate JSONL file using
`dataset/human_review_decisions.example.jsonl` as the schema reference.
Decisions may be `approve`, `reject`, or `pending`; a corrected response belongs
in `corrected_answer`. Never alter system/user messages, source evidence, ASINs,
or provenance fields.

`scripts/apply_human_review_decisions.py` can validate decisions in dry-run
mode, but its current apply path predates the separate evaluation split and
rebuilds all approved examples into `training.jsonl`. **Do not run it with
`--apply` against the current canonical dataset until it is made split-aware.**

Dry-run validation remains non-mutating:

```bash
python3 scripts/apply_human_review_decisions.py \
  --decisions data/dataset/human_review_decisions.jsonl
```

## Data handling

- Treat `examples_master.jsonl` and `manifest.json` as the canonical ledger.
- Keep training and evaluation product-disjoint.
- Do not infer review prevalence from the selected evidence subset; only the
  complete per-product statistics support frequency claims.
- Catalog fields are seller/listing claims unless independently corroborated.
- Large source and derived files should remain under the repository's Git LFS
  or other explicit data-versioning policy.
