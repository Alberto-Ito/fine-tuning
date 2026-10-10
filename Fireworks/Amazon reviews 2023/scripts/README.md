# Data preparation scripts

These scripts prepare the `Industrial_and_Scientific` source subset and the first English-only SFT pilot.

## Pipeline

1. `build_industrial_subset.py` streams the official review and metadata files and creates the 150,000-review curated subset.
2. `validate_industrial_subset.py` checks counts, identifiers, product coverage, and selection constraints.
3. `join_reviews_metadata.py` joins each review with product metadata through `parent_asin`.
4. `build_sft_pilot.py` groups one product's reviews, filters content and language, selects task-specific evidence, and creates four SFT examples.
5. `test_sqlite_staging.py` validates disk-backed grouping with a small product sample.
6. `generate_foundry_sft.py` generates product-level SFT candidates with a Microsoft Foundry teacher.
7. `revalidate_sft_exports.py` rebuilds routed outputs after quality checks.

## Build the source subset

The script does not clone the complete dataset or persist full category files. It performs two streaming review passes and one metadata pass:

1. Count unique valid reviews by `parent_asin`.
2. Select 4,000 core products with at least 20 reviews.
3. Allocate 20–40 reviews per core product for a 120,000-review target.
4. Select 10,000 additional long-tail products with three reviews each.
5. Retain reviews deterministically by stable hash.
6. Retain metadata only for selected products.
7. Write a product plan and reproducibility manifest.

```bash
python3 scripts/build_industrial_subset.py \
  --output-dir data/industrial_and_scientific_150k
```

## Validate the source subset

```bash
python3 scripts/validate_industrial_subset.py \
  --data-dir data/industrial_and_scientific_150k
```

Validation covers counts, unique IDs, missing fields, rating distribution, metadata coverage, and plan compliance.

## Join reviews and metadata

```bash
python3 scripts/join_reviews_metadata.py \
  --data-dir data/industrial_and_scientific_150k
```

The output is `enriched_reviews.jsonl.gz`. Human-readable fields are stored under `product` and `review`; identifiers remain under `source` for traceability and leakage-safe splitting.

## Build the four-task SFT pilot

```bash
python3 scripts/build_sft_pilot.py \
  --input data/industrial_and_scientific_150k/enriched_reviews.jsonl.gz \
  --parent-asin B094R8RBWT \
  --teacher-answers scripts/pilot_teacher_answers.json \
  --output-dir data/sft_pilot_aqara
```

The command creates three JSONL files at the same time:

| File | Contents |
|---|---|
| `examples_master.jsonl` | Every example with messages and provenance metadata. |
| `train_fireworks.jsonl` | Approved examples only, containing only `messages`. |
| `pending_human_review.jsonl` | Pending examples with messages and metadata. |

The three files are opened together. Each generated example is written
immediately to the master file and then routed to either the Fireworks file or
the pending-review file. The process does not build one complete output and
then reread it to produce the others.

This pilot keeps only the selected product's reviews in memory. The full
multi-product pipeline must use disk-backed grouping (for example SQLite or an
external sort by `parent_asin`) because reviews for the same product are not
guaranteed to be contiguous in the source stream. It must then load and process
only one product group at a time.

The tasks are `pros_and_cons`, `use_cases`, `common_problems`, and `purchase_recommendation`. Prompts, evidence, and teacher answers are English-only. New examples default to `pending_human_review`. A task can be approved or rejected by adding a `_review_statuses` object to `pilot_teacher_answers.json`; only `approved` examples are exported to the Fireworks file.

## Product distribution

| Cohort | Products | Reviews per product | Total reviews |
|---|---:|---:|---:|
| Core | 4,000 | 20–40 | 120,000 |
| Long tail | 10,000 | 3 | 30,000 |
| Total | 14,000 | Variable | 150,000 |

The raw curated layer preserves the natural rating distribution. Task balance is introduced only when building the SFT dataset.

## Transfer and storage behavior

- The approximately 750 GB full repository is not downloaded.
- Only the two official `Industrial_and_Scientific` streams are accessed.
- Full source streams are never persisted locally.
- Temporary outputs are renamed only after successful completion.
- Existing final source outputs are not overwritten automatically.

## SQLite staging pilot

This command streams the complete enriched source, stores only ten selected
products and their reviews in SQLite, and then loads one product group at a
time for validation:

```bash
python3 scripts/test_sqlite_staging.py \
  --input data/industrial_and_scientific_150k/enriched_reviews.jsonl.gz \
  --output-dir data/sqlite_pilot_10 \
  --products 10
```

The pilot creates `staging.sqlite` and `report.json`. It does not generate SFT
answers; it validates staging, indexing, product grouping, rating distribution,
and bounded-memory processing.
