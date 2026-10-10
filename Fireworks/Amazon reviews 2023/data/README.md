# Local data

This directory contains the selected `Industrial_and_Scientific` source data and derived datasets.

```text
data/
├── industrial_and_scientific_150k/
│   ├── reviews.jsonl.gz
│   ├── metadata.jsonl.gz
│   ├── enriched_reviews.jsonl.gz
│   ├── product_plan.json
│   └── manifest.json
├── sft_pilot_aqara/
│   ├── examples_master.jsonl
│   ├── train_fireworks.jsonl
│   ├── pending_human_review.jsonl
│   └── manifest.json
├── sqlite_pilot_10/
│   ├── staging.sqlite
│   └── report.json
└── foundry_sft_10000/
    ├── examples_master.jsonl
    ├── training.jsonl
    ├── pending_human_review.jsonl
    ├── staging.sqlite
    └── manifest.json
```

`reviews.jsonl.gz` and `metadata.jsonl.gz` are the curated raw layer. They preserve original source text, which may contain languages other than English.

`enriched_reviews.jsonl.gz` is the joined intermediate layer. It is not an SFT dataset. All derived examples from one `source.parent_asin` must remain in the same split.

`examples_master.jsonl` is the canonical output with provenance metadata. `train_fireworks.jsonl` contains only approved examples and only the Fireworks-compatible `messages` field. `pending_human_review.jsonl` is the review queue. The current four pilot conversations are pending and must not be used for production training until approved.

## Current status

| Metric | Value |
|---|---:|
| Selected reviews | 150,000 |
| Selected products | 14,000 |
| Core products | 4,000 |
| Long-tail products | 10,000 |
| Products with metadata | 14,000 |
| Duplicate review IDs | 0 |
| Missing product metadata | 0 |
| Pilot SFT examples | 4 |

Large source and derived data files should not be committed unless the repository explicitly uses an appropriate data-versioning strategy.

## Foundry-generated pilot

`foundry_sft_10000` contains a GPT-4.1-mini teacher run targeting 10,000 SFT
candidates. `training.jsonl` contains only automatically approved,
Fireworks-compatible `messages` records. `pending_human_review.jsonl` retains
full metadata for cases that failed a quality or language gate. See its
`manifest.json` for final counts and provenance.
