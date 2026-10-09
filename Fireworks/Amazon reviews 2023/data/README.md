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
└── sft_pilot_aqara/
    ├── train_sample.jsonl
    └── manifest.json
```

`reviews.jsonl.gz` and `metadata.jsonl.gz` are the curated raw layer. They preserve original source text, which may contain languages other than English.

`enriched_reviews.jsonl.gz` is the joined intermediate layer. It is not an SFT dataset. All derived examples from one `source.parent_asin` must remain in the same split.

`sft_pilot_aqara/train_sample.jsonl` contains four English-only pilot conversations in `messages` format. They are marked `pending_human_review` and must not be used for production training until approved.

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
