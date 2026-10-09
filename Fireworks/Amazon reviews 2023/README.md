# Amazon Reviews 2023

This directory documents the analysis and preparation of **Amazon Reviews 2023**, a public product-review and product-metadata dataset collected by McAuley Lab. It records the sampling, cleaning, joining, splitting, transformation, and evaluation decisions required before fine-tuning.

> The correct name is **Amazon Reviews 2023**. It is not an AWS dataset.

## Official sources

- [Hugging Face dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [Dataset paper](https://arxiv.org/abs/2403.03952)
- [UCSD data repository](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/)

## Dataset summary

| Metric | Approximate value |
|---|---:|
| Reviews/ratings | 571.54 million |
| Users | 54.51 million |
| Products | 48.19 million |
| Time range | May 1996–September 2023 |
| Declared catalog categories | 33 plus `Unknown` |

The source contains three main information groups:

1. Reviews, ratings, timestamps, and verified-purchase indicators.
2. Product metadata such as name, description, features, price, images, and technical details.
3. Product and user relationships, including identifiers and `bought_together` associations when available.

See [categories.md](categories.md) for the category catalog and [use_case.md](use_case.md) for the industrial demo recommendation.

## Review schema

| Field | Type | Meaning |
|---|---|---|
| `rating` | `float` | Rating from 1 to 5. |
| `title` | `string` | Review title. |
| `text` | `string` | Review body. |
| `images` | `list` | User-attached images, when present. |
| `asin` | `string` | Identifier for a specific product variant. |
| `parent_asin` | `string` | Identifier used to group related variants. |
| `user_id` | `string` | Anonymized user identifier. |
| `timestamp` | `integer` | Unix timestamp in milliseconds. |
| `helpful_vote` | `integer` | Number of helpful votes. |
| `verified_purchase` | `boolean` | Whether Amazon recorded a verified purchase. |

## Product metadata schema

| Field | Type | Meaning |
|---|---|---|
| `main_category` | `string` | Main catalog category. |
| `title` | `string` | Published product name. |
| `average_rating` | `float` | Catalog average rating at capture time. |
| `rating_number` | `integer` | Catalog rating count at capture time. |
| `features` | `list[string]` | Published product features. |
| `description` | `list[string]` | Product description. |
| `price` | `float/null` | Published price when available. |
| `images` | `list[object]` | Product image references. |
| `videos` | `list[object]` | Product video references. |
| `store` | `string/null` | Brand or store label. |
| `categories` | `list[string]` | Catalog hierarchy. |
| `details` | `object` | Technical attributes. |
| `parent_asin` | `string` | Join key for reviews and metadata. |
| `bought_together` | `list/string/null` | Related co-purchase products. |

## Important relationships

- Join reviews and metadata through `parent_asin`.
- One `parent_asin` may contain multiple `asin` variants.
- One user may review products across categories and time periods.
- Some products have incomplete or contaminated metadata.
- Reviews may be duplicated across variants or related products.

## Category-level loading

Do not download the full repository for early iterations. Stream only the required category or official raw files.

```python
from datasets import load_dataset

reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Industrial_and_Scientific",
    split="full",
    streaming=True,
)
```

If remote dataset scripts are unavailable in the installed `datasets` version, load the official JSONL files directly with `streaming=True`. Streaming avoids materializing the complete category, but selected records still consume local storage when saved.

## Intended use cases

The dataset can support:

- review summarization and aspect extraction;
- product-quality and defect classification;
- grounded product comparison;
- technical purchasing assistance;
- evidence-backed question answering;
- recommendation research and retrieval experiments;
- instruction-data generation from product evidence.

It does not contain real-time inventory, support tickets, returns, work orders, internal policies, or tool traces. Those workflows require proprietary data or controlled synthetic scenarios.

## Fine-tuning strategy

Do not convert raw reviews directly into arbitrary user/assistant pairs. Use this sequence:

1. Select categories aligned with the business scenario.
2. Define a task taxonomy and quality rubric.
3. Remove empty, duplicated, corrupted, unsafe, and off-domain content.
4. Enforce English on all derived training content.
5. Join reviews and product metadata through `parent_asin`.
6. Create product-level splits before generating synthetic examples.
7. Generate grounded examples while retaining evidence IDs in metadata.
8. Apply automatic graders and human review.
9. Version prompts, transformations, model outputs, and manifests.

A reasonable first source layer contains 100,000–300,000 reviews, but the SFT set should be smaller and more curated, initially around 30,000–80,000 conversations.

## Splitting and leakage prevention

Never perform a random review-level split. Reviews from the same product or variant would leak across sets.

- Group by `parent_asin`.
- Assign every product exclusively to `train`, `validation`, or `test`.
- Keep all synthetic examples derived from the same evidence in one split.
- Use time to create a temporal test when possible.
- Deduplicate before and after example generation.
- Maintain a test slice containing unseen products or subcategories.

Suggested initial allocation:

| Split | Share | Purpose |
|---|---:|---|
| Train | 80% | Parameter updates. |
| Validation | 10% | Hyperparameter and checkpoint selection. |
| Test | 10% | Final evaluation without development exposure. |

## Quality controls

- Exact and near-duplicate detection.
- Minimum information-content requirements.
- English-language validation for all model-facing content.
- Removal of catalog contamination and obvious category errors.
- Personally identifiable or sensitive-information detection.
- Product, rating, date, cohort, and task balance.
- Evidence-entailment checks for synthetic answers.
- Schema and JSON validation.
- Product-level split-isolation checks.
- Human review of a sample and the complete gold evaluation set.

## Risks and limitations

- Reviewers are self-selected and do not represent all buyers.
- Ratings may be manipulated, duplicated, or influenced by incentives.
- Prices and attributes may be obsolete.
- Category, product, and rating distributions are highly uneven.
- Source text can include personal or sensitive information.
- `verified_purchase` does not guarantee correctness.
- Catalog metadata and reviews may refer to different variants.
- The dataset is useful for retail analysis but is not equivalent to real customer-support conversations.
- Licensing and commercial-use conditions must be reviewed before production use.

## Derived-data convention

Each SFT example should use `messages` and retain provenance outside model-visible text:

```json
{
  "messages": [
    {"role": "system", "content": "You are a technical purchasing copilot."},
    {"role": "user", "content": "Product evidence and task..."},
    {"role": "assistant", "content": "Grounded response..."}
  ],
  "metadata": {
    "example_id": "deterministic_hash",
    "task": "purchase_recommendation",
    "parent_asin": "source_only",
    "source_review_ids": ["source_only"],
    "split": "train",
    "language": "en",
    "pipeline_version": "v1"
  }
}
```

Identifiers are used for provenance, deduplication, and splitting; they must not appear in the conversational content unless the task explicitly requires them.

## Current project artifacts

- 150,000 selected reviews covering 14,000 products.
- Complete metadata coverage for selected products.
- Joined `enriched_reviews.jsonl.gz` intermediate dataset.
- Four English-only pilot SFT examples for one product.
- Reproducible build, validation, join, and pilot-generation scripts.
