from typing import Any

from datasets import DatasetDict, load_dataset


def load_raw_dataset(config: dict[str, Any]):
    # ``source_type`` is canonical; ``loader`` remains a compatibility fallback.
    source = config.get("source_type", config.get("loader", "csv"))
    if source == "csv":
        raw = load_dataset("csv", data_files=config["data_files"])
        source_label = config.get("source_label_column")
        target_label = config["label_column"]
        if source_label and source_label != target_label:
            raw = raw.rename_column(source_label, target_label)
        return raw.class_encode_column(target_label)
    if source == "huggingface":
        return load_dataset(
            config["huggingface_id"], config.get("subset"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unsupported dataset loader: {source}")


def prepare_datasets(dataset_config: dict[str, Any], tokenizer, max_length: int):
    raw = load_raw_dataset(dataset_config)
    train_split = dataset_config["train_split"]
    split = raw[train_split].train_test_split(
        test_size=float(dataset_config["validation_fraction"]),
        seed=int(dataset_config.get("seed", 42)),
        stratify_by_column=dataset_config["label_column"],
    )
    datasets = DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw[dataset_config["test_split"]],
    )
    max_examples = dataset_config.get("max_examples")
    if max_examples is not None:
        limit = int(max_examples)
        seed = int(dataset_config.get("seed", 42))
        datasets = DatasetDict({
            name: split.shuffle(seed=seed).select(range(min(limit, len(split))))
            for name, split in datasets.items()
        })
    label_feature = datasets["train"].features[dataset_config["label_column"]]
    label_names = list(label_feature.names)
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(
            batch[text_column], truncation=True, max_length=max_length, padding="max_length"
        )
        encoded["labels"] = batch[label_column]
        return encoded

    tokenized = datasets.map(
        tokenize, batched=True, remove_columns=datasets["train"].column_names
    )
    return datasets, tokenized, label_names
