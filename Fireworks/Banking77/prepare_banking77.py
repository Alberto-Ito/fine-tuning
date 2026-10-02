import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


SYSTEM_PROMPT = (
    "Identify the intent of the customer's banking request. "
    "Respond with only the intent label and no additional text."
)
REQUIRED_COLUMNS = {"text", "category"}
EXPECTED_CATEGORY_COUNT = 77
VALIDATION_SIZE = 0.10
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a Banking77 CSV file and validate its required fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    if dataframe[list(REQUIRED_COLUMNS)].isnull().any().any():
        null_counts = dataframe[list(REQUIRED_COLUMNS)].isnull().sum()
        details = ", ".join(
            f"{column}={count}"
            for column, count in null_counts.items()
            if count > 0
        )
        raise ValueError(f"{path.name} contains null values: {details}")

    return dataframe


def convert_row_to_messages(row: pd.Series) -> dict[str, list[dict[str, str]]]:
    """Convert one CSV row to the chat format expected for SFT."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["text"])},
            {"role": "assistant", "content": str(row["category"])},
        ]
    }


def write_jsonl(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write one compact JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for _, row in dataframe.iterrows():
            record = convert_row_to_messages(row)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_jsonl(path: Path, expected_examples: int) -> None:
    """Validate the generated JSONL structure and number of examples."""
    example_count = 0
    expected_roles = ["system", "user", "assistant"]

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

            if (
                not isinstance(record, dict)
                or set(record) != {"messages"}
                or not isinstance(record["messages"], list)
            ):
                raise ValueError(f"Invalid record structure in {path} at line {line_number}")

            messages = record["messages"]
            if len(messages) != 3:
                raise ValueError(f"Expected 3 messages in {path} at line {line_number}")

            for message, expected_role in zip(messages, expected_roles):
                if (
                    not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message["role"] != expected_role
                    or not isinstance(message["content"], str)
                ):
                    raise ValueError(
                        f"Invalid {expected_role} message in {path} at line {line_number}"
                    )

            if messages[0]["content"] != SYSTEM_PROMPT:
                raise ValueError(f"Invalid system prompt in {path} at line {line_number}")

            example_count += 1

    if example_count != expected_examples:
        raise ValueError(
            f"{path} contains {example_count} examples; expected {expected_examples}"
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    original_train = load_dataset(base_dir / "train.csv")
    test = load_dataset(base_dir / "test.csv")

    categories = set(original_train["category"]) | set(test["category"])
    category_count = len(categories)

    print(f"Original training examples: {len(original_train)}")
    print(f"Original test examples: {len(test)}")
    print(f"Unique categories: {category_count}")

    if category_count != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORY_COUNT} categories, found {category_count}"
        )

    train, validation = train_test_split(
        original_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=original_train["category"],
    )

    output_files = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "test": output_dir / "test.jsonl",
    }
    datasets = {"train": train, "validation": validation, "test": test}

    for name, dataframe in datasets.items():
        write_jsonl(dataframe, output_files[name])
        validate_jsonl(output_files[name], len(dataframe))

    print("\nBanking77 dataset prepared successfully.\n")
    print(f"Original training examples: {len(original_train)}")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(validation)}")
    print(f"Test examples: {len(test)}")
    print(f"Categories: {category_count}\n")
    print("Generated:")
    for path in output_files.values():
        print(path.relative_to(base_dir))


if __name__ == "__main__":
    main()
