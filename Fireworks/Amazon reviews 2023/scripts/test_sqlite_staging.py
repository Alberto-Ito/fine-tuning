#!/usr/bin/env python3
"""Build and validate a disk-backed SQLite staging pilot for N products."""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE products (
    parent_asin TEXT PRIMARY KEY,
    product_json TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    review_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    parent_asin TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_json TEXT NOT NULL,
    FOREIGN KEY (parent_asin) REFERENCES products(parent_asin)
);
CREATE INDEX idx_reviews_parent_asin ON reviews(parent_asin);
CREATE INDEX idx_reviews_parent_asin_rating ON reviews(parent_asin, rating);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products", type=int, default=10)
    return parser.parse_args()


def initialize_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing database: {path}")
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def stage_products(
    connection: sqlite3.Connection, input_path: Path, product_limit: int
) -> tuple[int, set[str]]:
    selected: set[str] = set()
    source_rows = 0
    inserted_reviews = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as source:
        for line in source:
            source_rows += 1
            row = json.loads(line)
            parent_asin = row["source"]["parent_asin"]
            if parent_asin not in selected:
                if len(selected) >= product_limit:
                    continue
                selected.add(parent_asin)
                connection.execute(
                    "INSERT INTO products(parent_asin, product_json) VALUES (?, ?)",
                    (parent_asin, json.dumps(row["product"], ensure_ascii=False)),
                )

            review = row["review"]
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO reviews(
                    review_id, parent_asin, rating, review_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    row["source"]["review_id"],
                    parent_asin,
                    int(review["rating"]),
                    json.dumps(review, ensure_ascii=False),
                ),
            )
            inserted_reviews += cursor.rowcount
            if inserted_reviews and inserted_reviews % 100 == 0:
                connection.commit()

    connection.execute(
        """
        UPDATE products
        SET review_count = (
            SELECT COUNT(*) FROM reviews
            WHERE reviews.parent_asin = products.parent_asin
        )
        """
    )
    connection.commit()
    return source_rows, selected


def validate_and_process(connection: sqlite3.Connection) -> list[dict]:
    summaries = []
    products = connection.execute(
        """
        SELECT parent_asin, product_json, review_count
        FROM products ORDER BY parent_asin
        """
    ).fetchall()

    for parent_asin, product_json, review_count in products:
        rows = connection.execute(
            """
            SELECT rating, review_json FROM reviews
            WHERE parent_asin = ? ORDER BY rating, review_id
            """,
            (parent_asin,),
        ).fetchall()
        if len(rows) != review_count:
            raise RuntimeError(
                f"Review count mismatch for {parent_asin}: "
                f"expected {review_count}, loaded {len(rows)}"
            )

        ratings = {str(rating): 0 for rating in range(1, 6)}
        for rating, _ in rows:
            ratings[str(rating)] += 1
        product = json.loads(product_json)
        summaries.append(
            {
                "parent_asin": parent_asin,
                "product_name": product.get("name"),
                "review_count": review_count,
                "ratings": ratings,
                "loaded_in_memory_for_this_product": len(rows),
            }
        )
        connection.execute(
            "UPDATE products SET processing_status = 'completed' WHERE parent_asin = ?",
            (parent_asin,),
        )
        connection.commit()
    return summaries


def main() -> None:
    args = parse_args()
    if args.products < 1:
        raise ValueError("--products must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "staging.sqlite"
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    connection = initialize_database(database_path)
    try:
        source_rows, selected = stage_products(connection, args.input, args.products)
        summaries = validate_and_process(connection)
        counts = {
            "products": connection.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "reviews": connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0],
            "completed_products": connection.execute(
                "SELECT COUNT(*) FROM products WHERE processing_status = 'completed'"
            ).fetchone()[0],
        }
    finally:
        connection.close()

    if len(selected) != args.products:
        raise RuntimeError(f"Requested {args.products} products but found {len(selected)}")

    report = {
        "input": str(args.input),
        "database": str(database_path),
        "source_rows_scanned": source_rows,
        "requested_products": args.products,
        "database_counts": counts,
        "processing_strategy": "one product group loaded at a time",
        "products": summaries,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"source_rows_scanned={source_rows:,}")
    print(f"products={counts['products']:,}")
    print(f"reviews={counts['reviews']:,}")
    print(f"completed_products={counts['completed_products']:,}")
    print(f"database={database_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
