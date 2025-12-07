from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

# Новые тесты
def test_quality_flags_constant_column() -> None:
    """Проверяем, что константная колонка правильно детектится (has_constant_columns)."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [100, 200, 300, 400],
            "status": ["active", "active", "active", "active"],
            "category": ["X", "Y", "X", "Z"],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert flags["constant_columns_count"] == 1
    assert "status" in flags["constant_columns"]


def test_quality_flags_high_cardinality_categoricals() -> None:
    """Проверяем детект высокой кардинальности в категориальных колонках."""
    df = pd.DataFrame(
        {
            "user_id": [1001, 1002, 1003, 1004, 1005],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "unique_tag": ["tag_001", "tag_002", "tag_003", "tag_004", "tag_005"],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["high_cardinality_columns_count"] >= 1
    assert "unique_tag" in flags["high_cardinality_columns"]
    assert "user_id" not in flags["high_cardinality_columns"]
