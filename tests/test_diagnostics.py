from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.diagnostics import (
    prediction_error_summary,
    scaffold_distribution_summary,
    split_target_summary,
    target_summary,
    train_test_similarity_summary,
    worst_predictions,
)


def _write_prepared_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "c1ccccc1O", "C1CCCCC1"],
            "target": [1.0, 2.0, 3.0, 1.5, 3.5, 2.5],
            "dataset_name": ["synthetic"] * 6,
            "split": ["train", "train", "train", "test", "test", "test"],
        }
    ).to_csv(path, index=False)


def _write_predictions_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCCl", "c1ccccc1O"],
            "split": ["val", "val", "test", "test"],
            "y_true": [1.0, 2.0, 1.0, 2.0],
            "y_pred": [1.5, 1.5, 2.0, 0.0],
        }
    ).to_csv(path, index=False)


def test_target_summary() -> None:
    summary = target_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(1.1180339887)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["q25"] == pytest.approx(1.75)
    assert summary["q75"] == pytest.approx(3.25)


def test_split_target_summary(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    _write_prepared_csv(prepared_csv)

    summary = split_target_summary(prepared_csv)

    assert set(summary) == {"train", "test"}
    assert summary["train"]["count"] == 3
    assert summary["test"]["mean"] == pytest.approx(2.5)


def test_prediction_error_summary(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    summary = prediction_error_summary(predictions_csv)["test"]

    assert summary["n"] == 2
    assert summary["mae"] == pytest.approx(1.5)
    assert summary["rmse"] == pytest.approx(2.5**0.5)
    assert summary["mean_error"] == pytest.approx(-0.5)
    assert summary["median_abs_error"] == pytest.approx(1.5)
    assert summary["max_abs_error"] == pytest.approx(2.0)


def test_worst_predictions_sorts_by_absolute_error(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    rows = worst_predictions(predictions_csv, split="test", n=2)

    assert [row["absolute_error"] for row in rows] == [2.0, 1.0]
    assert rows[0]["smiles"] == "c1ccccc1O"


def test_scaffold_distribution_summary(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    _write_prepared_csv(prepared_csv)

    summary = scaffold_distribution_summary(prepared_csv)

    assert {
        "n_unique_scaffolds",
        "largest_scaffold_group_size",
        "median_scaffold_group_size",
        "top_10_scaffold_groups",
        "n_singleton_scaffolds",
    } <= set(summary)


def test_train_test_similarity_summary(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    _write_prepared_csv(prepared_csv)

    summary = train_test_similarity_summary(prepared_csv, n_bits=64)

    assert summary["n_test"] == 3
    assert 0.0 <= summary["min_max_similarity"] <= 1.0
    assert 0.0 <= summary["mean_max_similarity"] <= 1.0
    assert 0.0 <= summary["q25"] <= summary["q75"] <= 1.0
    assert {"n_below_0_3", "n_below_0_5", "n_below_0_7"} <= set(summary)
