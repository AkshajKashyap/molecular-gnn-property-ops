from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.datasets import load_csv_dataset


def test_load_csv_dataset_loads_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "molecule": [" CCO ", "CCN"],
            "activity": [1.5, 2.0],
        }
    ).to_csv(csv_path, index=False)

    records = load_csv_dataset(csv_path, "molecule", "activity", "example")

    assert len(records) == 2
    assert records[0].smiles == "CCO"
    assert records[0].canonical_smiles == "CCO"
    assert records[0].sample_id == "example:0"
    assert records[0].target == 1.5
    assert records[0].dataset_name == "example"


def test_load_csv_dataset_raises_for_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CSV dataset not found"):
        load_csv_dataset(tmp_path / "missing.csv", "smiles", "target", "example")


def test_load_csv_dataset_raises_for_missing_smiles_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame({"target": [1.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="SMILES column 'smiles'"):
        load_csv_dataset(csv_path, "smiles", "target", "example")


def test_load_csv_dataset_drops_rows_with_missing_smiles(tmp_path: Path) -> None:
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", None, "   ", "CCN"],
            "target": [1.0, 2.0, 3.0, None],
        }
    ).to_csv(csv_path, index=False)

    records = load_csv_dataset(csv_path, "smiles", "target", "example")

    assert [record.smiles for record in records] == ["CCO", "CCN"]
    assert records[1].target is None


def test_load_csv_dataset_sample_ids_follow_original_source_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", None, "OCC"],
            "target": [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    first = load_csv_dataset(csv_path, "smiles", "target", "example")
    second = load_csv_dataset(csv_path, "smiles", "target", "example")

    assert [record.sample_id for record in first] == ["example:0", "example:2"]
    assert [record.sample_id for record in second] == ["example:0", "example:2"]
    assert [record.canonical_smiles for record in first] == ["CCO", "CCO"]
