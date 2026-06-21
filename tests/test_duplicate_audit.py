from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.duplicate_audit import audit_duplicate_molecules


def test_duplicate_audit_reports_identical_and_conflicting_targets(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "sample_id": ["data:0", "data:1", "data:2", "data:3", "data:4"],
            "smiles": ["CCO", "OCC", "CCN", "NCC", "CCC"],
            "canonical_smiles": ["CCO", "CCO", "CCN", "CCN", "CCC"],
            "target": [1.0, 1.0, 2.0, 2.5, 3.0],
            "dataset_name": ["data"] * 5,
            "split": ["train", "train", "val", "val", "test"],
        }
    ).to_csv(prepared_csv, index=False)

    audit = audit_duplicate_molecules(prepared_csv)

    assert audit["total_rows"] == 5
    assert audit["unique_sample_ids"] == 5
    assert audit["unique_canonical_smiles"] == 3
    assert audit["duplicate_canonical_smiles_groups"] == 2
    assert audit["rows_in_duplicate_groups"] == 4
    assert audit["duplicate_groups_with_identical_targets"] == 1
    assert audit["duplicate_groups_with_conflicting_targets"] == 1
    assert audit["conflicting_groups"][0]["sample_ids"] == ["data:2", "data:3"]


def test_duplicate_audit_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "sample_id": ["data:0", "data:0"],
            "smiles": ["CCO", "CCN"],
            "canonical_smiles": ["CCO", "CCN"],
            "target": [1.0, 2.0],
            "dataset_name": ["data", "data"],
            "split": ["train", "test"],
        }
    ).to_csv(prepared_csv, index=False)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit_duplicate_molecules(prepared_csv)
