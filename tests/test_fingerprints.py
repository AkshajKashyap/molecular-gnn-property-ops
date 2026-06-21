from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molgnn_ops.fingerprints import (
    featurize_fingerprints_from_csv,
    morgan_fingerprint,
)


def test_morgan_fingerprint_is_binary_with_expected_length() -> None:
    fingerprint = morgan_fingerprint("CCO", radius=2, n_bits=128)

    assert len(fingerprint) == 128
    assert set(fingerprint) <= {0, 1}


def test_morgan_fingerprint_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        morgan_fingerprint("not-a-smiles")


def test_featurize_fingerprints_from_csv_skips_unusable_rows(tmp_path: Path) -> None:
    input_csv = tmp_path / "prepared.csv"
    output_npz = tmp_path / "nested" / "fingerprints.npz"
    pd.DataFrame(
        {
            "smiles": ["CCO", "not-a-smiles", "Cl", "CCN"],
            "target": [1.0, 0.0, None, 0.0],
            "split": ["train", "val", "test", "test"],
            "dataset_name": ["example"] * 4,
        }
    ).to_csv(input_csv, index=False)

    summary = featurize_fingerprints_from_csv(
        input_csv,
        output_npz,
        n_bits=64,
    )

    with np.load(output_npz, allow_pickle=False) as dataset:
        assert dataset["X"].shape == (2, 64)
        assert dataset["y"].tolist() == [1.0, 0.0]
        assert dataset["splits"].tolist() == ["train", "test"]
        assert set(dataset.files) == {
            "X",
            "y",
            "splits",
            "smiles",
            "dataset_name",
            "sample_id",
            "canonical_smiles",
        }
    assert summary == {
        "n_rows": 4,
        "n_valid": 3,
        "n_invalid": 1,
        "n_missing_targets": 1,
        "n_written": 2,
    }
