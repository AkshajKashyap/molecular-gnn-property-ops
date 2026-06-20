from pathlib import Path

import pandas as pd

from molgnn_ops.prep import prepare_dataset


def _write_dataset(path: Path) -> None:
    pd.DataFrame(
        {
            "molecule": [
                "c1ccccc1O",
                "c1ccccc1N",
                "C1CCCCC1O",
                "C1CCCCC1N",
                None,
            ],
            "activity": [1.0, 2.0, 3.0, None, 5.0],
        }
    ).to_csv(path, index=False)


def test_prepare_dataset_writes_random_split_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "nested" / "prepared.csv"
    _write_dataset(input_csv)

    summary = prepare_dataset(
        input_csv,
        output_csv,
        "molecule",
        "activity",
        "example",
        "random",
    )
    prepared = pd.read_csv(output_csv)

    assert list(prepared.columns) == ["smiles", "target", "dataset_name", "split"]
    assert len(prepared) == 4
    assert set(prepared["split"]) == {"train", "val", "test"}
    assert summary.n_rows == 5
    assert summary.n_valid_smiles == 4
    assert summary.n_missing_targets == 1
    assert summary.n_train + summary.n_val + summary.n_test == 4
    assert summary.split_strategy == "random"
    assert summary.output_path == output_csv


def test_prepare_dataset_supports_scaffold_split(tmp_path: Path) -> None:
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "prepared.csv"
    _write_dataset(input_csv)

    summary = prepare_dataset(
        input_csv,
        output_csv,
        "molecule",
        "activity",
        "example",
        "scaffold",
    )

    assert output_csv.is_file()
    assert summary.split_strategy == "scaffold"
    assert summary.n_train + summary.n_val + summary.n_test == 4
