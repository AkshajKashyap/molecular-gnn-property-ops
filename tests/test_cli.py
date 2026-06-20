from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner


def test_cli_import_works() -> None:
    from molgnn_ops.cli import app

    assert app is not None


def test_prepare_csv_command_smoke(tmp_path: Path) -> None:
    from molgnn_ops.cli import app

    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "prepared.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC"], "target": [1.0, 2.0, 3.0]}
    ).to_csv(input_csv, index=False)

    result = CliRunner().invoke(
        app,
        [
            "prepare-csv",
            str(input_csv),
            str(output_csv),
            "--smiles-col",
            "smiles",
            "--target-col",
            "target",
            "--dataset-name",
            "example",
            "--split-strategy",
            "random",
            "--seed",
            "11",
        ],
    )

    assert result.exit_code == 0
    assert "Train/val/test" in result.output
    assert output_csv.is_file()


def test_featurize_csv_command_smoke(tmp_path: Path) -> None:
    from molgnn_ops.cli import app

    input_csv = tmp_path / "prepared.csv"
    output_jsonl = tmp_path / "graphs.jsonl"
    pd.DataFrame({"smiles": ["CCO", "invalid"]}).to_csv(input_csv, index=False)

    result = CliRunner().invoke(
        app,
        ["featurize-csv", str(input_csv), str(output_jsonl)],
    )

    assert result.exit_code == 0
    assert "Graphs written: 1" in result.output
    assert output_jsonl.is_file()


def test_fingerprint_csv_command_smoke(tmp_path: Path) -> None:
    from molgnn_ops.cli import app

    input_csv = tmp_path / "prepared.csv"
    output_npz = tmp_path / "fingerprints.npz"
    pd.DataFrame(
        {
            "smiles": ["CCO", "invalid"],
            "target": [1.0, 0.0],
            "split": ["train", "test"],
            "dataset_name": ["example", "example"],
        }
    ).to_csv(input_csv, index=False)

    result = CliRunner().invoke(
        app,
        ["fingerprint-csv", str(input_csv), str(output_npz), "--n-bits", "64"],
    )

    assert result.exit_code == 0
    assert "Rows written: 1" in result.output
    assert output_npz.is_file()


def test_train_fingerprint_baseline_command_smoke(tmp_path: Path) -> None:
    from molgnn_ops.cli import app

    input_npz = tmp_path / "fingerprints.npz"
    output_dir = tmp_path / "baseline"
    indices = np.arange(12)
    np.savez_compressed(
        input_npz,
        X=np.column_stack([indices % 2, (indices // 2) % 2]),
        y=(indices % 2).astype(float),
        splits=np.asarray(["train"] * 8 + ["val"] * 2 + ["test"] * 2),
        smiles=np.asarray([f"molecule-{index}" for index in indices]),
        dataset_name=np.asarray(["example"] * len(indices)),
    )

    result = CliRunner().invoke(
        app,
        [
            "train-fingerprint-baseline",
            str(input_npz),
            str(output_dir),
            "--task-type",
            "classification",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "Best model:" in result.output
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "report.md").is_file()
