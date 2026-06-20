from pathlib import Path

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
