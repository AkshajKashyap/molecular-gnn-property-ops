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


def test_list_datasets_command_smoke() -> None:
    from molgnn_ops.cli import app

    result = CliRunner().invoke(app, ["list-datasets"])

    assert result.exit_code == 0
    assert "esol" in result.output
    assert "regression" in result.output


def test_download_dataset_command_smoke(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import cli as cli_module

    output_path = tmp_path / "delaney-processed.csv"
    monkeypatch.setattr(
        cli_module,
        "download_dataset",
        lambda name, overwrite=False: output_path,
    )

    result = CliRunner().invoke(cli_module.app, ["download-dataset", "esol"])

    assert result.exit_code == 0
    assert str(output_path) in result.output.replace("\n", "")


def test_run_fingerprint_benchmark_command_smoke(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import cli as cli_module

    summary = {
        "dataset_name": "esol",
        "task_type": "regression",
        "split_strategy": "scaffold",
        "best_model": "ridge",
        "key_metric": "rmse",
        "validation_metric": 0.5,
        "test_metric": 0.6,
        "metrics_json": str(tmp_path / "metrics.json"),
        "report_md": str(tmp_path / "report.md"),
        "summary_json": str(tmp_path / "benchmark_summary.json"),
    }
    monkeypatch.setattr(
        cli_module,
        "run_fingerprint_benchmark",
        lambda *args, **kwargs: summary,
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["run-fingerprint-benchmark", "esol", "--seed", "42"],
    )

    assert result.exit_code == 0
    assert "Best model: ridge" in result.output
    assert "Test rmse: 0.6" in result.output


def test_diagnose_benchmark_command_smoke(tmp_path: Path) -> None:
    from molgnn_ops.cli import app

    prepared_csv = tmp_path / "prepared.csv"
    predictions_csv = tmp_path / "predictions.csv"
    output_dir = tmp_path / "diagnostics"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "c1ccccc1O"],
            "target": [1.0, 2.0, 3.0, 1.5, 3.5],
            "split": ["train", "train", "train", "test", "test"],
        }
    ).to_csv(prepared_csv, index=False)
    pd.DataFrame(
        {
            "smiles": ["CCCl", "c1ccccc1O"],
            "split": ["test", "test"],
            "y_true": [1.5, 3.5],
            "y_pred": [1.8, 3.0],
        }
    ).to_csv(predictions_csv, index=False)

    result = CliRunner().invoke(
        app,
        ["diagnose-benchmark", str(prepared_csv), str(predictions_csv), str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Test RMSE:" in result.output
    assert (output_dir / "diagnostics.json").is_file()
    assert (output_dir / "diagnostics_report.md").is_file()


def test_compare_splits_command_smoke(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import cli as cli_module

    summary = {
        "by_split_strategy": {
            "random": {"key_metric": "rmse", "mean_test_metric": 0.8, "n_runs": 2},
            "scaffold": {"key_metric": "rmse", "mean_test_metric": 1.2, "n_runs": 2},
        },
        "comparison_metrics_csv": str(tmp_path / "comparison_metrics.csv"),
        "comparison_summary_json": str(tmp_path / "comparison_summary.json"),
        "comparison_report_md": str(tmp_path / "comparison_report.md"),
    }
    captured = {}

    def fake_comparison(*args, **kwargs):
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(cli_module, "run_split_comparison", fake_comparison)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "compare-splits",
            "esol",
            str(tmp_path),
            "--seeds",
            "42,43",
            "--split-strategies",
            "random,scaffold",
        ],
    )

    assert result.exit_code == 0
    assert captured["seeds"] == [42, 43]
    assert captured["split_strategies"] == ["random", "scaffold"]
    assert "scaffold: mean test rmse=1.2" in result.output


def test_train_gnn_regressor_command_smoke(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import cli as cli_module
    from molgnn_ops import gnn_train

    summary = {
        "model_name": "gcn",
        "device": "cpu",
        "best_epoch": 2,
        "best_val_rmse": 1.1,
        "test_rmse": 1.2,
        "artifacts": {
            "metrics": str(tmp_path / "metrics.json"),
            "report": str(tmp_path / "report.md"),
        },
    }
    monkeypatch.setattr(gnn_train, "train_gnn_regressor", lambda *args, **kwargs: summary)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "train-gnn-regressor",
            str(tmp_path / "graphs.jsonl"),
            str(tmp_path / "output"),
            "--epochs",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Best validation RMSE: 1.1" in result.output
    assert "Test RMSE: 1.2" in result.output


def test_run_gnn_benchmark_command_smoke(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import cli as cli_module

    summary = {
        "dataset_name": "esol",
        "model_name": "gcn",
        "split_strategy": "scaffold",
        "best_epoch": 4,
        "best_val_rmse": 1.3,
        "test_rmse": 1.4,
        "metrics_json": str(tmp_path / "metrics.json"),
        "report_md": str(tmp_path / "report.md"),
        "summary_json": str(tmp_path / "summary.json"),
    }
    monkeypatch.setattr(cli_module, "run_gnn_benchmark", lambda *args, **kwargs: summary)

    result = CliRunner().invoke(
        cli_module.app,
        ["run-gnn-benchmark", "esol", str(tmp_path), "--epochs", "2"],
    )

    assert result.exit_code == 0
    assert "Completed molecular GNN benchmark" in result.output
    assert "Test RMSE: 1.4" in result.output
