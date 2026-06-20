import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgnn_ops.baselines import infer_task_type, train_fingerprint_baseline
from molgnn_ops.reporting import write_markdown_report


def _write_fingerprint_dataset(path: Path, task_type: str) -> None:
    indices = np.arange(12)
    features = np.column_stack(
        [
            indices % 2,
            (indices // 2) % 2,
            (indices // 4) % 2,
            (indices // 8) % 2,
        ]
    ).astype(np.uint8)
    if task_type == "classification":
        targets = (indices % 2).astype(float)
    else:
        targets = indices.astype(float) + 0.25
    splits = np.asarray(["train"] * 8 + ["val"] * 2 + ["test"] * 2)
    np.savez_compressed(
        path,
        X=features,
        y=targets,
        splits=splits,
        smiles=np.asarray([f"molecule-{index}" for index in indices]),
        dataset_name=np.asarray(["synthetic"] * len(indices)),
    )


def test_infer_task_type_detects_binary_classification() -> None:
    assert infer_task_type(np.asarray([0.0, 1.0, 0.0, 1.0])) == "classification"


def test_infer_task_type_detects_regression() -> None:
    assert infer_task_type(np.asarray([0.1, 1.4, 2.8, 4.2])) == "regression"


def test_train_fingerprint_baseline_classification(tmp_path: Path) -> None:
    input_npz = tmp_path / "classification.npz"
    output_dir = tmp_path / "classification_run"
    _write_fingerprint_dataset(input_npz, "classification")

    metrics = train_fingerprint_baseline(input_npz, output_dir, seed=7)
    predictions = pd.read_csv(output_dir / "predictions.csv")

    assert metrics["task_type"] == "classification"
    assert metrics["best_model"] in {"logistic_regression", "random_forest"}
    assert set(predictions.columns) == {"smiles", "split", "y_true", "y_pred", "y_score"}
    assert (output_dir / "models" / "fingerprint_baseline.joblib").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "report.md").is_file()


def test_train_fingerprint_baseline_regression(tmp_path: Path) -> None:
    input_npz = tmp_path / "regression.npz"
    output_dir = tmp_path / "regression_run"
    _write_fingerprint_dataset(input_npz, "regression")

    metrics = train_fingerprint_baseline(
        input_npz,
        output_dir,
        task_type="regression",
        seed=7,
    )
    predictions = pd.read_csv(output_dir / "predictions.csv")

    assert metrics["task_type"] == "regression"
    assert metrics["best_model"] in {"ridge", "random_forest"}
    assert set(metrics["test_metrics"]) == {"mae", "rmse", "r2"}
    assert set(predictions.columns) == {"smiles", "split", "y_true", "y_pred"}
    assert (output_dir / "models" / "fingerprint_baseline.joblib").is_file()


def test_write_markdown_report(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "baseline.md"
    metrics = {
        "task_type": "classification",
        "best_model": "logistic_regression",
        "selection_metric": "accuracy",
        "models": {
            "logistic_regression": {"validation": {"accuracy": 0.8}},
            "random_forest": {"validation": {"accuracy": 0.7}},
        },
        "test_metrics": {"accuracy": 0.75},
    }

    write_markdown_report(metrics, output_path, "Baseline Report")
    report = output_path.read_text(encoding="utf-8")

    assert "# Baseline Report" in report
    assert "logistic_regression" in report
    assert "## Test Metrics" in report


def test_saved_metrics_are_valid_json(tmp_path: Path) -> None:
    input_npz = tmp_path / "classification.npz"
    output_dir = tmp_path / "run"
    _write_fingerprint_dataset(input_npz, "classification")

    train_fingerprint_baseline(input_npz, output_dir)

    with (output_dir / "metrics.json").open(encoding="utf-8") as metrics_file:
        assert json.load(metrics_file)["task_type"] == "classification"
