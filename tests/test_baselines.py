import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgnn_ops.baselines import infer_task_type, train_fingerprint_baseline
from molgnn_ops.reporting import (
    write_diagnostics_report,
    write_gnn_uncertainty_report,
    write_markdown_report,
)


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
    assert set(predictions.columns) == {
        "sample_id",
        "smiles",
        "canonical_smiles",
        "split",
        "y_true",
        "y_pred",
        "y_score",
    }
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
    assert set(predictions.columns) == {
        "sample_id",
        "smiles",
        "canonical_smiles",
        "split",
        "y_true",
        "y_pred",
    }
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


def test_write_diagnostics_report(tmp_path: Path) -> None:
    output_path = tmp_path / "diagnostics.md"
    diagnostics = {
        "target_distribution": {"test": {"count": 2, "mean": 1.5}},
        "prediction_errors": {"test": {"n": 2, "mae": 0.4, "rmse": 0.5}},
        "worst_test_predictions": [
            {
                "smiles": "CCO",
                "y_true": 1.0,
                "y_pred": 1.7,
                "error": 0.7,
                "absolute_error": 0.7,
            }
        ],
        "scaffold_distribution": {
            "n_unique_scaffolds": 2,
            "largest_scaffold_group_size": 3,
            "median_scaffold_group_size": 2.0,
            "n_singleton_scaffolds": 1,
            "top_10_scaffold_groups": [{"scaffold": "ring", "size": 3}],
        },
        "train_test_similarity": {"mean_max_similarity": 0.45},
        "plots": {"target_distribution": "figures/targets.png"},
    }

    write_diagnostics_report(diagnostics, output_path)
    report = output_path.read_text(encoding="utf-8")

    assert "# Benchmark Diagnostics Report" in report
    assert "## Worst Test Predictions" in report
    assert "figures/targets.png" in report


def test_write_gnn_uncertainty_report(tmp_path: Path) -> None:
    summary = {
        "ensemble_members": 3,
        "seeds": [42, 43, 44],
        "ensemble_test_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        "interval_results": [
            {
                "target_coverage": 0.9,
                "interval_scale": 2.0,
                "empirical_coverage": 0.88,
                "mean_interval_width": 2.5,
                "median_interval_width": 2.2,
            }
        ],
        "uncertainty_error_correlations": {"pearson": 0.4, "spearman": 0.5},
        "selective_prediction": [
            {
                "retained_fraction": 1.0,
                "n_retained": 4,
                "rmse": 1.0,
                "mae": 0.8,
                "mean_uncertainty": 0.3,
            }
        ],
        "uncertainty_buckets": [
            {
                "bucket": "low",
                "n": 4,
                "mean_uncertainty": 0.3,
                "rmse": 1.0,
                "mae": 0.8,
                "empirical_coverage": 0.88,
            }
        ],
        "worst_predictions": [
            {
                "smiles": "CCO",
                "y_true": 1.0,
                "ensemble_mean": 0.0,
                "ensemble_std": 0.3,
                "absolute_error": 1.0,
                "interval_lower": -0.6,
                "interval_upper": 0.6,
                "covered": False,
                "molecular_weight": 46.1,
                "heavy_atom_count": 3,
                "ring_count": 0,
                "rotatable_bond_count": 0,
                "heteroatom_count": 1,
            }
        ],
        "descriptor_error_summary": {
            "molecular_weight": [
                {
                    "group": "low",
                    "n": 4,
                    "rmse": 1.0,
                    "mae": 0.8,
                    "mean_uncertainty": 0.3,
                    "empirical_coverage": 0.88,
                }
            ]
        },
        "plots": {"interval_coverage": "figures/coverage.png"},
    }
    output_path = tmp_path / "uncertainty.md"

    write_gnn_uncertainty_report(summary, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "# GNN Ensemble Uncertainty Report" in content
    assert "## Selective Prediction" in content
    assert "## Limitations" in content
    assert "may not maintain nominal coverage" in content
