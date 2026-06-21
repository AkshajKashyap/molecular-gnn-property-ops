from pathlib import Path

import pandas as pd

from molgnn_ops.plots import (
    plot_absolute_error_histogram,
    plot_gnn_metric_by_seed,
    plot_gnn_rmse_by_model,
    plot_interval_coverage,
    plot_predicted_vs_actual,
    plot_selective_prediction_curve,
    plot_target_distribution,
    plot_test_similarity_histogram,
    plot_uncertainty_buckets,
    plot_uncertainty_vs_error,
)


def test_diagnostic_plots_create_files(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CCCl", "c1ccccc1O", "C1CCCCC1"],
            "target": [1.0, 2.0, 3.0, 1.5, 3.5, 2.5],
            "split": ["train", "train", "train", "test", "test", "test"],
        }
    ).to_csv(prepared_csv, index=False)
    pd.DataFrame(
        {
            "smiles": ["CCCl", "c1ccccc1O", "C1CCCCC1"],
            "split": ["test"] * 3,
            "y_true": [1.5, 3.5, 2.5],
            "y_pred": [1.8, 3.0, 2.1],
        }
    ).to_csv(predictions_csv, index=False)

    output_paths = [
        tmp_path / "figures" / "targets.png",
        tmp_path / "figures" / "predicted.png",
        tmp_path / "figures" / "errors.png",
        tmp_path / "figures" / "similarity.png",
    ]
    plot_target_distribution(prepared_csv, output_paths[0])
    plot_predicted_vs_actual(predictions_csv, output_paths[1])
    plot_absolute_error_histogram(predictions_csv, output_paths[2])
    plot_test_similarity_histogram(prepared_csv, output_paths[3])

    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)


def test_gnn_comparison_plots_create_files(tmp_path: Path) -> None:
    comparison_csv = tmp_path / "comparison.csv"
    pd.DataFrame(
        {
            "model_name": ["gcn", "gcn", "gin", "gin"],
            "seed": [42, 43, 42, 43],
            "test_rmse": [1.1, 1.3, 1.4, 1.2],
            "test_mae": [0.8, 0.9, 1.0, 0.9],
        }
    ).to_csv(comparison_csv, index=False)
    rmse_path = tmp_path / "figures" / "rmse.png"
    seed_path = tmp_path / "figures" / "mae_by_seed.png"

    plot_gnn_rmse_by_model(comparison_csv, rmse_path)
    plot_gnn_metric_by_seed(comparison_csv, seed_path, metric="test_mae")

    assert rmse_path.is_file() and rmse_path.stat().st_size > 0
    assert seed_path.is_file() and seed_path.stat().st_size > 0


def test_uncertainty_plots_create_files(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    interval_csv = tmp_path / "intervals.csv"
    selective_csv = tmp_path / "selective.csv"
    buckets_csv = tmp_path / "buckets.csv"
    pd.DataFrame(
        {
            "split": ["test"] * 4,
            "ensemble_std": [0.1, 0.2, 0.4, 0.8],
            "absolute_error": [0.2, 0.1, 0.5, 1.0],
        }
    ).to_csv(predictions_csv, index=False)
    pd.DataFrame(
        {
            "target_coverage": [0.8, 0.9, 0.95],
            "empirical_coverage": [0.75, 0.88, 0.94],
        }
    ).to_csv(interval_csv, index=False)
    pd.DataFrame(
        {
            "retained_fraction": [0.25, 0.5, 0.75, 1.0],
            "rmse": [0.5, 0.6, 0.8, 1.0],
        }
    ).to_csv(selective_csv, index=False)
    pd.DataFrame(
        {
            "bucket": ["low", "medium", "high"],
            "rmse": [0.5, 0.8, 1.2],
        }
    ).to_csv(buckets_csv, index=False)
    output_paths = [
        tmp_path / "uncertainty.png",
        tmp_path / "coverage.png",
        tmp_path / "selective.png",
        tmp_path / "buckets.png",
    ]

    plot_uncertainty_vs_error(predictions_csv, output_paths[0])
    plot_interval_coverage(interval_csv, output_paths[1])
    plot_selective_prediction_curve(selective_csv, output_paths[2])
    plot_uncertainty_buckets(buckets_csv, output_paths[3])

    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)
