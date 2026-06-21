from pathlib import Path

import pandas as pd

from molgnn_ops.plots import (
    plot_absolute_error_histogram,
    plot_gnn_metric_by_seed,
    plot_gnn_rmse_by_model,
    plot_predicted_vs_actual,
    plot_target_distribution,
    plot_test_similarity_histogram,
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
