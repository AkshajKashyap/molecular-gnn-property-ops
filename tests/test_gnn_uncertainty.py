from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molgnn_ops.gnn_uncertainty import (
    add_prediction_intervals,
    compute_ensemble_predictions,
    fit_interval_scale,
    interval_metrics,
    load_ensemble_predictions,
    selective_prediction_metrics,
    uncertainty_error_correlation,
)


def _write_prediction_runs(tmp_path: Path) -> list[Path]:
    first = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "split": ["val", "val", "test", "test"],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [0.8, 1.6, 2.5, 3.0],
        }
    )
    second = pd.DataFrame(
        {
            "smiles": ["CCCl", "CCC", "CCN", "CCO"],
            "split": ["test", "test", "val", "val"],
            "y_true": [4.0, 3.0, 2.0, 1.0],
            "y_pred": [5.0, 3.5, 2.4, 1.2],
        }
    )
    paths = [tmp_path / "run_0.csv", tmp_path / "run_1.csv"]
    first.to_csv(paths[0], index=False)
    second.to_csv(paths[1], index=False)
    return paths


def test_ensemble_predictions_align_and_compute_statistics(tmp_path: Path) -> None:
    aligned = load_ensemble_predictions(_write_prediction_runs(tmp_path))
    ensemble = compute_ensemble_predictions(aligned)

    assert aligned["smiles"].tolist() == ["CCO", "CCN", "CCC", "CCCl"]
    assert aligned["y_pred_run_1"].tolist() == [1.2, 2.4, 3.5, 5.0]
    cco = ensemble.iloc[0]
    assert cco["ensemble_mean"] == pytest.approx(1.0)
    assert cco["ensemble_std"] == pytest.approx(np.std([0.8, 1.2], ddof=1))
    assert cco["number_of_models"] == 2
    assert cco["absolute_error"] == pytest.approx(0.0)


def test_load_ensemble_predictions_rejects_mismatched_targets(tmp_path: Path) -> None:
    paths = _write_prediction_runs(tmp_path)
    dataframe = pd.read_csv(paths[1])
    dataframe.loc[dataframe["smiles"] == "CCO", "y_true"] = 9.0
    dataframe.to_csv(paths[1], index=False)

    with pytest.raises(ValueError, match="mismatched y_true"):
        load_ensemble_predictions(paths)


def test_load_ensemble_predictions_rejects_duplicate_molecules(tmp_path: Path) -> None:
    paths = _write_prediction_runs(tmp_path)
    dataframe = pd.read_csv(paths[0])
    dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)
    dataframe.to_csv(paths[0], index=False)

    with pytest.raises(ValueError, match="duplicate molecules"):
        load_ensemble_predictions(paths)


def test_load_ensemble_predictions_rejects_mismatched_molecules(tmp_path: Path) -> None:
    paths = _write_prediction_runs(tmp_path)
    dataframe = pd.read_csv(paths[1])
    dataframe.loc[dataframe["smiles"] == "CCO", "smiles"] = "COC"
    dataframe.to_csv(paths[1], index=False)

    with pytest.raises(ValueError, match="mismatched molecules"):
        load_ensemble_predictions(paths)


def test_interval_scale_intervals_and_metrics() -> None:
    validation = pd.DataFrame(
        {
            "y_true": [0.0, 2.0, 5.0],
            "ensemble_mean": [0.0, 1.0, 3.0],
            "ensemble_std": [1.0, 1.0, 1.0],
        }
    )
    interval_scale = fit_interval_scale(validation, target_coverage=0.90)
    with_intervals = add_prediction_intervals(validation, interval_scale)
    metrics = interval_metrics(with_intervals)

    assert interval_scale == pytest.approx(2.0)
    assert {
        "interval_lower",
        "interval_upper",
        "interval_width",
        "covered",
    } <= set(with_intervals.columns)
    assert metrics["empirical_coverage"] == pytest.approx(1.0)
    assert metrics["mean_interval_width"] == pytest.approx(4.0)


def test_uncertainty_correlations_and_selective_prediction() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "ensemble_mean": [0.1, 0.2, 1.0, 2.0],
            "ensemble_std": [0.1, 0.2, 0.8, 1.0],
            "absolute_error": [0.1, 0.2, 1.0, 2.0],
        }
    )
    correlations = uncertainty_error_correlation(predictions)
    selective = selective_prediction_metrics(predictions)

    assert set(correlations) == {"pearson", "spearman"}
    assert correlations["pearson"] > 0.9
    assert correlations["spearman"] == pytest.approx(1.0)
    assert selective[0]["n_retained"] == 1
    assert selective[0]["rmse"] == pytest.approx(0.1)
    assert selective[-1]["n_retained"] == 4
