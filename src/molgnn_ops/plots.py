import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from molgnn_ops.diagnostics import compute_test_max_similarities


def _get_pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "molgnn_ops_matplotlib"),
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save_figure(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    _get_pyplot().close(figure)


def plot_target_distribution(prepared_csv: Path, output_path: Path) -> None:
    """Plot target histograms for each available split."""
    dataframe = pd.read_csv(prepared_csv)
    if not {"target", "split"} <= set(dataframe.columns):
        raise ValueError("Prepared CSV must contain target and split columns")
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name, split_frame in dataframe.groupby("split", sort=True):
        axis.hist(split_frame["target"].dropna(), bins=20, alpha=0.5, label=str(split_name))
    axis.set_title("Target Distribution by Split")
    axis.set_xlabel("Target")
    axis.set_ylabel("Count")
    axis.legend()
    _save_figure(figure, output_path)


def plot_predicted_vs_actual(
    predictions_csv: Path,
    output_path: Path,
    split: str = "test",
) -> None:
    """Plot predicted values against observed values for one split."""
    dataframe = pd.read_csv(predictions_csv)
    selected = dataframe[dataframe["split"] == split].dropna(subset=["y_true", "y_pred"])
    if selected.empty:
        raise ValueError(f"Predictions CSV contains no usable rows for split '{split}'")
    lower = float(min(selected["y_true"].min(), selected["y_pred"].min()))
    upper = float(max(selected["y_true"].max(), selected["y_pred"].max()))

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(selected["y_true"], selected["y_pred"], alpha=0.7)
    axis.plot([lower, upper], [lower, upper], linestyle="--", label="Ideal")
    axis.set_title(f"Predicted vs Actual ({split})")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.legend()
    _save_figure(figure, output_path)


def plot_absolute_error_histogram(
    predictions_csv: Path,
    output_path: Path,
    split: str = "test",
) -> None:
    """Plot the absolute prediction-error distribution for one split."""
    dataframe = pd.read_csv(predictions_csv)
    selected = dataframe[dataframe["split"] == split].dropna(subset=["y_true", "y_pred"])
    if selected.empty:
        raise ValueError(f"Predictions CSV contains no usable rows for split '{split}'")
    absolute_errors = np.abs(selected["y_pred"] - selected["y_true"])

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(absolute_errors, bins=20)
    axis.set_title(f"Absolute Error Distribution ({split})")
    axis.set_xlabel("Absolute error")
    axis.set_ylabel("Count")
    _save_figure(figure, output_path)


def plot_test_similarity_histogram(prepared_csv: Path, output_path: Path) -> None:
    """Plot maximum train-set Morgan similarities for test molecules."""
    similarities = compute_test_max_similarities(prepared_csv)
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(similarities, bins=20)
    axis.set_title("Test-to-Train Maximum Morgan Similarity")
    axis.set_xlabel("Maximum Tanimoto similarity")
    axis.set_ylabel("Test molecule count")
    _save_figure(figure, output_path)


def plot_gnn_rmse_by_model(comparison_csv: Path, output_path: Path) -> None:
    """Plot mean test RMSE by model with one standard deviation error bars."""
    dataframe = pd.read_csv(comparison_csv)
    required_columns = {"model_name", "test_rmse"}
    if not required_columns <= set(dataframe.columns):
        raise ValueError("Comparison CSV must contain model_name and test_rmse columns")
    grouped = dataframe.groupby("model_name", sort=True)["test_rmse"]
    means = grouped.mean()
    standard_deviations = grouped.std(ddof=0)

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        means.index.tolist(),
        means.to_numpy(),
        yerr=standard_deviations.to_numpy(),
        capsize=4,
    )
    axis.set_title("Repeated-Seed GNN Test RMSE")
    axis.set_xlabel("Model")
    axis.set_ylabel("Test RMSE")
    _save_figure(figure, output_path)


def plot_gnn_metric_by_seed(
    comparison_csv: Path,
    output_path: Path,
    metric: str = "test_rmse",
) -> None:
    """Plot one comparison metric across seeds for every GNN model."""
    dataframe = pd.read_csv(comparison_csv)
    required_columns = {"model_name", "seed", metric}
    if not required_columns <= set(dataframe.columns):
        raise ValueError(
            "Comparison CSV must contain model_name, seed, and " f"{metric} columns"
        )

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, model_frame in dataframe.groupby("model_name", sort=True):
        ordered = model_frame.sort_values("seed")
        axis.plot(
            ordered["seed"],
            ordered[metric],
            marker="o",
            label=str(model_name),
        )
    axis.set_title(f"{metric.replace('_', ' ').title()} by Seed")
    axis.set_xlabel("Seed")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend()
    _save_figure(figure, output_path)


def plot_uncertainty_vs_error(predictions_csv: Path, output_path: Path) -> None:
    """Plot ensemble disagreement against absolute prediction error."""
    dataframe = pd.read_csv(predictions_csv)
    required = {"ensemble_std", "absolute_error"}
    if not required <= set(dataframe.columns):
        raise ValueError("Predictions CSV must contain ensemble_std and absolute_error")
    if "split" in dataframe and (dataframe["split"] == "test").any():
        dataframe = dataframe[dataframe["split"] == "test"]

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(dataframe["ensemble_std"], dataframe["absolute_error"], alpha=0.7)
    axis.set_title("Ensemble Uncertainty vs Absolute Error")
    axis.set_xlabel("Ensemble standard deviation")
    axis.set_ylabel("Absolute error")
    _save_figure(figure, output_path)


def plot_interval_coverage(
    coverage_results: pd.DataFrame | Path,
    output_path: Path,
) -> None:
    """Plot nominal regression interval coverage against empirical coverage."""
    dataframe = (
        pd.read_csv(coverage_results)
        if isinstance(coverage_results, Path)
        else coverage_results.copy()
    )
    required = {"target_coverage", "empirical_coverage"}
    if not required <= set(dataframe.columns):
        raise ValueError(
            "Coverage results must contain target_coverage and empirical_coverage"
        )
    ordered = dataframe.sort_values("target_coverage")

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        ordered["target_coverage"],
        ordered["empirical_coverage"],
        marker="o",
        label="Empirical test coverage",
    )
    limits = [0.0, 1.0]
    axis.plot(limits, limits, linestyle="--", label="Nominal coverage")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_title("Regression Prediction Interval Coverage")
    axis.set_xlabel("Target coverage")
    axis.set_ylabel("Empirical test coverage")
    axis.legend()
    _save_figure(figure, output_path)


def plot_selective_prediction_curve(
    selective_metrics_csv: Path,
    output_path: Path,
) -> None:
    """Plot RMSE as increasingly uncertain predictions are retained."""
    dataframe = pd.read_csv(selective_metrics_csv)
    required = {"retained_fraction", "rmse"}
    if not required <= set(dataframe.columns):
        raise ValueError("Selective metrics CSV must contain retained_fraction and rmse")
    ordered = dataframe.sort_values("retained_fraction")

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(ordered["retained_fraction"], ordered["rmse"], marker="o")
    axis.set_title("Selective Prediction Performance")
    axis.set_xlabel("Retained fraction (least uncertain first)")
    axis.set_ylabel("RMSE")
    axis.set_xlim(0.0, 1.0)
    _save_figure(figure, output_path)


def plot_uncertainty_buckets(
    bucket_summary_csv: Path,
    output_path: Path,
) -> None:
    """Plot RMSE for low, medium, and high ensemble-uncertainty buckets."""
    dataframe = pd.read_csv(bucket_summary_csv)
    required = {"bucket", "rmse"}
    if not required <= set(dataframe.columns):
        raise ValueError("Bucket summary CSV must contain bucket and rmse")
    order = ["low", "medium", "high"]
    dataframe["bucket"] = pd.Categorical(dataframe["bucket"], order, ordered=True)
    ordered = dataframe.sort_values("bucket")

    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(ordered["bucket"].astype(str), ordered["rmse"])
    axis.set_title("Error by Ensemble-Uncertainty Bucket")
    axis.set_xlabel("Uncertainty bucket")
    axis.set_ylabel("RMSE")
    _save_figure(figure, output_path)
