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
