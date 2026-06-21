import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_PREDICTION_COLUMNS = {"smiles", "split", "y_true", "y_pred"}


def _validate_prediction_columns(dataframe: pd.DataFrame, source: Path | str) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Prediction file {source} is missing columns: {', '.join(missing)}")


def load_ensemble_predictions(prediction_paths: list[Path]) -> pd.DataFrame:
    """Load repeated-run predictions and strictly align them by molecule and split."""
    if not prediction_paths:
        raise ValueError("At least one prediction path is required")

    aligned: pd.DataFrame | None = None
    reference_keys: pd.MultiIndex | None = None
    for run_index, path in enumerate(prediction_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV does not exist: {path}")
        dataframe = pd.read_csv(path)
        _validate_prediction_columns(dataframe, path)
        if dataframe[["smiles", "split"]].isna().any(axis=None):
            raise ValueError(f"Prediction file {path} contains missing molecule keys")
        duplicated = dataframe.duplicated(["smiles", "split"], keep=False)
        if duplicated.any():
            examples = dataframe.loc[duplicated, ["smiles", "split"]].head(3)
            formatted = ", ".join(
                f"({row.smiles}, {row.split})" for row in examples.itertuples()
            )
            raise ValueError(f"Prediction file {path} contains duplicate molecules: {formatted}")

        current = dataframe.loc[:, ["smiles", "split", "y_true", "y_pred"]].copy()
        current_keys = pd.MultiIndex.from_frame(current[["smiles", "split"]])
        if reference_keys is None:
            reference_keys = current_keys
            aligned = current.rename(columns={"y_pred": f"y_pred_run_{run_index}"})
            continue

        if set(current_keys) != set(reference_keys):
            missing_count = len(set(reference_keys).difference(current_keys))
            extra_count = len(set(current_keys).difference(reference_keys))
            raise ValueError(
                f"Prediction file {path} has mismatched molecules "
                f"({missing_count} missing, {extra_count} unexpected)"
            )

        assert aligned is not None
        current = current.rename(
            columns={
                "y_true": f"y_true_run_{run_index}",
                "y_pred": f"y_pred_run_{run_index}",
            }
        )
        aligned = aligned.merge(
            current,
            on=["smiles", "split"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        other_targets = aligned.pop(f"y_true_run_{run_index}")
        if not np.allclose(
            aligned["y_true"].to_numpy(dtype=float),
            other_targets.to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-8,
            equal_nan=False,
        ):
            raise ValueError(f"Prediction file {path} has mismatched y_true values")

    assert aligned is not None
    return aligned


def _prediction_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = sorted(
        (column for column in dataframe.columns if column.startswith("y_pred_run_")),
        key=lambda column: int(column.rsplit("_", maxsplit=1)[1]),
    )
    if not columns:
        raise ValueError("Aligned predictions contain no ensemble prediction columns")
    return columns


def compute_ensemble_predictions(aligned_predictions: pd.DataFrame) -> pd.DataFrame:
    """Add ensemble point predictions, disagreement, and observed errors."""
    if "y_true" not in aligned_predictions:
        raise ValueError("Aligned predictions must contain y_true")
    result = aligned_predictions.copy()
    prediction_columns = _prediction_columns(result)
    result["ensemble_mean"] = result[prediction_columns].mean(axis=1)
    if len(prediction_columns) >= 2:
        result["ensemble_std"] = result[prediction_columns].std(axis=1, ddof=1)
    else:
        result["ensemble_std"] = 0.0
    errors = result["y_true"] - result["ensemble_mean"]
    result["absolute_error"] = errors.abs()
    result["squared_error"] = errors.pow(2)
    result["number_of_models"] = len(prediction_columns)
    return result


def fit_interval_scale(
    validation_predictions: pd.DataFrame,
    target_coverage: float = 0.90,
    epsilon: float = 1e-6,
) -> float:
    """Fit a validation-calibrated ensemble interval scale.

    This empirical scale is calibrated on held-out validation residuals. It is not a
    formal coverage guarantee under arbitrary distribution shift.
    """
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")
    required = {"y_true", "ensemble_mean", "ensemble_std"}
    if not required <= set(validation_predictions.columns):
        raise ValueError(
            "Validation predictions must contain y_true, ensemble_mean, and ensemble_std"
        )
    if validation_predictions.empty:
        raise ValueError("Validation predictions must not be empty")

    denominator = np.maximum(
        validation_predictions["ensemble_std"].to_numpy(dtype=float),
        epsilon,
    )
    scores = np.abs(
        validation_predictions["y_true"].to_numpy(dtype=float)
        - validation_predictions["ensemble_mean"].to_numpy(dtype=float)
    ) / denominator
    sorted_scores = np.sort(scores)
    quantile_rank = math.ceil((len(sorted_scores) + 1) * target_coverage)
    return float(sorted_scores[min(quantile_rank, len(sorted_scores)) - 1])


def add_prediction_intervals(
    predictions: pd.DataFrame,
    interval_scale: float,
) -> pd.DataFrame:
    """Add symmetric ensemble-disagreement intervals and coverage indicators."""
    if interval_scale < 0 or not math.isfinite(interval_scale):
        raise ValueError("interval_scale must be a finite non-negative number")
    required = {"y_true", "ensemble_mean", "ensemble_std"}
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for intervals")
    result = predictions.copy()
    half_width = interval_scale * result["ensemble_std"]
    result["interval_lower"] = result["ensemble_mean"] - half_width
    result["interval_upper"] = result["ensemble_mean"] + half_width
    result["interval_width"] = 2 * half_width
    result["covered"] = result["y_true"].between(
        result["interval_lower"],
        result["interval_upper"],
        inclusive="both",
    )
    return result


def interval_metrics(predictions: pd.DataFrame) -> dict:
    """Summarize regression accuracy, empirical coverage, and interval width."""
    required = {
        "y_true",
        "ensemble_mean",
        "interval_width",
        "covered",
    }
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for interval metrics")
    if predictions.empty:
        raise ValueError("Predictions must not be empty")
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["ensemble_mean"].to_numpy(dtype=float)
    return {
        "n": len(predictions),
        "empirical_coverage": float(predictions["covered"].mean()),
        "mean_interval_width": float(predictions["interval_width"].mean()),
        "median_interval_width": float(predictions["interval_width"].median()),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(predictions) >= 2 else None,
    }


def _finite_correlation(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def uncertainty_error_correlation(predictions: pd.DataFrame) -> dict:
    """Correlate ensemble disagreement with absolute prediction error."""
    required = {"ensemble_std", "absolute_error"}
    if not required <= set(predictions.columns):
        raise ValueError("Predictions must contain ensemble_std and absolute_error")
    if predictions.empty:
        raise ValueError("Predictions must not be empty")
    uncertainty = predictions["ensemble_std"].astype(float)
    error = predictions["absolute_error"].astype(float)
    pearson = uncertainty.corr(error, method="pearson")
    spearman_style = uncertainty.rank(method="average").corr(
        error.rank(method="average"),
        method="pearson",
    )
    return {
        "pearson": _finite_correlation(pearson),
        "spearman": _finite_correlation(spearman_style),
    }


def selective_prediction_metrics(
    predictions: pd.DataFrame,
    coverage_levels: list[float] | None = None,
) -> list[dict]:
    """Evaluate accuracy after retaining the least-uncertain predictions."""
    levels = coverage_levels or [0.25, 0.50, 0.75, 1.00]
    if not levels or any(not 0 < level <= 1 for level in levels):
        raise ValueError("coverage_levels must contain values in the range (0, 1]")
    required = {"y_true", "ensemble_mean", "ensemble_std"}
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for selective metrics")
    if predictions.empty:
        raise ValueError("Predictions must not be empty")

    ordered = predictions.sort_values("ensemble_std", kind="stable")
    results = []
    for retained_fraction in levels:
        n_retained = max(1, math.ceil(len(ordered) * retained_fraction))
        retained = ordered.iloc[:n_retained]
        errors = retained["y_true"] - retained["ensemble_mean"]
        results.append(
            {
                "retained_fraction": retained_fraction,
                "n_retained": n_retained,
                "rmse": float(np.sqrt(np.mean(errors.pow(2)))),
                "mae": float(errors.abs().mean()),
                "mean_uncertainty": float(retained["ensemble_std"].mean()),
            }
        )
    return results
