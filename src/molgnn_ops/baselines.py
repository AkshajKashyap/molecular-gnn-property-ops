import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from molgnn_ops.reporting import write_markdown_report

_NPZ_KEYS = ("X", "y", "splits", "smiles", "dataset_name")
_IDENTITY_KEYS = ("sample_id", "canonical_smiles")


def load_fingerprint_npz(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a compressed fingerprint dataset without pickle."""
    if not path.is_file():
        raise FileNotFoundError(f"Fingerprint dataset not found: {path}")

    with np.load(path, allow_pickle=False) as dataset:
        missing_keys = [key for key in _NPZ_KEYS if key not in dataset]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Required arrays missing from {path}: {missing}")
        arrays = {key: dataset[key].copy() for key in _NPZ_KEYS}
        for key in _IDENTITY_KEYS:
            if key in dataset:
                arrays[key] = dataset[key].copy()

    row_count = len(arrays["y"])
    if arrays["X"].ndim != 2:
        raise ValueError("Fingerprint array X must be two-dimensional")
    if any(len(arrays[key]) != row_count for key in _NPZ_KEYS[2:]):
        raise ValueError("Fingerprint dataset arrays must have matching row counts")
    if len(arrays["X"]) != row_count:
        raise ValueError("Fingerprint dataset arrays must have matching row counts")
    if "sample_id" not in arrays:
        arrays["sample_id"] = np.asarray(
            [f"legacy:{index}" for index in range(row_count)],
            dtype=str,
        )
    if "canonical_smiles" not in arrays:
        arrays["canonical_smiles"] = arrays["smiles"].copy()
    if any(len(arrays[key]) != row_count for key in _IDENTITY_KEYS):
        raise ValueError("Fingerprint identity arrays must have matching row counts")
    if len(set(arrays["sample_id"].tolist())) != row_count:
        raise ValueError("Fingerprint sample_id values must be unique")
    return arrays


def infer_task_type(y) -> str:
    """Infer classification for small integer-like targets, regression otherwise."""
    targets = np.asarray(y, dtype=float)
    if targets.size == 0:
        raise ValueError("Cannot infer task type from an empty target array")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Targets must contain only finite values")

    unique_targets = np.unique(targets)
    integer_like = np.allclose(unique_targets, np.round(unique_targets))
    if integer_like and len(unique_targets) <= 20:
        return "classification"
    return "regression"


def _classification_predictions(
    model: Any,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = model.predict(features)
    if not hasattr(model, "predict_proba") or len(model.classes_) != 2:
        return predictions, None
    probabilities = model.predict_proba(features)
    return predictions, probabilities[:, 1]


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    classes: np.ndarray,
) -> dict[str, float]:
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    observed_labels = set(np.unique(y_true).tolist())
    model_labels = set(classes.tolist())
    if y_score is not None and len(observed_labels) == 2 and observed_labels <= model_labels:
        binary_targets = (y_true == classes[1]).astype(int)
        metrics["roc_auc"] = float(roc_auc_score(binary_targets, y_score))
        metrics["average_precision"] = float(
            average_precision_score(binary_targets, y_score)
        )
    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def _validate_training_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    split_masks = {name: data["splits"] == name for name in ("train", "val", "test")}
    empty_splits = [name for name, mask in split_masks.items() if not np.any(mask)]
    if empty_splits:
        missing = ", ".join(empty_splits)
        raise ValueError(f"Fingerprint dataset has no rows for split(s): {missing}")
    if not np.all(np.isfinite(data["y"])):
        raise ValueError("Targets must contain only finite values")
    return split_masks


def train_fingerprint_baseline(
    input_npz: Path,
    output_dir: Path,
    task_type: str | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Train, select, evaluate, and persist deterministic fingerprint baselines."""
    data = load_fingerprint_npz(input_npz)
    split_masks = _validate_training_data(data)
    if task_type in {None, "auto"}:
        resolved_task_type = infer_task_type(data["y"])
    elif task_type in {"classification", "regression"}:
        resolved_task_type = task_type
    else:
        raise ValueError("task_type must be 'auto', 'classification', or 'regression'")

    train_mask = split_masks["train"]
    val_mask = split_masks["val"]
    test_mask = split_masks["test"]
    train_features = data["X"][train_mask]
    train_targets = data["y"][train_mask]
    val_features = data["X"][val_mask]
    val_targets = data["y"][val_mask]

    if resolved_task_type == "classification":
        if len(np.unique(train_targets)) < 2:
            raise ValueError("Classification training requires at least two target classes")
        candidate_models: dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=50,
                random_state=seed,
                n_jobs=1,
            ),
        }
        selection_metric = "accuracy"
    else:
        candidate_models = {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                random_state=seed,
                n_jobs=1,
            ),
        }
        selection_metric = "rmse"

    validation_results: dict[str, dict[str, dict[str, float | None]]] = {}
    for model_name, model in candidate_models.items():
        model.fit(train_features, train_targets)
        if resolved_task_type == "classification":
            predictions, scores = _classification_predictions(model, val_features)
            validation_metrics = _classification_metrics(
                val_targets,
                predictions,
                scores,
                np.asarray(model.classes_),
            )
        else:
            predictions = model.predict(val_features)
            validation_metrics = _regression_metrics(val_targets, predictions)
        validation_results[model_name] = {"validation": validation_metrics}

    if resolved_task_type == "classification":
        best_model_name = max(
            candidate_models,
            key=lambda name: validation_results[name]["validation"][selection_metric],
        )
    else:
        best_model_name = min(
            candidate_models,
            key=lambda name: validation_results[name]["validation"][selection_metric],
        )
    best_model = candidate_models[best_model_name]

    prediction_frames: list[pd.DataFrame] = []
    test_metrics: dict[str, float | None] = {}
    for split_name, split_mask in (("val", val_mask), ("test", test_mask)):
        features = data["X"][split_mask]
        targets = data["y"][split_mask]
        if resolved_task_type == "classification":
            predictions, scores = _classification_predictions(best_model, features)
            frame_data: dict[str, object] = {
                "sample_id": data["sample_id"][split_mask],
                "smiles": data["smiles"][split_mask],
                "canonical_smiles": data["canonical_smiles"][split_mask],
                "split": data["splits"][split_mask],
                "y_true": targets,
                "y_pred": predictions,
            }
            if scores is not None:
                frame_data["y_score"] = scores
            if split_name == "test":
                test_metrics = _classification_metrics(
                    targets,
                    predictions,
                    scores,
                    np.asarray(best_model.classes_),
                )
        else:
            predictions = best_model.predict(features)
            frame_data = {
                "sample_id": data["sample_id"][split_mask],
                "smiles": data["smiles"][split_mask],
                "canonical_smiles": data["canonical_smiles"][split_mask],
                "split": data["splits"][split_mask],
                "y_true": targets,
                "y_pred": predictions,
            }
            if split_name == "test":
                test_metrics = _regression_metrics(targets, predictions)
        prediction_frames.append(pd.DataFrame(frame_data))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "models" / "fingerprint_baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"

    joblib.dump(best_model, model_path)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)

    metrics: dict[str, object] = {
        "task_type": resolved_task_type,
        "seed": seed,
        "selection_metric": selection_metric,
        "best_model": best_model_name,
        "models": validation_results,
        "test_metrics": test_metrics,
        "split_counts": {
            name: int(mask.sum()) for name, mask in split_masks.items()
        },
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "report": str(report_path),
        },
    }
    write_markdown_report(metrics, report_path, "Morgan Fingerprint Baseline")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics
