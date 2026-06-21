from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def molecular_descriptors(smiles: str) -> dict:
    """Compute a compact set of interpretable RDKit molecular descriptors."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    return {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "ring_count": int(Lipinski.RingCount(molecule)),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(molecule)),
        "heteroatom_count": int(Lipinski.NumHeteroatoms(molecule)),
    }


def attach_molecular_descriptors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach RDKit descriptors to each prediction row."""
    if "smiles" not in predictions:
        raise ValueError("Predictions must contain a smiles column")
    descriptor_frame = pd.DataFrame(
        [molecular_descriptors(smiles) for smiles in predictions["smiles"]],
        index=predictions.index,
    )
    return pd.concat([predictions.copy(), descriptor_frame], axis=1)


def worst_ensemble_predictions(
    predictions: pd.DataFrame,
    n: int = 20,
) -> list[dict]:
    """Return the largest ensemble errors with intervals and descriptors."""
    if n <= 0:
        raise ValueError("n must be greater than 0")
    required = {
        "smiles",
        "y_true",
        "ensemble_mean",
        "ensemble_std",
        "absolute_error",
        "interval_lower",
        "interval_upper",
        "covered",
        "molecular_weight",
        "heavy_atom_count",
        "ring_count",
        "rotatable_bond_count",
        "heteroatom_count",
    }
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for error analysis")
    columns = [
        "smiles",
        "y_true",
        "ensemble_mean",
        "ensemble_std",
        "absolute_error",
        "interval_lower",
        "interval_upper",
        "covered",
        "molecular_weight",
        "heavy_atom_count",
        "ring_count",
        "rotatable_bond_count",
        "heteroatom_count",
    ]
    records = (
        predictions.sort_values("absolute_error", ascending=False, kind="stable")
        .head(n)
        .loc[:, columns]
        .to_dict(orient="records")
    )
    return [_to_builtin_record(record) for record in records]


def _to_builtin_record(record: dict[str, Any]) -> dict:
    converted = {}
    for key, value in record.items():
        if isinstance(value, np.generic):
            value = value.item()
        converted[key] = value
    return converted


def _group_error_metrics(group: pd.DataFrame) -> dict:
    errors = group["y_true"] - group["ensemble_mean"]
    return {
        "n": len(group),
        "rmse": float(np.sqrt(np.mean(errors.pow(2)))),
        "mae": float(errors.abs().mean()),
        "mean_uncertainty": float(group["ensemble_std"].mean()),
        "empirical_coverage": float(group["covered"].mean()),
    }


def descriptor_error_summary(predictions: pd.DataFrame) -> dict:
    """Summarize errors across descriptor quantiles without causal interpretation."""
    descriptor_names = ["molecular_weight", "heavy_atom_count", "ring_count"]
    required = {
        *descriptor_names,
        "y_true",
        "ensemble_mean",
        "ensemble_std",
        "covered",
    }
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for descriptor summaries")
    summary = {}
    for descriptor_name in descriptor_names:
        quantiles = pd.qcut(predictions[descriptor_name], q=4, duplicates="drop")
        groups = []
        grouped = predictions.groupby(quantiles, observed=True, sort=True)
        if quantiles.isna().all():
            grouped_items = [("all", predictions)]
        else:
            grouped_items = grouped
        for interval, group in grouped_items:
            groups.append(
                {
                    "group": str(interval),
                    "minimum": float(group[descriptor_name].min()),
                    "maximum": float(group[descriptor_name].max()),
                    **_group_error_metrics(group),
                }
            )
        summary[descriptor_name] = groups
    return summary


def uncertainty_bucket_summary(predictions: pd.DataFrame) -> list[dict]:
    """Summarize performance in low, medium, and high uncertainty thirds."""
    required = {"y_true", "ensemble_mean", "ensemble_std", "covered"}
    if not required <= set(predictions.columns):
        raise ValueError("Predictions are missing columns required for uncertainty buckets")
    if predictions.empty:
        raise ValueError("Predictions must not be empty")
    percentile_ranks = predictions["ensemble_std"].rank(method="first", pct=True)
    buckets = pd.cut(
        percentile_ranks,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    results = []
    for bucket_name, group in predictions.groupby(buckets, observed=True, sort=True):
        results.append({"bucket": str(bucket_name), **_group_error_metrics(group)})
    return results
