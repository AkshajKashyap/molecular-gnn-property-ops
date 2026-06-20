from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molgnn_ops.splits import scaffold_key_from_smiles


def _require_columns(dataframe: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing_columns = sorted(columns.difference(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Required columns missing from {path}: {missing}")


def target_summary(y) -> dict:
    """Summarize the finite values in a numeric target sequence."""
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot summarize an empty target array")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def split_target_summary(prepared_csv: Path) -> dict:
    """Summarize target distributions independently for every dataset split."""
    dataframe = pd.read_csv(prepared_csv)
    _require_columns(dataframe, {"split", "target"}, prepared_csv)
    summaries = {}
    for split_name, split_frame in dataframe.groupby("split", sort=True):
        summaries[str(split_name)] = target_summary(split_frame["target"])
    return summaries


def prediction_error_summary(predictions_csv: Path) -> dict:
    """Compute signed and absolute prediction-error diagnostics by split."""
    dataframe = pd.read_csv(predictions_csv)
    _require_columns(dataframe, {"split", "y_true", "y_pred"}, predictions_csv)
    dataframe = dataframe.dropna(subset=["split", "y_true", "y_pred"])

    summaries = {}
    for split_name, split_frame in dataframe.groupby("split", sort=True):
        errors = split_frame["y_pred"].to_numpy(float) - split_frame["y_true"].to_numpy(float)
        absolute_errors = np.abs(errors)
        summaries[str(split_name)] = {
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mean_error": float(np.mean(errors)),
            "median_abs_error": float(np.median(absolute_errors)),
            "max_abs_error": float(np.max(absolute_errors)),
            "n": int(len(errors)),
        }
    return summaries


def worst_predictions(
    predictions_csv: Path,
    split: str = "test",
    n: int = 20,
) -> list[dict]:
    """Return the largest absolute prediction errors for one split."""
    if n < 0:
        raise ValueError("n must be non-negative")
    dataframe = pd.read_csv(predictions_csv)
    _require_columns(dataframe, {"split", "y_true", "y_pred"}, predictions_csv)
    selected = dataframe[dataframe["split"] == split].dropna(subset=["y_true", "y_pred"]).copy()
    selected["error"] = selected["y_pred"] - selected["y_true"]
    selected["absolute_error"] = selected["error"].abs()
    selected = selected.sort_values("absolute_error", ascending=False, kind="stable").head(n)
    return selected.to_dict(orient="records")


def scaffold_distribution_summary(prepared_csv: Path) -> dict:
    """Summarize molecule counts across Bemis-Murcko scaffold groups."""
    dataframe = pd.read_csv(prepared_csv)
    _require_columns(dataframe, {"smiles"}, prepared_csv)

    scaffold_counts: Counter[str] = Counter()
    for smiles in dataframe["smiles"].dropna():
        normalized_smiles = str(smiles).strip()
        if normalized_smiles:
            scaffold_counts[scaffold_key_from_smiles(normalized_smiles)] += 1
    if not scaffold_counts:
        raise ValueError("Prepared dataset contains no nonblank SMILES")

    group_sizes = np.asarray(list(scaffold_counts.values()), dtype=int)
    top_groups = sorted(scaffold_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "n_unique_scaffolds": int(len(scaffold_counts)),
        "largest_scaffold_group_size": int(np.max(group_sizes)),
        "median_scaffold_group_size": float(np.median(group_sizes)),
        "top_10_scaffold_groups": [
            {"scaffold": scaffold, "size": int(size)} for scaffold, size in top_groups
        ],
        "n_singleton_scaffolds": int(np.sum(group_sizes == 1)),
    }


def compute_test_max_similarities(
    prepared_csv: Path,
    n_bits: int = 2048,
    radius: int = 2,
) -> list[float]:
    """Return each test molecule's maximum Morgan similarity to a train molecule."""
    if n_bits <= 0:
        raise ValueError("n_bits must be greater than 0")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    dataframe = pd.read_csv(prepared_csv)
    _require_columns(dataframe, {"smiles", "split"}, prepared_csv)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints: dict[str, list] = {"train": [], "test": []}
    for row in dataframe.itertuples(index=False):
        split_name = str(row.split)
        if split_name not in fingerprints:
            continue
        smiles = str(row.smiles).strip()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            fingerprints[split_name].append(generator.GetFingerprint(molecule))

    if not fingerprints["train"] or not fingerprints["test"]:
        raise ValueError("Prepared dataset must contain valid train and test molecules")
    return [
        float(max(DataStructs.BulkTanimotoSimilarity(test_fp, fingerprints["train"])))
        for test_fp in fingerprints["test"]
    ]


def train_test_similarity_summary(
    prepared_csv: Path,
    n_bits: int = 2048,
    radius: int = 2,
) -> dict:
    """Summarize nearest-train Morgan similarity for every test molecule."""
    similarities = np.asarray(
        compute_test_max_similarities(prepared_csv, n_bits=n_bits, radius=radius),
        dtype=float,
    )
    return {
        "n_test": int(similarities.size),
        "mean_max_similarity": float(np.mean(similarities)),
        "median_max_similarity": float(np.median(similarities)),
        "min_max_similarity": float(np.min(similarities)),
        "q25": float(np.quantile(similarities, 0.25)),
        "q75": float(np.quantile(similarities, 0.75)),
        "n_below_0_3": int(np.sum(similarities < 0.3)),
        "n_below_0_5": int(np.sum(similarities < 0.5)),
        "n_below_0_7": int(np.sum(similarities < 0.7)),
    }
