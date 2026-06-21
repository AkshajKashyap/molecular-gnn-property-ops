from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol

from molgnn_ops.datasets import ensure_prepared_identity


def _validate_fingerprint_config(radius: int, n_bits: int) -> None:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_bits <= 0:
        raise ValueError("n_bits must be greater than 0")


def _parse_smiles(smiles: str) -> tuple[str, Mol]:
    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    molecule = Chem.MolFromSmiles(normalized_smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return normalized_smiles, molecule


def _fingerprint_array(molecule: Mol, generator, n_bits: int) -> np.ndarray:
    fingerprint = generator.GetFingerprint(molecule)
    values = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[int]:
    """Return a binary Morgan fingerprint for a valid SMILES string."""
    _validate_fingerprint_config(radius, n_bits)
    _, molecule = _parse_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _fingerprint_array(molecule, generator, n_bits).astype(int).tolist()


def featurize_fingerprints_from_csv(
    input_csv: Path,
    output_npz: Path,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, int]:
    """Write valid, labeled prepared CSV rows as a compressed fingerprint dataset."""
    _validate_fingerprint_config(radius, n_bits)
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV dataset not found: {input_csv}")

    dataframe = ensure_prepared_identity(pd.read_csv(input_csv))
    required_columns = {"smiles", "target", "split", "dataset_name"}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Required columns missing from {input_csv}: {missing}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint_rows: list[np.ndarray] = []
    targets: list[float] = []
    splits: list[str] = []
    smiles_values: list[str] = []
    dataset_names: list[str] = []
    sample_ids: list[str] = []
    canonical_smiles_values: list[str] = []
    n_valid = 0
    n_invalid = 0

    for _, row in dataframe.iterrows():
        if pd.isna(row["smiles"]):
            n_invalid += 1
            continue

        try:
            normalized_smiles, molecule = _parse_smiles(str(row["smiles"]))
        except ValueError:
            n_invalid += 1
            continue

        n_valid += 1
        if pd.isna(row["target"]):
            continue

        fingerprint_rows.append(_fingerprint_array(molecule, generator, n_bits))
        targets.append(float(row["target"]))
        splits.append("" if pd.isna(row["split"]) else str(row["split"]))
        smiles_values.append(normalized_smiles)
        dataset_names.append(
            "" if pd.isna(row["dataset_name"]) else str(row["dataset_name"])
        )
        sample_ids.append(str(row["sample_id"]))
        canonical_smiles_values.append(str(row["canonical_smiles"]))

    if fingerprint_rows:
        features = np.stack(fingerprint_rows)
    else:
        features = np.empty((0, n_bits), dtype=np.uint8)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            X=features,
            y=np.asarray(targets, dtype=float),
            splits=np.asarray(splits, dtype=str),
            smiles=np.asarray(smiles_values, dtype=str),
            dataset_name=np.asarray(dataset_names, dtype=str),
            sample_id=np.asarray(sample_ids, dtype=str),
            canonical_smiles=np.asarray(canonical_smiles_values, dtype=str),
        )

    return {
        "n_rows": len(dataframe),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_missing_targets": int(dataframe["target"].isna().sum()),
        "n_written": len(fingerprint_rows),
    }
