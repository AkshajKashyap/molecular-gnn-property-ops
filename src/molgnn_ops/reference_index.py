from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from molgnn_ops.datasets import ensure_prepared_identity
from molgnn_ops.featurization import canonicalize_smiles
from molgnn_ops.fingerprints import morgan_fingerprint
from molgnn_ops.gnn_error_analysis import molecular_descriptors

DESCRIPTOR_NAMES = (
    "molecular_weight",
    "heavy_atom_count",
    "ring_count",
    "rotatable_bond_count",
    "heteroatom_count",
)


class ReferenceMolecule(BaseModel):
    sample_id: str
    smiles: str
    canonical_smiles: str
    target: float
    split: str
    molecular_weight: float
    heavy_atom_count: int
    ring_count: int
    rotatable_bond_count: int
    heteroatom_count: int


@dataclass(frozen=True)
class ReferenceIndex:
    sample_ids: np.ndarray
    smiles: np.ndarray
    canonical_smiles: np.ndarray
    targets: np.ndarray
    splits: np.ndarray
    fingerprints: np.ndarray
    descriptors: dict[str, np.ndarray]
    radius: int
    n_bits: int

    def __len__(self) -> int:
        return len(self.sample_ids)


def build_reference_index(
    prepared_csv: Path,
    output_path: Path,
    split: str = "train",
    radius: int = 2,
    n_bits: int = 2048,
) -> dict:
    """Build a lossless fingerprint reference index for one prepared split."""
    if not prepared_csv.is_file():
        raise FileNotFoundError(f"Prepared CSV not found: {prepared_csv}")
    if radius < 0 or n_bits <= 0:
        raise ValueError("radius must be non-negative and n_bits must be greater than 0")
    dataframe = ensure_prepared_identity(pd.read_csv(prepared_csv))
    required = {"sample_id", "smiles", "canonical_smiles", "target", "split"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Prepared CSV is missing columns: {', '.join(missing)}")
    selected = dataframe[dataframe["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Prepared CSV contains no rows for split {split!r}")
    if selected["target"].isna().any():
        raise ValueError(f"Reference split {split!r} contains missing targets")

    records = []
    fingerprint_rows = []
    for row in selected.itertuples(index=False):
        canonical = canonicalize_smiles(str(row.smiles))
        descriptors = molecular_descriptors(canonical)
        records.append(
            ReferenceMolecule(
                sample_id=str(row.sample_id),
                smiles=str(row.smiles),
                canonical_smiles=canonical,
                target=float(row.target),
                split=str(row.split),
                **descriptors,
            )
        )
        fingerprint_rows.append(morgan_fingerprint(canonical, radius=radius, n_bits=n_bits))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sample_id": np.asarray([record.sample_id for record in records], dtype=str),
        "smiles": np.asarray([record.smiles for record in records], dtype=str),
        "canonical_smiles": np.asarray(
            [record.canonical_smiles for record in records],
            dtype=str,
        ),
        "target": np.asarray([record.target for record in records], dtype=float),
        "split": np.asarray([record.split for record in records], dtype=str),
        "fingerprints": np.asarray(fingerprint_rows, dtype=np.uint8),
        "radius": np.asarray(radius, dtype=np.int64),
        "n_bits": np.asarray(n_bits, dtype=np.int64),
    }
    for descriptor_name in DESCRIPTOR_NAMES:
        arrays[descriptor_name] = np.asarray(
            [getattr(record, descriptor_name) for record in records]
        )
    with output_path.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)

    duplicate_groups = (
        pd.Series(arrays["canonical_smiles"]).value_counts().gt(1).sum()
    )
    return {
        "prepared_csv": str(prepared_csv),
        "output_path": str(output_path),
        "split": split,
        "n_reference_molecules": len(records),
        "n_unique_sample_ids": len(set(arrays["sample_id"].tolist())),
        "n_unique_canonical_smiles": len(set(arrays["canonical_smiles"].tolist())),
        "duplicate_canonical_smiles_groups": int(duplicate_groups),
        "radius": radius,
        "n_bits": n_bits,
    }


def load_reference_index(path: Path) -> ReferenceIndex:
    """Load and validate a compressed molecular reference index."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference index not found: {path}")
    required = {
        "sample_id",
        "smiles",
        "canonical_smiles",
        "target",
        "split",
        "fingerprints",
        "radius",
        "n_bits",
        *DESCRIPTOR_NAMES,
    }
    with np.load(path, allow_pickle=False) as stored:
        missing = sorted(required.difference(stored.files))
        if missing:
            raise ValueError(f"Reference index is missing arrays: {', '.join(missing)}")
        arrays = {name: stored[name].copy() for name in required}
    row_count = len(arrays["sample_id"])
    if row_count == 0:
        raise ValueError("Reference index must contain at least one molecule")
    if len(set(arrays["sample_id"].tolist())) != row_count:
        raise ValueError("Reference index sample IDs must be unique")
    row_arrays = {
        "smiles",
        "canonical_smiles",
        "target",
        "split",
        *DESCRIPTOR_NAMES,
    }
    if any(len(arrays[name]) != row_count for name in row_arrays):
        raise ValueError("Reference index arrays must have matching row counts")
    n_bits = int(arrays["n_bits"].item())
    if arrays["fingerprints"].shape != (row_count, n_bits):
        raise ValueError("Reference fingerprint dimensions do not match n_bits")
    return ReferenceIndex(
        sample_ids=arrays["sample_id"],
        smiles=arrays["smiles"],
        canonical_smiles=arrays["canonical_smiles"],
        targets=arrays["target"],
        splits=arrays["split"],
        fingerprints=arrays["fingerprints"],
        descriptors={name: arrays[name] for name in DESCRIPTOR_NAMES},
        radius=int(arrays["radius"].item()),
        n_bits=n_bits,
    )


def find_similar_molecules(
    smiles: str,
    reference_index: ReferenceIndex,
    top_k: int = 5,
) -> list[dict]:
    """Rank reference samples by Morgan-fingerprint Tanimoto similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    canonical = canonicalize_smiles(smiles)
    query = np.asarray(
        morgan_fingerprint(
            canonical,
            radius=reference_index.radius,
            n_bits=reference_index.n_bits,
        ),
        dtype=np.uint8,
    )
    reference_fingerprints = reference_index.fingerprints.astype(np.int32, copy=False)
    query_values = query.astype(np.int32, copy=False)
    intersections = reference_fingerprints @ query_values
    unions = reference_fingerprints.sum(axis=1) + query_values.sum() - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions != 0,
    )
    ranked_indices = np.argsort(-similarities, kind="stable")[:top_k]
    results = []
    for index in ranked_indices:
        results.append(
            {
                "sample_id": str(reference_index.sample_ids[index]),
                "smiles": str(reference_index.smiles[index]),
                "canonical_smiles": str(reference_index.canonical_smiles[index]),
                "measured_target": float(reference_index.targets[index]),
                "tanimoto_similarity": float(similarities[index]),
                "descriptors": {
                    name: float(reference_index.descriptors[name][index])
                    for name in DESCRIPTOR_NAMES
                },
            }
        )
    return results
