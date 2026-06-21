from pathlib import Path

import pandas as pd

from molgnn_ops.datasets import ensure_prepared_identity


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def audit_duplicate_molecules(prepared_csv: Path) -> dict:
    """Audit repeated canonical molecules without modifying their measurements."""
    if not prepared_csv.is_file():
        raise FileNotFoundError(f"Prepared CSV not found: {prepared_csv}")
    dataframe = ensure_prepared_identity(pd.read_csv(prepared_csv))
    required = {"sample_id", "canonical_smiles", "target", "split"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Prepared CSV is missing columns: {', '.join(missing)}")
    if dataframe["canonical_smiles"].isna().any():
        raise ValueError("Prepared CSV contains invalid or missing canonical_smiles values")

    grouped = dataframe.groupby("canonical_smiles", sort=True, dropna=False)
    duplicate_groups = [group for _, group in grouped if len(group) > 1]
    identical_target_groups = 0
    conflicting_target_groups = 0
    conflicting_details = []
    for group in duplicate_groups:
        unique_targets = group["target"].nunique(dropna=False)
        if unique_targets == 1:
            identical_target_groups += 1
            continue
        conflicting_target_groups += 1
        conflicting_details.append(
            {
                "canonical_smiles": str(group["canonical_smiles"].iloc[0]),
                "sample_ids": group["sample_id"].astype(str).tolist(),
                "targets": [_json_value(value) for value in group["target"]],
                "splits": group["split"].astype(str).tolist(),
            }
        )

    return {
        "total_rows": len(dataframe),
        "unique_sample_ids": int(dataframe["sample_id"].nunique()),
        "unique_canonical_smiles": int(dataframe["canonical_smiles"].nunique()),
        "duplicate_canonical_smiles_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(len(group) for group in duplicate_groups),
        "duplicate_groups_with_identical_targets": identical_target_groups,
        "duplicate_groups_with_conflicting_targets": conflicting_target_groups,
        "conflicting_groups": conflicting_details,
    }
