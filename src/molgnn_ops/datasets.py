from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from rdkit import Chem


class DatasetRecord(BaseModel):
    sample_id: str
    smiles: str
    canonical_smiles: str
    target: float | int | None
    dataset_name: str


def _canonical_smiles(smiles: str, source_row: object) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES at source row {source_row}: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def ensure_prepared_identity(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic legacy IDs/canonical SMILES and validate sample identity."""
    if "smiles" not in dataframe.columns:
        raise ValueError("Prepared dataset must contain a smiles column")
    result = dataframe.copy()
    if "sample_id" not in result.columns:
        if "dataset_name" not in result.columns:
            raise ValueError(
                "Prepared dataset has no sample_id or dataset_name; regenerate it"
            )
        result["sample_id"] = [
            f"{dataset_name}:{row_index}"
            for row_index, dataset_name in zip(
                result.index,
                result["dataset_name"],
                strict=True,
            )
        ]
    if result["sample_id"].isna().any() or (result["sample_id"].astype(str) == "").any():
        raise ValueError("Prepared dataset contains missing sample_id values")
    duplicated_ids = result["sample_id"].duplicated(keep=False)
    if duplicated_ids.any():
        duplicates = sorted(result.loc[duplicated_ids, "sample_id"].astype(str).unique())
        raise ValueError(f"Prepared dataset contains duplicate sample_id values: {duplicates}")

    if "canonical_smiles" not in result.columns:
        canonical_values: list[str | None] = []
        for _, smiles in result["smiles"].items():
            if pd.isna(smiles):
                canonical_values.append(None)
                continue
            molecule = Chem.MolFromSmiles(str(smiles).strip())
            canonical_values.append(
                Chem.MolToSmiles(molecule, canonical=True)
                if molecule is not None
                else None
            )
        result["canonical_smiles"] = canonical_values
    return result


def load_csv_dataset(
    path: str | Path,
    smiles_col: str,
    target_col: str | None,
    dataset_name: str,
) -> list[DatasetRecord]:
    """Load and validate molecular records from a CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if smiles_col not in dataframe.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {csv_path}")
    if target_col is not None and target_col not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    dataframe = dataframe[dataframe[smiles_col].notna()].copy()
    dataframe[smiles_col] = dataframe[smiles_col].astype(str).str.strip()
    dataframe = dataframe[dataframe[smiles_col] != ""]

    records: list[DatasetRecord] = []
    for source_index, row in dataframe.iterrows():
        target = row[target_col] if target_col is not None else None
        if pd.isna(target):
            target = None

        records.append(
            DatasetRecord(
                sample_id=f"{dataset_name}:{source_index}",
                smiles=row[smiles_col],
                canonical_smiles=_canonical_smiles(row[smiles_col], source_index),
                target=target,
                dataset_name=dataset_name,
            )
        )

    return records
