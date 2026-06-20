from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class DatasetRecord(BaseModel):
    smiles: str
    target: float | int | None
    dataset_name: str


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
    for _, row in dataframe.iterrows():
        target = row[target_col] if target_col is not None else None
        if pd.isna(target):
            target = None

        records.append(
            DatasetRecord(
                smiles=row[smiles_col],
                target=target,
                dataset_name=dataset_name,
            )
        )

    return records
