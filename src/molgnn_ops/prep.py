from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from molgnn_ops.datasets import load_csv_dataset
from molgnn_ops.splits import random_split_indices, scaffold_split_indices


class PreparedDatasetSummary(BaseModel):
    dataset_name: str
    n_rows: int
    n_valid_smiles: int
    n_missing_targets: int
    n_train: int
    n_val: int
    n_test: int
    split_strategy: str
    split_seed: int
    output_path: Path


def prepare_dataset(
    input_csv: Path,
    output_csv: Path,
    smiles_col: str,
    target_col: str,
    dataset_name: str,
    split_strategy: str,
    seed: int | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_seed: int = 42,
) -> PreparedDatasetSummary:
    """Load, split, and persist a model-ready tabular molecular dataset."""
    if split_strategy not in {"random", "scaffold"}:
        raise ValueError("split_strategy must be either 'random' or 'scaffold'")

    records = load_csv_dataset(input_csv, smiles_col, target_col, dataset_name)
    resolved_split_seed = split_seed if seed is None else seed
    n_rows = len(pd.read_csv(input_csv))
    smiles = [record.smiles for record in records]

    if split_strategy == "random":
        split_indices = random_split_indices(
            len(records), train_frac, val_frac, test_frac, resolved_split_seed
        )
    else:
        split_indices = scaffold_split_indices(
            smiles, train_frac, val_frac, test_frac, resolved_split_seed
        )

    split_labels = [""] * len(records)
    for split_name, indices in split_indices.items():
        for index in indices:
            split_labels[index] = split_name

    prepared = pd.DataFrame(
        {
            "sample_id": [record.sample_id for record in records],
            "smiles": smiles,
            "canonical_smiles": [record.canonical_smiles for record in records],
            "target": [record.target for record in records],
            "dataset_name": [record.dataset_name for record in records],
            "split": split_labels,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv, index=False)

    return PreparedDatasetSummary(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_valid_smiles=len(records),
        n_missing_targets=sum(record.target is None for record in records),
        n_train=len(split_indices["train"]),
        n_val=len(split_indices["val"]),
        n_test=len(split_indices["test"]),
        split_strategy=split_strategy,
        split_seed=resolved_split_seed,
        output_path=output_csv,
    )
