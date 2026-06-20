from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    raw_filename: str
    smiles_col: str
    target_col: str
    task_type: str
    default_split_strategy: str
    description: str


_DATASET_SPECS = {
    "esol": DatasetSpec(
        name="esol",
        url=(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/"
            "datasets/delaney-processed.csv"
        ),
        raw_filename="delaney-processed.csv",
        smiles_col="smiles",
        target_col="measured log solubility in mols per litre",
        task_type="regression",
        default_split_strategy="scaffold",
        description="ESOL/Delaney aqueous solubility regression benchmark.",
    )
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a registered dataset specification by case-insensitive name."""
    normalized_name = name.strip().lower()
    try:
        return _DATASET_SPECS[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(_DATASET_SPECS))
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        ) from error


def list_dataset_specs() -> list[DatasetSpec]:
    """Return all registered datasets in stable name order."""
    return [_DATASET_SPECS[name] for name in sorted(_DATASET_SPECS)]
