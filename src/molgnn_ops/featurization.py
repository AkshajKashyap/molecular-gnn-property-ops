from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, BondType, Mol


@dataclass(frozen=True)
class AtomFeatureConfig:
    symbols: tuple[str, ...] = (
        "C",
        "N",
        "O",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
        "H",
        "Other",
    )
    hybridizations: tuple[str, ...] = (
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "Other",
    )


@dataclass(frozen=True)
class BondFeatureConfig:
    bond_types: tuple[str, ...] = (
        "single",
        "double",
        "triple",
        "aromatic",
        "other",
    )


class MoleculeGraph(BaseModel):
    smiles: str
    canonical_smiles: str
    atom_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    target: float | int | None = None
    split: str | None = None
    dataset_name: str | None = None


ATOM_FEATURE_CONFIG = AtomFeatureConfig()
BOND_FEATURE_CONFIG = BondFeatureConfig()


def _parse_smiles(smiles: str) -> tuple[str, Mol]:
    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    molecule = Chem.MolFromSmiles(normalized_smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return normalized_smiles, molecule


def canonicalize_smiles(smiles: str) -> str:
    """Return the canonical RDKit representation of a valid SMILES string."""
    _, molecule = _parse_smiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True)


def atom_features(atom: Atom) -> list[float]:
    """Create a deterministic numeric feature vector for one atom."""
    symbol = atom.GetSymbol()
    if symbol not in ATOM_FEATURE_CONFIG.symbols[:-1]:
        symbol = "Other"
    symbol_features = [
        float(symbol == candidate) for candidate in ATOM_FEATURE_CONFIG.symbols
    ]

    hybridization = str(atom.GetHybridization())
    if hybridization not in ATOM_FEATURE_CONFIG.hybridizations[:-1]:
        hybridization = "Other"
    hybridization_features = [
        float(hybridization == candidate)
        for candidate in ATOM_FEATURE_CONFIG.hybridizations
    ]

    return [
        *symbol_features,
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        *hybridization_features,
        float(atom.GetTotalNumHs()),
        float(atom.IsInRing()),
    ]


def bond_features(bond: Bond) -> list[float]:
    """Create a deterministic numeric feature vector for one bond."""
    bond_type_lookup = {
        BondType.SINGLE: "single",
        BondType.DOUBLE: "double",
        BondType.TRIPLE: "triple",
        BondType.AROMATIC: "aromatic",
    }
    bond_type = bond_type_lookup.get(bond.GetBondType(), "other")
    type_features = [
        float(bond_type == candidate) for candidate in BOND_FEATURE_CONFIG.bond_types
    ]
    return [
        *type_features,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def _featurize_molecule(
    smiles: str,
    molecule: Mol,
    target: float | int | None = None,
    split: str | None = None,
    dataset_name: str | None = None,
) -> MoleculeGraph:
    node_features = [atom_features(atom) for atom in molecule.GetAtoms()]
    source_indices: list[int] = []
    destination_indices: list[int] = []
    directed_edge_features: list[list[float]] = []

    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        source_indices.extend((begin, end))
        destination_indices.extend((end, begin))
        directed_edge_features.extend((features, features.copy()))

    return MoleculeGraph(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(molecule, canonical=True),
        atom_features=node_features,
        edge_index=[source_indices, destination_indices],
        edge_features=directed_edge_features,
        target=target,
        split=split,
        dataset_name=dataset_name,
    )


def featurize_smiles(
    smiles: str,
    target: float | int | None = None,
    split: str | None = None,
    dataset_name: str | None = None,
) -> MoleculeGraph:
    """Convert a valid SMILES string into an inspectable molecular graph."""
    normalized_smiles, molecule = _parse_smiles(smiles)
    return _featurize_molecule(
        normalized_smiles,
        molecule,
        target=target,
        split=split,
        dataset_name=dataset_name,
    )


def _optional_value(row: pd.Series, column: str) -> object | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return row[column]


def featurize_records_from_csv(
    input_csv: Path,
    output_jsonl: Path,
) -> dict[str, int]:
    """Featurize CSV rows, skipping invalid SMILES and writing valid graphs as JSONL."""
    if not input_csv.is_file():
        raise FileNotFoundError(f"CSV dataset not found: {input_csv}")

    dataframe = pd.read_csv(input_csv)
    if "smiles" not in dataframe.columns:
        raise ValueError(f"SMILES column 'smiles' not found in {input_csv}")

    graphs: list[MoleculeGraph] = []
    n_invalid = 0
    for _, row in dataframe.iterrows():
        smiles_value = _optional_value(row, "smiles")
        if smiles_value is None:
            n_invalid += 1
            continue

        try:
            normalized_smiles, molecule = _parse_smiles(str(smiles_value))
        except ValueError:
            n_invalid += 1
            continue

        split_value = _optional_value(row, "split")
        dataset_value = _optional_value(row, "dataset_name")
        graphs.append(
            _featurize_molecule(
                normalized_smiles,
                molecule,
                target=_optional_value(row, "target"),
                split=str(split_value) if split_value is not None else None,
                dataset_name=str(dataset_value) if dataset_value is not None else None,
            )
        )

    from molgnn_ops.graph_io import save_graphs_jsonl

    save_graphs_jsonl(graphs, output_jsonl)
    return {
        "n_rows": len(dataframe),
        "n_valid": len(graphs),
        "n_invalid": n_invalid,
        "n_graphs_written": len(graphs),
    }
