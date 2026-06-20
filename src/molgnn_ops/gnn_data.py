from pathlib import Path

import torch
from torch_geometric.data import Data

from molgnn_ops.featurization import BOND_FEATURE_CONFIG, MoleculeGraph
from molgnn_ops.graph_io import load_graphs_jsonl


def molecule_graph_to_pyg_data(graph: MoleculeGraph) -> Data:
    """Convert an inspectable MoleculeGraph into a PyTorch Geometric Data object."""
    x = torch.tensor(graph.atom_features, dtype=torch.float32)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long).reshape(2, -1)
    edge_feature_dim = (
        len(graph.edge_features[0])
        if graph.edge_features
        else len(BOND_FEATURE_CONFIG.bond_types) + 2
    )
    if graph.edge_features:
        edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32)
    else:
        edge_attr = torch.empty((0, edge_feature_dim), dtype=torch.float32)
    target = float("nan") if graph.target is None else float(graph.target)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float32),
        smiles=graph.smiles,
        split=graph.split,
        dataset_name=graph.dataset_name,
    )


def load_pyg_dataset_from_jsonl(path: Path) -> list[Data]:
    """Load labeled molecular graphs from JSONL as PyG Data objects."""
    return [
        molecule_graph_to_pyg_data(graph)
        for graph in load_graphs_jsonl(path)
        if graph.target is not None
    ]


def split_pyg_dataset(data_list: list[Data]) -> dict[str, list[Data]]:
    """Partition PyG graphs by persistent train, validation, and test metadata."""
    splits = {"train": [], "val": [], "test": []}
    for data in data_list:
        split_name = getattr(data, "split", None)
        if split_name not in splits:
            raise ValueError(f"Graph has unsupported or missing split metadata: {split_name!r}")
        splits[split_name].append(data)

    missing_splits = [name for name, values in splits.items() if not values]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise ValueError(f"Graph dataset has no rows for split(s): {missing}")
    return splits
