from pathlib import Path

import torch

from molgnn_ops.featurization import featurize_smiles
from molgnn_ops.gnn_data import (
    load_pyg_dataset_from_jsonl,
    molecule_graph_to_pyg_data,
    split_pyg_dataset,
)
from molgnn_ops.graph_io import save_graphs_jsonl


def test_molecule_graph_to_pyg_data_converts_cco() -> None:
    data = molecule_graph_to_pyg_data(
        featurize_smiles("CCO", target=1.5, split="train", dataset_name="example")
    )

    assert data.x.shape == (3, 22)
    assert data.x.dtype == torch.float32
    assert data.edge_index.shape == (2, 4)
    assert data.edge_index.dtype == torch.long
    assert data.edge_attr.shape == (4, 7)
    assert data.y.shape == (1,)
    assert data.smiles == "CCO"
    assert data.split == "train"


def test_sample_identity_survives_graph_to_pyg_conversion() -> None:
    data = molecule_graph_to_pyg_data(
        featurize_smiles(
            "OCC",
            target=1.5,
            split="train",
            dataset_name="example",
            sample_id="example:3",
        )
    )

    assert data.sample_id == "example:3"
    assert data.canonical_smiles == "CCO"


def test_load_pyg_dataset_skips_missing_targets(tmp_path: Path) -> None:
    graph_path = tmp_path / "graphs.jsonl"
    save_graphs_jsonl(
        [
            featurize_smiles("CCO", target=1.0, split="train"),
            featurize_smiles("CCN", target=None, split="val"),
        ],
        graph_path,
    )

    data_list = load_pyg_dataset_from_jsonl(graph_path)

    assert len(data_list) == 1
    assert data_list[0].smiles == "CCO"


def test_split_pyg_dataset_returns_required_splits() -> None:
    data_list = [
        molecule_graph_to_pyg_data(featurize_smiles("CCO", target=1.0, split="train")),
        molecule_graph_to_pyg_data(featurize_smiles("CCN", target=2.0, split="val")),
        molecule_graph_to_pyg_data(featurize_smiles("CCC", target=3.0, split="test")),
    ]

    splits = split_pyg_dataset(data_list)

    assert {name: len(values) for name, values in splits.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
