from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.featurization import (
    canonicalize_smiles,
    featurize_records_from_csv,
    featurize_smiles,
)
from molgnn_ops.graph_io import load_graphs_jsonl, save_graphs_jsonl


def test_canonicalize_smiles() -> None:
    assert canonicalize_smiles("OCC") == "CCO"


@pytest.mark.parametrize("smiles", ["not-a-smiles", "   "])
def test_canonicalize_smiles_raises_for_invalid_smiles(smiles: str) -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        canonicalize_smiles(smiles)


def test_featurize_smiles_creates_directed_graph() -> None:
    graph = featurize_smiles("CCO")

    assert len(graph.atom_features) == 3
    assert all(len(features) == 22 for features in graph.atom_features)
    assert len(graph.edge_index) == 2
    assert len(graph.edge_index[0]) == 4
    assert len(graph.edge_index[1]) == 4
    assert len(graph.edge_features) == 4
    assert all(len(features) == 7 for features in graph.edge_features)


def test_featurize_smiles_handles_single_atom() -> None:
    graph = featurize_smiles("Cl")

    assert len(graph.atom_features) == 1
    assert graph.edge_index == [[], []]
    assert graph.edge_features == []


def test_featurize_smiles_preserves_metadata() -> None:
    graph = featurize_smiles(
        "CCO",
        target=1.5,
        split="train",
        dataset_name="example",
    )

    assert graph.target == 1.5
    assert graph.split == "train"
    assert graph.dataset_name == "example"


def test_graph_io_round_trip(tmp_path: Path) -> None:
    graphs = [featurize_smiles("CCO", target=1.0), featurize_smiles("Cl")]
    output_path = tmp_path / "nested" / "graphs.jsonl"

    save_graphs_jsonl(graphs, output_path)
    loaded = load_graphs_jsonl(output_path)

    assert loaded == graphs


def test_featurize_records_from_csv_skips_invalid_smiles(tmp_path: Path) -> None:
    input_csv = tmp_path / "prepared.csv"
    output_jsonl = tmp_path / "graphs.jsonl"
    pd.DataFrame(
        {
            "smiles": ["CCO", "not-a-smiles", "Cl", None],
            "target": [1.0, 2.0, None, 4.0],
            "split": ["train", "val", "test", "train"],
            "dataset_name": ["example"] * 4,
        }
    ).to_csv(input_csv, index=False)

    summary = featurize_records_from_csv(input_csv, output_jsonl)
    graphs = load_graphs_jsonl(output_jsonl)

    assert summary == {
        "n_rows": 4,
        "n_valid": 2,
        "n_invalid": 2,
        "n_graphs_written": 2,
    }
    assert [graph.canonical_smiles for graph in graphs] == ["CCO", "Cl"]
    assert graphs[0].target == 1.0
    assert graphs[0].split == "train"
    assert graphs[0].dataset_name == "example"
