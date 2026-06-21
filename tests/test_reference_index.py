from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.reference_index import (
    build_reference_index,
    find_similar_molecules,
    load_reference_index,
)


def _write_prepared(path: Path) -> None:
    pd.DataFrame(
        {
            "sample_id": ["data:0", "data:1", "data:2", "data:3", "data:4"],
            "smiles": ["CCO", "OCC", "CCN", "CCC", "c1ccccc1"],
            "canonical_smiles": ["CCO", "CCO", "CCN", "CCC", "c1ccccc1"],
            "target": [-0.7, -0.8, -0.5, -1.0, -2.0],
            "dataset_name": ["data"] * 5,
            "split": ["train", "train", "train", "train", "test"],
        }
    ).to_csv(path, index=False)


def test_reference_index_build_load_and_similarity(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    index_path = tmp_path / "reference.npz"
    _write_prepared(prepared_csv)

    summary = build_reference_index(prepared_csv, index_path, n_bits=64)
    reference = load_reference_index(index_path)
    neighbors = find_similar_molecules("CCO", reference, top_k=4)

    assert summary["n_reference_molecules"] == 4
    assert summary["n_unique_sample_ids"] == 4
    assert summary["duplicate_canonical_smiles_groups"] == 1
    assert len(reference) == 4
    assert reference.fingerprints.shape == (4, 64)
    assert reference.n_bits == 64
    assert set(reference.splits.tolist()) == {"train"}
    assert [row["sample_id"] for row in neighbors[:2]] == ["data:0", "data:1"]
    assert neighbors[0]["tanimoto_similarity"] == pytest.approx(1.0)
    assert neighbors[1]["tanimoto_similarity"] == pytest.approx(1.0)
    assert [row["tanimoto_similarity"] for row in neighbors] == sorted(
        [row["tanimoto_similarity"] for row in neighbors],
        reverse=True,
    )


def test_similarity_search_rejects_invalid_inputs(tmp_path: Path) -> None:
    prepared_csv = tmp_path / "prepared.csv"
    index_path = tmp_path / "reference.npz"
    _write_prepared(prepared_csv)
    build_reference_index(prepared_csv, index_path, n_bits=64)
    reference = load_reference_index(index_path)

    with pytest.raises(ValueError, match="Invalid SMILES"):
        find_similar_molecules("invalid", reference)
    with pytest.raises(ValueError, match="top_k"):
        find_similar_molecules("CCO", reference, top_k=0)
