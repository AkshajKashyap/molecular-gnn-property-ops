from pathlib import Path

import pandas as pd

from molgnn_ops.applicability import molecule_context
from molgnn_ops.reference_index import build_reference_index, load_reference_index


def _reference(tmp_path: Path):
    prepared_csv = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "sample_id": ["data:0", "data:1", "data:2"],
            "smiles": ["CCO", "CCN", "CCC"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "target": [-0.7, -0.5, -1.0],
            "dataset_name": ["data"] * 3,
            "split": ["train"] * 3,
        }
    ).to_csv(prepared_csv, index=False)
    index_path = tmp_path / "reference.npz"
    build_reference_index(prepared_csv, index_path, n_bits=64)
    return load_reference_index(index_path)


def test_applicability_similarity_and_cautious_language(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    context = molecule_context("CCO", reference, top_k=2)

    assert context["maximum_similarity"] == 1.0
    assert 0 <= context["mean_top_k_similarity"] <= 1
    assert len(context["nearest_neighbors"]) == 2
    assert all("confidence" not in warning.lower() for warning in context["warnings"])
    assert any("does not guarantee" in warning for warning in context["warnings"])


def test_low_similarity_and_descriptor_range_warnings(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    context = molecule_context("[U]", reference)

    assert context["maximum_similarity"] < 0.3
    assert any("structurally dissimilar" in warning for warning in context["warnings"])
    assert any("molecular weight" in warning for warning in context["warnings"])


def test_missing_reference_index_is_handled_cleanly() -> None:
    context = molecule_context("CCO", None)

    assert context["nearest_neighbors"] == []
    assert context["maximum_similarity"] is None
    assert "No training reference index" in context["warnings"][0]
