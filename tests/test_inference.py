import math
from pathlib import Path

import pytest

from molgnn_ops.inference import (
    load_promoted_model,
    predict_smiles,
    predict_smiles_batch,
    predict_smiles_with_context,
)


def test_promoted_gcn_loads_and_predicts(promoted_manifest_path: Path) -> None:
    loaded = load_promoted_model(promoted_manifest_path)
    prediction = predict_smiles("OCC", loaded)

    assert loaded.model.training is False
    assert prediction["canonical_smiles"] == "CCO"
    assert math.isfinite(prediction["predicted_log_solubility"])
    assert prediction["predicted_solubility_mol_per_litre"] == pytest.approx(
        10 ** prediction["predicted_log_solubility"]
    )
    assert prediction["model_id"] == "synthetic-gcn-v1"
    assert "uncertainty" not in prediction
    assert all("uncertainty" not in key for key in prediction)
    assert "nearest_training_molecules" not in prediction


def test_prediction_with_context_returns_neighbors_without_uncertainty(
    promoted_manifest_path: Path,
) -> None:
    loaded = load_promoted_model(promoted_manifest_path)
    result = predict_smiles_with_context("CCO", loaded, top_k=2)

    assert set(result) == {
        "prediction",
        "molecular_descriptors",
        "applicability",
        "nearest_training_molecules",
    }
    assert result["applicability"]["maximum_similarity"] == pytest.approx(1.0)
    assert len(result["nearest_training_molecules"]) == 2
    assert "uncertainty" not in str(result).lower()


def test_standard_prediction_can_include_compact_applicability(
    promoted_manifest_path: Path,
) -> None:
    loaded = load_promoted_model(promoted_manifest_path)
    prediction = predict_smiles(
        "CCO",
        loaded,
        include_applicability=True,
        top_k=2,
    )

    assert prediction["applicability"]["maximum_similarity"] == pytest.approx(1.0)
    assert "nearest_training_molecules" not in prediction


def test_invalid_smiles_raises_clear_error(promoted_manifest_path: Path) -> None:
    loaded = load_promoted_model(promoted_manifest_path)

    with pytest.raises(ValueError, match="Invalid SMILES"):
        predict_smiles("not-a-smiles", loaded)
    with pytest.raises(ValueError, match="must not be blank"):
        predict_smiles("   ", loaded)


def test_batch_prediction_preserves_order_and_isolates_errors(
    promoted_manifest_path: Path,
) -> None:
    loaded = load_promoted_model(promoted_manifest_path)
    results = predict_smiles_batch(["CCO", "invalid", "CCN"], loaded)

    assert [result["input_smiles"] for result in results] == ["CCO", "invalid", "CCN"]
    assert [result["success"] for result in results] == [True, False, True]
    assert results[0]["prediction"] is not None
    assert results[1]["prediction"] is None
    assert "Invalid SMILES" in results[1]["error"]
