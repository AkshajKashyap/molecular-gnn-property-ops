import pandas as pd
import pytest

from molgnn_ops.gnn_error_analysis import (
    attach_molecular_descriptors,
    descriptor_error_summary,
    molecular_descriptors,
    uncertainty_bucket_summary,
    worst_ensemble_predictions,
)


def _analyzed_predictions() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCCl", "C1CCCCC1", "CC(=O)O", "CCCC"],
            "y_true": [0.0, 1.0, 2.0, 3.0, 1.5, 0.5],
            "ensemble_mean": [0.1, 1.3, 1.0, 2.8, 1.7, 0.9],
            "ensemble_std": [0.1, 0.2, 0.9, 0.3, 0.5, 0.7],
            "absolute_error": [0.1, 0.3, 1.0, 0.2, 0.2, 0.4],
            "interval_lower": [-0.2, 0.9, 0.5, 2.2, 1.0, 0.0],
            "interval_upper": [0.4, 1.7, 1.5, 3.4, 2.4, 1.8],
            "covered": [True, True, False, True, True, True],
        }
    )
    return attach_molecular_descriptors(dataframe)


def test_molecular_descriptors_and_invalid_smiles() -> None:
    descriptors = molecular_descriptors("CCO")

    assert descriptors["heavy_atom_count"] == 3
    assert descriptors["molecular_weight"] > 40
    assert descriptors["ring_count"] == 0
    with pytest.raises(ValueError, match="Invalid SMILES"):
        molecular_descriptors("not-a-smiles")


def test_worst_predictions_and_group_summaries() -> None:
    predictions = _analyzed_predictions()
    worst = worst_ensemble_predictions(predictions, n=2)
    descriptor_summary = descriptor_error_summary(predictions)
    buckets = uncertainty_bucket_summary(predictions)

    assert [row["smiles"] for row in worst] == ["CCCl", "CCCC"]
    assert set(descriptor_summary) == {
        "molecular_weight",
        "heavy_atom_count",
        "ring_count",
    }
    assert all(descriptor_summary.values())
    assert [row["bucket"] for row in buckets] == ["low", "medium", "high"]
    assert sum(row["n"] for row in buckets) == len(predictions)
