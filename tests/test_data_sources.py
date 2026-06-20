import pytest

from molgnn_ops.data_sources import get_dataset_spec, list_dataset_specs


def test_get_esol_dataset_spec() -> None:
    spec = get_dataset_spec("esol")

    assert spec.smiles_col == "smiles"
    assert spec.target_col == "measured log solubility in mols per litre"
    assert spec.task_type == "regression"
    assert spec.default_split_strategy == "scaffold"


def test_unknown_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_spec("unknown")


def test_list_dataset_specs_includes_esol() -> None:
    assert "esol" in {spec.name for spec in list_dataset_specs()}
