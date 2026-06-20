import pytest

from molgnn_ops import splits as split_module
from molgnn_ops.splits import (
    random_split_indices,
    scaffold_key_from_smiles,
    scaffold_split_indices,
)


def test_random_split_is_deterministic() -> None:
    first = random_split_indices(20, 0.7, 0.15, 0.15, seed=17)
    second = random_split_indices(20, 0.7, 0.15, 0.15, seed=17)

    assert first == second


def test_random_split_has_no_overlap() -> None:
    splits = random_split_indices(20, 0.6, 0.2, 0.2, seed=42)
    split_sets = {name: set(indices) for name, indices in splits.items()}

    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set().union(*split_sets.values()) == set(range(20))


def test_invalid_split_fractions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        random_split_indices(10, 0.7, 0.2, 0.2, seed=42)


def test_scaffold_split_keeps_scaffold_groups_together() -> None:
    smiles = [
        "c1ccccc1O",
        "c1ccccc1N",
        "C1CCCCC1O",
        "C1CCCCC1N",
        "c1ccncc1O",
        "c1ccncc1N",
    ]
    splits = scaffold_split_indices(smiles, 0.5, 0.25, 0.25, seed=7)

    scaffold_locations: dict[str, set[str]] = {}
    for split_name, indices in splits.items():
        for index in indices:
            key = scaffold_key_from_smiles(smiles[index])
            scaffold_locations.setdefault(key, set()).add(split_name)

    assert scaffold_key_from_smiles(smiles[0]) == scaffold_key_from_smiles(smiles[1])
    assert all(len(locations) == 1 for locations in scaffold_locations.values())


def test_scaffold_key_has_explicit_non_chemical_fallback(monkeypatch) -> None:
    monkeypatch.setattr(split_module, "Chem", None)
    monkeypatch.setattr(split_module, "MurckoScaffold", None)

    hydroxy_key = scaffold_key_from_smiles("c1ccccc1O")
    amino_key = scaffold_key_from_smiles("c1ccccc1N")

    assert hydroxy_key.startswith("fallback_shape:")
    assert hydroxy_key == amino_key
