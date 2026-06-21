import importlib

import pandas as pd
import pytest
from PIL import Image

from molgnn_ops.dashboard import format_metric, nearest_neighbors_table, render_molecule


def test_dashboard_module_imports_without_launching_server() -> None:
    module = importlib.import_module("molgnn_ops.dashboard")

    assert callable(module.main)


def test_dashboard_rendering_and_format_helpers() -> None:
    image = render_molecule("CCO", size=(200, 120))

    assert isinstance(image, Image.Image)
    assert image.size == (200, 120)
    assert format_metric(1.23456, digits=3) == "1.235"
    assert format_metric(None) == "N/A"
    with pytest.raises(ValueError, match="Invalid SMILES"):
        render_molecule("invalid")


def test_nearest_neighbors_table_formats_selected_fields() -> None:
    table = nearest_neighbors_table(
        [
            {
                "canonical_smiles": "CCO",
                "measured_target": -0.7,
                "tanimoto_similarity": 1.0,
                "descriptors": {
                    "molecular_weight": 46.1,
                    "heavy_atom_count": 3,
                    "ring_count": 0,
                },
            }
        ]
    )

    assert isinstance(table, pd.DataFrame)
    assert table.loc[0, "canonical_smiles"] == "CCO"
    assert table.loc[0, "tanimoto_similarity"] == 1.0
