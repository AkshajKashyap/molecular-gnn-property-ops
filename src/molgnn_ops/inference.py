import math
from dataclasses import dataclass
from pathlib import Path

import torch

from molgnn_ops.featurization import (
    ATOM_FEATURE_CONFIG,
    BOND_FEATURE_CONFIG,
    featurize_smiles,
)
from molgnn_ops.gnn_data import molecule_graph_to_pyg_data
from molgnn_ops.gnn_error_analysis import molecular_descriptors
from molgnn_ops.gnn_models import GCNRegressor, GINRegressor
from molgnn_ops.model_registry import ModelManifest
from molgnn_ops.reference_index import ReferenceIndex, load_reference_index


@dataclass
class LoadedPromotedModel:
    manifest: ModelManifest
    model: torch.nn.Module
    device: torch.device
    target_mean: float
    target_std: float
    reference_index: ReferenceIndex | None = None


def _current_feature_dimensions() -> tuple[int, int]:
    atom_dimension = (
        len(ATOM_FEATURE_CONFIG.symbols)
        + 3
        + len(ATOM_FEATURE_CONFIG.hybridizations)
        + 2
    )
    edge_dimension = len(BOND_FEATURE_CONFIG.bond_types) + 2
    return atom_dimension, edge_dimension


def load_promoted_model(
    manifest_path: Path,
    device: str | None = None,
) -> LoadedPromotedModel:
    """Load and validate a self-contained promoted GNN package."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = manifest_path.parent / manifest.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Promoted checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(device or "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    atom_dimension, edge_dimension = _current_feature_dimensions()
    if manifest.atom_feature_dim != atom_dimension:
        raise ValueError(
            f"Manifest atom feature dimension {manifest.atom_feature_dim} does not match "
            f"runtime dimension {atom_dimension}"
        )
    if manifest.edge_feature_dim != edge_dimension:
        raise ValueError(
            f"Manifest edge feature dimension {manifest.edge_feature_dim} does not match "
            f"runtime dimension {edge_dimension}"
        )
    if int(checkpoint.get("input_dim", -1)) != manifest.atom_feature_dim:
        raise ValueError("Checkpoint input dimension does not match the model manifest")
    if checkpoint.get("model_name") != manifest.model_type:
        raise ValueError("Checkpoint model type does not match the model manifest")
    checkpoint_config = checkpoint.get("hyperparameters", {})
    expected_config = {
        "hidden_dim": manifest.hidden_dim,
        "num_layers": manifest.num_layers,
        "dropout": manifest.dropout,
    }
    for name, expected_value in expected_config.items():
        if checkpoint_config.get(name) != expected_value:
            raise ValueError(f"Checkpoint {name} does not match the model manifest")

    model_classes = {"gcn": GCNRegressor, "gin": GINRegressor}
    if manifest.model_type not in model_classes:
        raise ValueError(f"Unsupported promoted model type: {manifest.model_type}")
    model = model_classes[manifest.model_type](
        input_dim=manifest.atom_feature_dim,
        hidden_dim=manifest.hidden_dim,
        num_layers=manifest.num_layers,
        dropout=manifest.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    reference_index = None
    if manifest.reference_index_path is not None:
        reference_path = manifest_path.parent / manifest.reference_index_path
        reference_index = load_reference_index(reference_path)
        if manifest.reference_index_size != len(reference_index):
            raise ValueError("Reference index size does not match the model manifest")
        if manifest.reference_index_radius != reference_index.radius:
            raise ValueError("Reference index radius does not match the model manifest")
        if manifest.reference_index_n_bits != reference_index.n_bits:
            raise ValueError("Reference index bit count does not match the model manifest")
    return LoadedPromotedModel(
        manifest=manifest,
        model=model,
        device=resolved_device,
        target_mean=float(checkpoint["target_mean"]),
        target_std=float(checkpoint["target_std"]),
        reference_index=reference_index,
    )


def _prediction_warnings(smiles: str, dataset_name: str) -> list[str]:
    descriptors = molecular_descriptors(smiles)
    warnings = [
        f"This model was trained on {dataset_name.upper()}; predictions outside similar "
        "chemical space may be unreliable."
    ]
    if descriptors["molecular_weight"] > 1000 or descriptors["heavy_atom_count"] > 100:
        warnings.append(
            "This molecule is unusually large relative to typical small-molecule ESOL data."
        )
    return warnings


def predict_smiles(
    smiles: str,
    loaded_model: LoadedPromotedModel,
    *,
    include_applicability: bool = False,
    top_k: int = 5,
) -> dict:
    """Predict aqueous log solubility for one valid SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must not be blank")
    graph = featurize_smiles(smiles)
    data = molecule_graph_to_pyg_data(graph).to(loaded_model.device)
    with torch.no_grad():
        normalized_prediction = float(loaded_model.model(data).item())
    log_solubility = (
        normalized_prediction * loaded_model.target_std + loaded_model.target_mean
    )
    molar_solubility = 10.0**log_solubility
    if not math.isfinite(log_solubility) or not math.isfinite(molar_solubility):
        raise RuntimeError("Model produced a non-finite solubility prediction")
    result = {
        "input_smiles": smiles,
        "canonical_smiles": graph.canonical_smiles,
        "predicted_log_solubility": float(log_solubility),
        "predicted_solubility_mol_per_litre": float(molar_solubility),
        "model_id": loaded_model.manifest.model_id,
        "model_type": loaded_model.manifest.model_type,
        "dataset_name": loaded_model.manifest.dataset_name,
        "warnings": _prediction_warnings(
            graph.canonical_smiles,
            loaded_model.manifest.dataset_name,
        ),
    }
    if include_applicability and loaded_model.reference_index is not None:
        from molgnn_ops.applicability import molecule_context

        context = molecule_context(
            graph.canonical_smiles,
            loaded_model.reference_index,
            top_k=top_k,
        )
        result["applicability"] = {
            "maximum_similarity": context["maximum_similarity"],
            "mean_top_k_similarity": context["mean_top_k_similarity"],
            "warnings": context["warnings"],
        }
    return result


def predict_smiles_with_context(
    smiles: str,
    loaded_model: LoadedPromotedModel,
    top_k: int = 5,
) -> dict:
    """Predict one molecule and return training-set applicability context."""
    from molgnn_ops.applicability import molecule_context

    if loaded_model.reference_index is None:
        raise ValueError("The promoted model has no training reference index")
    prediction = predict_smiles(smiles, loaded_model)
    context = molecule_context(smiles, loaded_model.reference_index, top_k=top_k)
    return {
        "prediction": prediction,
        "molecular_descriptors": context["query_descriptors"],
        "applicability": {
            "maximum_similarity": context["maximum_similarity"],
            "mean_top_k_similarity": context["mean_top_k_similarity"],
            "descriptor_ranges": context["descriptor_ranges"],
            "warnings": context["warnings"],
        },
        "nearest_training_molecules": context["nearest_neighbors"],
    }


def predict_smiles_batch(
    smiles_values: list[str],
    loaded_model: LoadedPromotedModel,
) -> list[dict]:
    """Predict an ordered batch while isolating per-item validation errors."""
    results = []
    for smiles in smiles_values:
        try:
            prediction = predict_smiles(smiles, loaded_model)
            results.append(
                {
                    "input_smiles": smiles,
                    "success": True,
                    "prediction": prediction,
                    "error": None,
                }
            )
        except (RuntimeError, ValueError) as error:
            results.append(
                {
                    "input_smiles": smiles,
                    "success": False,
                    "prediction": None,
                    "error": str(error),
                }
            )
    return results
