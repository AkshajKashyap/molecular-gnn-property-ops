import os
from pathlib import Path

import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from molgnn_ops.inference import load_promoted_model, predict_smiles_with_context

EXAMPLE_MOLECULES = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"]


def render_molecule(smiles: str, size: tuple[int, int] = (420, 300)) -> Image.Image:
    """Render a valid SMILES string as a two-dimensional molecule image."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Draw.MolToImage(molecule, size=size)


def format_metric(value: float | None, digits: int = 4) -> str:
    """Format optional numeric metrics for dashboard display."""
    return "N/A" if value is None else f"{value:.{digits}f}"


def nearest_neighbors_table(neighbors: list[dict]) -> pd.DataFrame:
    """Create a compact display table from nearest-neighbor records."""
    rows = []
    for neighbor in neighbors:
        descriptors = neighbor["descriptors"]
        rows.append(
            {
                "canonical_smiles": neighbor["canonical_smiles"],
                "measured_log_solubility": neighbor["measured_target"],
                "tanimoto_similarity": neighbor["tanimoto_similarity"],
                "molecular_weight": descriptors["molecular_weight"],
                "heavy_atom_count": descriptors["heavy_atom_count"],
                "ring_count": descriptors["ring_count"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the Streamlit molecule explorer."""
    import streamlit as st

    st.set_page_config(page_title="Molecular Solubility Explorer", layout="wide")
    default_manifest = os.environ.get(
        "MOLGNN_MANIFEST_PATH",
        "artifacts/registry/esol-gcn-v1/manifest.json",
    )
    manifest_value = st.sidebar.text_input("Promoted model manifest", default_manifest)

    @st.cache_resource
    def load_dashboard_model(manifest_path: str):
        return load_promoted_model(Path(manifest_path))

    try:
        loaded_model = load_dashboard_model(manifest_value)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Unable to load promoted model: {error}")
        st.stop()
    manifest = loaded_model.manifest

    st.sidebar.header("Promoted model")
    st.sidebar.write(f"**Model ID:** {manifest.model_id}")
    st.sidebar.write(f"**Model type:** {manifest.model_type.upper()}")
    st.sidebar.write(f"**Dataset:** {manifest.dataset_name.upper()}")
    st.sidebar.write(
        f"**Validation RMSE:** {format_metric(manifest.validation_metrics.get('rmse'))}"
    )
    st.sidebar.write(f"**Test RMSE:** {format_metric(manifest.test_metrics.get('rmse'))}")
    st.sidebar.write(f"**Split:** {manifest.split_strategy}")
    st.sidebar.write(f"**Split seed:** {manifest.split_seed}")
    st.sidebar.write(f"**Model seed:** {manifest.model_seed}")
    st.sidebar.warning(
        "This is an ML estimate trained on ESOL, not laboratory, medical, or "
        "pharmaceutical guidance. Training-set similarity is applicability context only."
    )

    st.title("Molecular Solubility Explorer")
    st.write(
        "Explore a promoted GNN estimate of aqueous log solubility, molecular descriptors, "
        "and structurally similar molecules from the training split."
    )
    example = st.selectbox("Example molecules", EXAMPLE_MOLECULES)
    smiles = st.text_input("SMILES", value=example)
    top_k = st.slider("Nearest training molecules", min_value=1, max_value=20, value=5)

    if not st.button("Predict", type="primary"):
        return
    try:
        result = predict_smiles_with_context(smiles, loaded_model, top_k=top_k)
    except ValueError as error:
        st.error(str(error))
        return

    prediction = result["prediction"]
    applicability = result["applicability"]
    image_column, prediction_column = st.columns([1, 1])
    with image_column:
        st.subheader("Molecule")
        st.image(render_molecule(prediction["canonical_smiles"]))
        st.code(prediction["canonical_smiles"])
    with prediction_column:
        st.subheader("Solubility estimate")
        st.metric(
            "Predicted log solubility",
            format_metric(prediction["predicted_log_solubility"]),
        )
        st.metric(
            "Predicted solubility (mol/L)",
            f"{prediction['predicted_solubility_mol_per_litre']:.6g}",
        )
        st.caption("This output is an ML estimate, not a laboratory measurement.")

    st.subheader("Molecular descriptors")
    st.dataframe(pd.DataFrame([result["molecular_descriptors"]]), hide_index=True)

    st.subheader("Training-set structural similarity")
    similarity_columns = st.columns(2)
    similarity_columns[0].metric(
        "Maximum Tanimoto similarity",
        format_metric(applicability["maximum_similarity"]),
    )
    similarity_columns[1].metric(
        "Mean top-k Tanimoto similarity",
        format_metric(applicability["mean_top_k_similarity"]),
    )
    for warning in applicability["warnings"]:
        st.warning(warning)
    st.caption(
        "Nearest neighbors provide applicability context, not a reliability guarantee. "
        "Similar molecules can still have different measured properties."
    )

    neighbors = result["nearest_training_molecules"]
    st.subheader("Nearest training molecules")
    st.dataframe(nearest_neighbors_table(neighbors), hide_index=True)
    if st.checkbox("Render nearest-neighbor molecules"):
        columns = st.columns(min(3, len(neighbors)))
        for index, neighbor in enumerate(neighbors):
            with columns[index % len(columns)]:
                st.image(render_molecule(neighbor["canonical_smiles"], size=(260, 180)))
                st.caption(
                    f"Similarity {neighbor['tanimoto_similarity']:.3f}; "
                    f"measured logS {neighbor['measured_target']:.3f}"
                )


if __name__ == "__main__":
    main()
