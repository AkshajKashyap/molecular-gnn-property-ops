from molgnn_ops.gnn_error_analysis import molecular_descriptors
from molgnn_ops.reference_index import (
    DESCRIPTOR_NAMES,
    ReferenceIndex,
    find_similar_molecules,
)


def molecule_context(
    smiles: str,
    reference_index: ReferenceIndex | None,
    top_k: int = 5,
) -> dict:
    """Describe query similarity and descriptor coverage without claiming certainty."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    query_descriptors = molecular_descriptors(smiles)
    if reference_index is None:
        return {
            "nearest_neighbors": [],
            "maximum_similarity": None,
            "mean_top_k_similarity": None,
            "query_descriptors": query_descriptors,
            "descriptor_ranges": {},
            "warnings": ["No training reference index is available for applicability context."],
        }

    nearest_neighbors = find_similar_molecules(smiles, reference_index, top_k=top_k)
    similarities = [neighbor["tanimoto_similarity"] for neighbor in nearest_neighbors]
    maximum_similarity = max(similarities)
    descriptor_ranges = {
        name: {
            "minimum": float(reference_index.descriptors[name].min()),
            "maximum": float(reference_index.descriptors[name].max()),
        }
        for name in DESCRIPTOR_NAMES
    }
    warnings = []
    if maximum_similarity < 0.3:
        warnings.append(
            "The query is structurally dissimilar to the training reference set."
        )
    elif maximum_similarity < 0.5:
        warnings.append(
            "The query has limited structural similarity to the training reference set."
        )
    for descriptor_name, value in query_descriptors.items():
        observed = descriptor_ranges[descriptor_name]
        if value < observed["minimum"] or value > observed["maximum"]:
            readable_name = descriptor_name.replace("_", " ")
            warnings.append(
                f"The query {readable_name} is outside the observed training range "
                f"[{observed['minimum']:.3f}, {observed['maximum']:.3f}]."
            )
    if (
        query_descriptors["molecular_weight"] > 1000
        or query_descriptors["heavy_atom_count"] > 100
    ):
        warnings.append(
            "This molecule is unusually large relative to typical small-molecule ESOL data."
        )
    warnings.append(
        "Structural similarity is descriptive context and does not guarantee prediction accuracy."
    )
    return {
        "nearest_neighbors": nearest_neighbors,
        "maximum_similarity": maximum_similarity,
        "mean_top_k_similarity": sum(similarities) / len(similarities),
        "query_descriptors": query_descriptors,
        "descriptor_ranges": descriptor_ranges,
        "warnings": warnings,
    }
