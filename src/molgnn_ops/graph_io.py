from pathlib import Path

from molgnn_ops.featurization import MoleculeGraph


def save_graphs_jsonl(graphs: list[MoleculeGraph], path: Path) -> None:
    """Write one serialized molecular graph per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for graph in graphs:
            output_file.write(graph.model_dump_json())
            output_file.write("\n")


def load_graphs_jsonl(path: Path) -> list[MoleculeGraph]:
    """Load molecular graphs from a JSON Lines file."""
    with path.open(encoding="utf-8") as input_file:
        return [
            MoleculeGraph.model_validate_json(line)
            for line in input_file
            if line.strip()
        ]
