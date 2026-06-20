from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from molgnn_ops.datasets import load_csv_dataset
from molgnn_ops.paths import ensure_project_dirs

app = typer.Typer(help="Utilities for the molecular property prediction project.")
console = Console()


@app.command("init-dirs")
def init_dirs() -> None:
    """Create the project's expected data and output directories."""
    ensure_project_dirs()
    console.print("Project directories created.")


@app.command("inspect-csv")
def inspect_csv(
    path: Annotated[Path, typer.Argument(help="Path to the CSV dataset.")],
    smiles_col: Annotated[str, typer.Option(help="Column containing SMILES strings.")],
    dataset_name: Annotated[str, typer.Option(help="Name assigned to each record.")],
    target_col: Annotated[
        str | None,
        typer.Option(help="Optional column containing prediction targets."),
    ] = None,
) -> None:
    """Load a CSV and print a compact summary of its molecular records."""
    records = load_csv_dataset(path, smiles_col, target_col, dataset_name)
    missing_targets = sum(record.target is None for record in records)

    console.print(f"Row count: {len(records)}")
    console.print(f"Missing targets: {missing_targets}")
    console.print("First 5 records:")
    for record in records[:5]:
        console.print(record.model_dump())


if __name__ == "__main__":
    app()
