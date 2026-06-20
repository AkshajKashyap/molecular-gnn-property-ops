from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from molgnn_ops.datasets import load_csv_dataset
from molgnn_ops.paths import ensure_project_dirs
from molgnn_ops.prep import prepare_dataset

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


@app.command("prepare-csv")
def prepare_csv(
    input_csv: Annotated[Path, typer.Argument(help="Source CSV dataset.")],
    output_csv: Annotated[Path, typer.Argument(help="Destination for the prepared CSV.")],
    smiles_col: Annotated[str, typer.Option(help="Column containing SMILES strings.")],
    target_col: Annotated[str, typer.Option(help="Column containing prediction targets.")],
    dataset_name: Annotated[str, typer.Option(help="Name assigned to each record.")],
    split_strategy: Annotated[
        Literal["random", "scaffold"],
        typer.Option(help="Dataset split strategy."),
    ],
    seed: Annotated[int, typer.Option(help="Random seed used for splitting.")] = 42,
) -> None:
    """Prepare a molecular CSV and add persistent split metadata."""
    summary = prepare_dataset(
        input_csv=input_csv,
        output_csv=output_csv,
        smiles_col=smiles_col,
        target_col=target_col,
        dataset_name=dataset_name,
        split_strategy=split_strategy,
        seed=seed,
    )

    console.print("[bold]Prepared dataset[/bold]")
    console.print(f"Dataset: {summary.dataset_name}")
    console.print(f"Rows: {summary.n_valid_smiles}/{summary.n_rows} nonblank SMILES")
    console.print(f"Missing targets: {summary.n_missing_targets}")
    console.print(f"Split strategy: {summary.split_strategy}")
    console.print(
        f"Train/val/test: {summary.n_train}/{summary.n_val}/{summary.n_test}"
    )
    console.print(f"Output: {summary.output_path}")


if __name__ == "__main__":
    app()
