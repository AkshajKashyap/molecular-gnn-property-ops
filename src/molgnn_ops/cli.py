from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from molgnn_ops.baselines import train_fingerprint_baseline
from molgnn_ops.datasets import load_csv_dataset
from molgnn_ops.featurization import featurize_records_from_csv
from molgnn_ops.fingerprints import featurize_fingerprints_from_csv
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


@app.command("featurize-csv")
def featurize_csv(
    input_csv: Annotated[Path, typer.Argument(help="Prepared molecular CSV.")],
    output_jsonl: Annotated[Path, typer.Argument(help="Destination JSONL graph file.")],
) -> None:
    """Convert molecular CSV rows into inspectable graph records."""
    summary = featurize_records_from_csv(input_csv, output_jsonl)

    console.print("[bold]Featurized molecular dataset[/bold]")
    console.print(f"Rows: {summary['n_rows']}")
    console.print(f"Valid: {summary['n_valid']}")
    console.print(f"Invalid: {summary['n_invalid']}")
    console.print(f"Graphs written: {summary['n_graphs_written']}")
    console.print(f"Output: {output_jsonl}")


@app.command("fingerprint-csv")
def fingerprint_csv(
    input_csv: Annotated[Path, typer.Argument(help="Prepared molecular CSV.")],
    output_npz: Annotated[Path, typer.Argument(help="Destination fingerprint NPZ file.")],
    radius: Annotated[int, typer.Option(help="Morgan fingerprint radius.")] = 2,
    n_bits: Annotated[int, typer.Option(help="Fingerprint bit count.")] = 2048,
) -> None:
    """Convert prepared molecular rows into Morgan fingerprints."""
    summary = featurize_fingerprints_from_csv(
        input_csv,
        output_npz,
        radius=radius,
        n_bits=n_bits,
    )

    console.print("[bold]Featurized Morgan fingerprints[/bold]")
    console.print(f"Rows: {summary['n_rows']}")
    console.print(f"Valid SMILES: {summary['n_valid']}")
    console.print(f"Invalid SMILES: {summary['n_invalid']}")
    console.print(f"Missing targets: {summary['n_missing_targets']}")
    console.print(f"Rows written: {summary['n_written']}")
    console.print(f"Output: {output_npz}")


@app.command("train-fingerprint-baseline")
def train_fingerprint_baseline_command(
    input_npz: Annotated[Path, typer.Argument(help="Fingerprint NPZ dataset.")],
    output_dir: Annotated[Path, typer.Argument(help="Baseline artifact directory.")],
    task_type: Annotated[
        Literal["auto", "classification", "regression"],
        typer.Option(help="Prediction task type."),
    ] = "auto",
    seed: Annotated[int, typer.Option(help="Random seed for model training.")] = 42,
) -> None:
    """Train and evaluate classical fingerprint baseline models."""
    metrics = train_fingerprint_baseline(
        input_npz,
        output_dir,
        task_type=task_type,
        seed=seed,
    )
    best_model = str(metrics["best_model"])
    model_results = metrics["models"]
    validation_metrics = model_results[best_model]["validation"]
    test_metrics = metrics["test_metrics"]

    console.print("[bold]Trained Morgan fingerprint baseline[/bold]")
    console.print(f"Task type: {metrics['task_type']}")
    console.print(f"Best model: {best_model}")
    console.print("Validation metrics:")
    for metric_name, value in validation_metrics.items():
        console.print(f"  {metric_name}: {value}")
    console.print("Test metrics:")
    for metric_name, value in test_metrics.items():
        console.print(f"  {metric_name}: {value}")
    console.print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    app()
