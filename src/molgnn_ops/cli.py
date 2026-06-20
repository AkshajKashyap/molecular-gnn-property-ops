import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from molgnn_ops.baselines import train_fingerprint_baseline
from molgnn_ops.benchmark_compare import run_split_comparison
from molgnn_ops.data_sources import list_dataset_specs
from molgnn_ops.datasets import load_csv_dataset
from molgnn_ops.diagnostics import (
    prediction_error_summary,
    scaffold_distribution_summary,
    split_target_summary,
    train_test_similarity_summary,
    worst_predictions,
)
from molgnn_ops.download import download_dataset
from molgnn_ops.featurization import featurize_records_from_csv
from molgnn_ops.fingerprints import featurize_fingerprints_from_csv
from molgnn_ops.paths import ARTIFACTS_DIR, ensure_project_dirs
from molgnn_ops.prep import prepare_dataset
from molgnn_ops.reporting import write_diagnostics_report
from molgnn_ops.workflows import run_fingerprint_benchmark, run_gnn_benchmark

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


@app.command("list-datasets")
def list_datasets() -> None:
    """List molecular benchmark datasets available in the registry."""
    table = Table(title="Available molecular datasets")
    table.add_column("Name")
    table.add_column("Task")
    table.add_column("Default split")
    table.add_column("Description")
    for spec in list_dataset_specs():
        table.add_row(
            spec.name,
            spec.task_type,
            spec.default_split_strategy,
            spec.description,
        )
    console.print(table)


@app.command("download-dataset")
def download_dataset_command(
    dataset_name: Annotated[str, typer.Argument(help="Registered dataset name.")],
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing raw dataset file."),
    ] = False,
) -> None:
    """Download a registered molecular benchmark dataset."""
    output_path = download_dataset(dataset_name, overwrite=overwrite)
    console.print(f"Downloaded dataset: {dataset_name}")
    console.print(f"Raw CSV: {output_path}")


@app.command("run-fingerprint-benchmark")
def run_fingerprint_benchmark_command(
    dataset_name: Annotated[str, typer.Argument(help="Registered dataset name.")],
    split_strategy: Annotated[
        Literal["random", "scaffold"],
        typer.Option(help="Dataset split strategy."),
    ] = "scaffold",
    seed: Annotated[int, typer.Option(help="Random seed for splitting and training.")] = 42,
    radius: Annotated[int, typer.Option(help="Morgan fingerprint radius.")] = 2,
    n_bits: Annotated[int, typer.Option(help="Fingerprint bit count.")] = 2048,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace cached source and benchmark artifacts."),
    ] = False,
) -> None:
    """Run a reproducible classical benchmark from download through report."""
    summary = run_fingerprint_benchmark(
        dataset_name,
        ARTIFACTS_DIR / "benchmarks",
        split_strategy=split_strategy,
        seed=seed,
        radius=radius,
        n_bits=n_bits,
        overwrite=overwrite,
    )

    console.print("[bold]Completed fingerprint benchmark[/bold]")
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Task type: {summary['task_type']}")
    console.print(f"Split strategy: {summary['split_strategy']}")
    console.print(f"Best model: {summary['best_model']}")
    console.print(
        f"Validation {summary['key_metric']}: {summary['validation_metric']}"
    )
    console.print(f"Test {summary['key_metric']}: {summary['test_metric']}")
    console.print(f"Metrics: {summary['metrics_json']}")
    console.print(f"Report: {summary['report_md']}")
    console.print(f"Summary: {summary['summary_json']}")


@app.command("diagnose-benchmark")
def diagnose_benchmark(
    prepared_csv: Annotated[Path, typer.Argument(help="Prepared benchmark CSV.")],
    predictions_csv: Annotated[Path, typer.Argument(help="Baseline predictions CSV.")],
    output_dir: Annotated[Path, typer.Argument(help="Diagnostics artifact directory.")],
) -> None:
    """Generate benchmark diagnostics, figures, and a Markdown report."""
    from molgnn_ops.plots import (
        plot_absolute_error_histogram,
        plot_predicted_vs_actual,
        plot_target_distribution,
        plot_test_similarity_histogram,
    )

    figures_dir = output_dir / "figures"
    plot_paths = {
        "target_distribution": figures_dir / "target_distribution.png",
        "predicted_vs_actual": figures_dir / "predicted_vs_actual.png",
        "absolute_error_histogram": figures_dir / "absolute_error_histogram.png",
        "test_similarity_histogram": figures_dir / "test_similarity_histogram.png",
    }
    plot_target_distribution(prepared_csv, plot_paths["target_distribution"])
    plot_predicted_vs_actual(predictions_csv, plot_paths["predicted_vs_actual"])
    plot_absolute_error_histogram(predictions_csv, plot_paths["absolute_error_histogram"])
    plot_test_similarity_histogram(prepared_csv, plot_paths["test_similarity_histogram"])

    diagnostics = {
        "target_distribution": split_target_summary(prepared_csv),
        "prediction_errors": prediction_error_summary(predictions_csv),
        "worst_test_predictions": worst_predictions(predictions_csv),
        "scaffold_distribution": scaffold_distribution_summary(prepared_csv),
        "train_test_similarity": train_test_similarity_summary(prepared_csv),
        "plots": {
            name: path.relative_to(output_dir).as_posix()
            for name, path in plot_paths.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_json = output_dir / "diagnostics.json"
    report_md = output_dir / "diagnostics_report.md"
    diagnostics_json.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_diagnostics_report(diagnostics, report_md)

    test_errors = diagnostics["prediction_errors"].get("test", {})
    similarity = diagnostics["train_test_similarity"]
    scaffolds = diagnostics["scaffold_distribution"]
    console.print("[bold]Completed benchmark diagnostics[/bold]")
    console.print(f"Test RMSE: {test_errors.get('rmse')}")
    console.print(f"Test MAE: {test_errors.get('mae')}")
    console.print(f"Unique scaffolds: {scaffolds['n_unique_scaffolds']}")
    console.print(f"Mean max train similarity: {similarity['mean_max_similarity']}")
    console.print(f"Diagnostics: {diagnostics_json}")
    console.print(f"Report: {report_md}")


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise typer.BadParameter("seeds must be comma-separated integers") from error
    if not seeds:
        raise typer.BadParameter("at least one seed is required")
    return seeds


def _parse_split_strategies(value: str) -> list[str]:
    strategies = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(strategies).difference({"random", "scaffold"}))
    if not strategies or invalid:
        raise typer.BadParameter("split strategies must be random and/or scaffold")
    return strategies


@app.command("compare-splits")
def compare_splits(
    dataset_name: Annotated[str, typer.Argument(help="Registered dataset name.")],
    output_dir: Annotated[Path, typer.Argument(help="Comparison artifact directory.")],
    seeds: Annotated[str, typer.Option(help="Comma-separated random seeds.")] = "42,43,44",
    split_strategies: Annotated[
        str,
        typer.Option(help="Comma-separated split strategies."),
    ] = "random,scaffold",
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace cached benchmark artifacts."),
    ] = False,
) -> None:
    """Compare fingerprint baselines across split strategies and seeds."""
    summary = run_split_comparison(
        dataset_name,
        output_dir,
        seeds=_parse_seeds(seeds),
        split_strategies=_parse_split_strategies(split_strategies),
        overwrite=overwrite,
    )

    console.print("[bold]Completed split comparison[/bold]")
    for strategy, values in summary["by_split_strategy"].items():
        console.print(
            f"{strategy}: mean test {values['key_metric']}="
            f"{values['mean_test_metric']} (n={values['n_runs']})"
        )
    console.print(f"Metrics: {summary['comparison_metrics_csv']}")
    console.print(f"Summary: {summary['comparison_summary_json']}")
    console.print(f"Report: {summary['comparison_report_md']}")


@app.command("train-gnn-regressor")
def train_gnn_regressor_command(
    graph_jsonl: Annotated[Path, typer.Argument(help="Labeled molecular graph JSONL.")],
    output_dir: Annotated[Path, typer.Argument(help="GNN artifact directory.")],
    model_name: Annotated[
        Literal["gcn", "gin"],
        typer.Option(help="Graph neural network architecture."),
    ] = "gcn",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    epochs: Annotated[int, typer.Option(help="Maximum training epochs.")] = 50,
    batch_size: Annotated[int, typer.Option(help="Graphs per training batch.")] = 32,
    hidden_dim: Annotated[int, typer.Option(help="Hidden feature width.")] = 64,
    num_layers: Annotated[int, typer.Option(help="Message-passing layer count.")] = 3,
    dropout: Annotated[float, typer.Option(help="Dropout probability.")] = 0.1,
) -> None:
    """Train a minimal GCN or GIN molecular regression baseline."""
    from molgnn_ops.gnn_train import train_gnn_regressor

    summary = train_gnn_regressor(
        graph_jsonl,
        output_dir,
        model_name=model_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    console.print("[bold]Completed GNN regression training[/bold]")
    console.print(f"Model: {summary['model_name']}")
    console.print(f"Device: {summary['device']}")
    console.print(f"Best epoch: {summary['best_epoch']}")
    console.print(f"Best validation RMSE: {summary['best_val_rmse']}")
    console.print(f"Test RMSE: {summary['test_rmse']}")
    console.print(f"Metrics: {summary['artifacts']['metrics']}")
    console.print(f"Report: {summary['artifacts']['report']}")


@app.command("run-gnn-benchmark")
def run_gnn_benchmark_command(
    dataset_name: Annotated[str, typer.Argument(help="Registered regression dataset.")],
    output_dir: Annotated[Path, typer.Argument(help="GNN benchmark artifact directory.")],
    split_strategy: Annotated[
        Literal["random", "scaffold"],
        typer.Option(help="Dataset split strategy."),
    ] = "scaffold",
    model_name: Annotated[
        Literal["gcn", "gin"],
        typer.Option(help="Graph neural network architecture."),
    ] = "gcn",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    epochs: Annotated[int, typer.Option(help="Maximum training epochs.")] = 50,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace cached source and benchmark artifacts."),
    ] = False,
) -> None:
    """Run dataset preparation, graph featurization, and GNN training."""
    summary = run_gnn_benchmark(
        dataset_name,
        output_dir,
        split_strategy=split_strategy,
        model_name=model_name,
        seed=seed,
        epochs=epochs,
        overwrite=overwrite,
    )
    console.print("[bold]Completed molecular GNN benchmark[/bold]")
    console.print(f"Dataset: {summary['dataset_name']}")
    console.print(f"Model: {summary['model_name']}")
    console.print(f"Split strategy: {summary['split_strategy']}")
    console.print(f"Best epoch: {summary['best_epoch']}")
    console.print(f"Best validation RMSE: {summary['best_val_rmse']}")
    console.print(f"Test RMSE: {summary['test_rmse']}")
    console.print(f"Metrics: {summary['metrics_json']}")
    console.print(f"Report: {summary['report_md']}")
    console.print(f"Summary: {summary['summary_json']}")


if __name__ == "__main__":
    app()
