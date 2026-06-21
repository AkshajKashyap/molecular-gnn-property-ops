import json
from pathlib import Path

from molgnn_ops.baselines import train_fingerprint_baseline
from molgnn_ops.data_sources import get_dataset_spec
from molgnn_ops.download import download_dataset
from molgnn_ops.featurization import featurize_records_from_csv
from molgnn_ops.fingerprints import featurize_fingerprints_from_csv
from molgnn_ops.prep import prepare_dataset


def run_fingerprint_benchmark(
    dataset_name: str,
    output_dir: Path,
    split_strategy: str | None = None,
    seed: int = 42,
    radius: int = 2,
    n_bits: int = 2048,
    overwrite: bool = False,
) -> dict:
    """Run the complete download-to-report classical benchmark workflow."""
    spec = get_dataset_spec(dataset_name)
    resolved_split_strategy = split_strategy or spec.default_split_strategy
    run_dir = output_dir / spec.name / f"seed_{seed}"
    summary_path = run_dir / "benchmark_summary.json"
    if summary_path.is_file() and not overwrite:
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        requested_config = {
            "dataset_name": spec.name,
            "split_strategy": resolved_split_strategy,
            "seed": seed,
            "radius": radius,
            "n_bits": n_bits,
        }
        if any(cached_summary.get(key) != value for key, value in requested_config.items()):
            raise ValueError(
                f"Benchmark directory {run_dir} contains a different configuration; "
                "set overwrite=True to replace it"
            )
        artifact_keys = ("prepared_csv", "fingerprint_npz", "metrics_json", "report_md")
        if all(Path(cached_summary[key]).is_file() for key in artifact_keys):
            return cached_summary

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = download_dataset(spec.name, overwrite=overwrite)
    prepared_csv = run_dir / "prepared.csv"
    fingerprint_npz = run_dir / "fingerprints.npz"
    baseline_output_dir = run_dir / "baseline"

    preparation_summary = prepare_dataset(
        input_csv=raw_csv,
        output_csv=prepared_csv,
        smiles_col=spec.smiles_col,
        target_col=spec.target_col,
        dataset_name=spec.name,
        split_strategy=resolved_split_strategy,
        seed=seed,
    )
    fingerprint_summary = featurize_fingerprints_from_csv(
        prepared_csv,
        fingerprint_npz,
        radius=radius,
        n_bits=n_bits,
    )
    metrics = train_fingerprint_baseline(
        fingerprint_npz,
        baseline_output_dir,
        task_type=spec.task_type,
        seed=seed,
    )

    best_model = str(metrics["best_model"])
    selection_metric = str(metrics["selection_metric"])
    model_results = metrics["models"]
    validation_metrics = model_results[best_model]["validation"]
    test_metrics = metrics["test_metrics"]
    metrics_json = baseline_output_dir / "metrics.json"
    report_md = baseline_output_dir / "report.md"

    summary = {
        "dataset_name": spec.name,
        "task_type": spec.task_type,
        "split_strategy": resolved_split_strategy,
        "seed": seed,
        "radius": radius,
        "n_bits": n_bits,
        "raw_csv": str(raw_csv),
        "prepared_csv": str(prepared_csv),
        "fingerprint_npz": str(fingerprint_npz),
        "baseline_output_dir": str(baseline_output_dir),
        "metrics_json": str(metrics_json),
        "report_md": str(report_md),
        "summary_json": str(summary_path),
        "best_model": best_model,
        "key_metric": selection_metric,
        "validation_metric": validation_metrics.get(selection_metric),
        "test_metric": test_metrics.get(selection_metric),
        "preparation": preparation_summary.model_dump(mode="json"),
        "fingerprints": fingerprint_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_gnn_benchmark(
    dataset_name: str,
    output_dir: Path,
    split_strategy: str = "scaffold",
    model_name: str = "gcn",
    seed: int = 42,
    epochs: int = 50,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    overwrite: bool = False,
) -> dict:
    """Run the complete download-to-report molecular GNN benchmark workflow."""
    from molgnn_ops.gnn_train import train_gnn_regressor

    spec = get_dataset_spec(dataset_name)
    if spec.task_type != "regression":
        raise ValueError("GNN benchmark currently supports regression datasets only")
    summary_path = output_dir / "gnn_benchmark_summary.json"
    if summary_path.is_file() and not overwrite:
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        requested_config = {
            "dataset_name": spec.name,
            "split_strategy": split_strategy,
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        if any(cached_summary.get(key) != value for key, value in requested_config.items()):
            raise ValueError(
                f"Benchmark directory {output_dir} contains a different configuration; "
                "set overwrite=True to replace it"
            )
        artifact_keys = ("graph_jsonl", "metrics_json", "report_md", "model_checkpoint")
        if all(Path(cached_summary[key]).is_file() for key in artifact_keys):
            return cached_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = download_dataset(spec.name, overwrite=overwrite)
    prepared_csv = output_dir / "prepared.csv"
    graph_jsonl = output_dir / "graphs.jsonl"
    training_output_dir = output_dir / "training"
    preparation_summary = prepare_dataset(
        input_csv=raw_csv,
        output_csv=prepared_csv,
        smiles_col=spec.smiles_col,
        target_col=spec.target_col,
        dataset_name=spec.name,
        split_strategy=split_strategy,
        seed=seed,
    )
    graph_summary = featurize_records_from_csv(prepared_csv, graph_jsonl)
    training_summary = train_gnn_regressor(
        graph_jsonl,
        training_output_dir,
        model_name=model_name,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    artifacts = training_summary["artifacts"]
    summary = {
        "dataset_name": spec.name,
        "task_type": spec.task_type,
        "split_strategy": split_strategy,
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "raw_csv": str(raw_csv),
        "prepared_csv": str(prepared_csv),
        "graph_jsonl": str(graph_jsonl),
        "training_output_dir": str(training_output_dir),
        "metrics_json": artifacts["metrics"],
        "report_md": artifacts["report"],
        "model_checkpoint": artifacts["model"],
        "summary_json": str(summary_path),
        "best_epoch": training_summary["best_epoch"],
        "best_val_rmse": training_summary["best_val_rmse"],
        "test_rmse": training_summary["test_rmse"],
        "validation_metrics": training_summary.get(
            "validation_metrics",
            {"rmse": training_summary["best_val_rmse"]},
        ),
        "test_metrics": training_summary.get(
            "test_metrics",
            {"rmse": training_summary["test_rmse"]},
        ),
        "fingerprint_comparison": training_summary["fingerprint_comparison"],
        "preparation": preparation_summary.model_dump(mode="json"),
        "graphs": graph_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
