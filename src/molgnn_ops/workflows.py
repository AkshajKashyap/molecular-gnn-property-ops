import json
import re
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


def _seed_from_prediction_path(path: Path, fallback: int) -> int:
    match = re.search(r"seed[_-](\d+)", str(path))
    return int(match.group(1)) if match else fallback


def run_gnn_uncertainty_analysis(
    prediction_paths: list[Path],
    output_dir: Path,
    target_coverages: list[float] | None = None,
) -> dict:
    """Calibrate and evaluate a repeated-run GNN regression ensemble."""
    import pandas as pd

    from molgnn_ops.gnn_error_analysis import (
        attach_molecular_descriptors,
        descriptor_error_summary,
        uncertainty_bucket_summary,
        worst_ensemble_predictions,
    )
    from molgnn_ops.gnn_uncertainty import (
        add_prediction_intervals,
        compute_ensemble_predictions,
        fit_interval_scale,
        interval_metrics,
        load_ensemble_predictions,
        selective_prediction_metrics,
        uncertainty_error_correlation,
    )
    from molgnn_ops.plots import (
        plot_interval_coverage,
        plot_selective_prediction_curve,
        plot_uncertainty_buckets,
        plot_uncertainty_vs_error,
    )
    from molgnn_ops.reporting import write_gnn_uncertainty_report

    if len(prediction_paths) < 2:
        raise ValueError("At least two prediction files are required for an ensemble")
    requested_coverages = target_coverages or [0.80, 0.90, 0.95]
    if not requested_coverages or any(
        not 0 < coverage < 1 for coverage in requested_coverages
    ):
        raise ValueError("target_coverages must contain values between 0 and 1")
    if len(set(requested_coverages)) != len(requested_coverages):
        raise ValueError("target_coverages must be unique")
    evaluated_coverages = sorted({*requested_coverages, 0.90})

    aligned = load_ensemble_predictions(prediction_paths)
    ensemble = compute_ensemble_predictions(aligned)
    validation = ensemble[ensemble["split"] == "val"].copy()
    test = ensemble[ensemble["split"] == "test"].copy()
    if validation.empty or test.empty:
        raise ValueError("Ensemble predictions must contain non-empty val and test splits")

    interval_results = []
    interval_scales = {}
    for target_coverage in evaluated_coverages:
        interval_scale = fit_interval_scale(validation, target_coverage)
        interval_scales[target_coverage] = interval_scale
        metrics = interval_metrics(add_prediction_intervals(test, interval_scale))
        interval_results.append(
            {
                "target_coverage": target_coverage,
                "interval_scale": interval_scale,
                **metrics,
            }
        )

    detailed_scale = interval_scales[0.90]
    detailed_predictions = add_prediction_intervals(ensemble, detailed_scale)
    detailed_predictions = attach_molecular_descriptors(detailed_predictions)
    detailed_test = detailed_predictions[detailed_predictions["split"] == "test"].copy()
    test_metrics = interval_metrics(detailed_test)
    correlations = uncertainty_error_correlation(detailed_test)
    selective_metrics = selective_prediction_metrics(detailed_test)
    bucket_summary = uncertainty_bucket_summary(detailed_test)
    descriptor_summary = descriptor_error_summary(detailed_test)
    worst_predictions = worst_ensemble_predictions(detailed_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    ensemble_csv = output_dir / "ensemble_predictions.csv"
    interval_metrics_csv = output_dir / "interval_metrics.csv"
    selective_metrics_csv = output_dir / "selective_prediction_metrics.csv"
    bucket_summary_csv = output_dir / "uncertainty_bucket_summary.csv"
    descriptor_summary_json = output_dir / "descriptor_error_summary.json"
    worst_predictions_json = output_dir / "worst_predictions.json"
    summary_json = output_dir / "uncertainty_summary.json"
    report_md = output_dir / "uncertainty_report.md"
    detailed_predictions.to_csv(ensemble_csv, index=False)
    pd.DataFrame(interval_results).to_csv(interval_metrics_csv, index=False)
    pd.DataFrame(selective_metrics).to_csv(selective_metrics_csv, index=False)
    pd.DataFrame(bucket_summary).to_csv(bucket_summary_csv, index=False)
    descriptor_summary_json.write_text(
        json.dumps(descriptor_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    worst_predictions_json.write_text(
        json.dumps(worst_predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plot_paths = {
        "uncertainty_vs_error": figures_dir / "uncertainty_vs_error.png",
        "interval_coverage": figures_dir / "interval_coverage.png",
        "selective_prediction_curve": figures_dir / "selective_prediction_curve.png",
        "uncertainty_buckets": figures_dir / "uncertainty_buckets.png",
    }
    plot_uncertainty_vs_error(ensemble_csv, plot_paths["uncertainty_vs_error"])
    plot_interval_coverage(interval_metrics_csv, plot_paths["interval_coverage"])
    plot_selective_prediction_curve(
        selective_metrics_csv,
        plot_paths["selective_prediction_curve"],
    )
    plot_uncertainty_buckets(bucket_summary_csv, plot_paths["uncertainty_buckets"])

    summary = {
        "prediction_paths": [str(path) for path in prediction_paths],
        "ensemble_members": len(prediction_paths),
        "seeds": [
            _seed_from_prediction_path(path, index)
            for index, path in enumerate(prediction_paths)
        ],
        "target_coverages": requested_coverages,
        "detailed_interval_coverage": 0.90,
        "detailed_interval_scale": detailed_scale,
        "ensemble_test_metrics": test_metrics,
        "interval_results": interval_results,
        "uncertainty_error_correlations": correlations,
        "selective_prediction": selective_metrics,
        "uncertainty_buckets": bucket_summary,
        "descriptor_error_summary": descriptor_summary,
        "worst_predictions": worst_predictions,
        "plots": {
            name: path.relative_to(output_dir).as_posix()
            for name, path in plot_paths.items()
        },
        "artifacts": {
            "ensemble_predictions_csv": str(ensemble_csv),
            "interval_metrics_csv": str(interval_metrics_csv),
            "selective_prediction_metrics_csv": str(selective_metrics_csv),
            "uncertainty_bucket_summary_csv": str(bucket_summary_csv),
            "descriptor_error_summary_json": str(descriptor_summary_json),
            "worst_predictions_json": str(worst_predictions_json),
            "uncertainty_summary_json": str(summary_json),
            "uncertainty_report_md": str(report_md),
        },
    }
    write_gnn_uncertainty_report(summary, report_md)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
