import json
from pathlib import Path

from molgnn_ops.baselines import train_fingerprint_baseline
from molgnn_ops.data_sources import get_dataset_spec
from molgnn_ops.download import download_dataset
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
