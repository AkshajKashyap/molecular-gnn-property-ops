import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgnn_ops.workflows import run_fingerprint_benchmark


def _write_comparison_report(summary: dict, output_path: Path) -> None:
    lines = [
        "# Fingerprint Split Comparison",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        f"- Split strategies: {', '.join(summary['split_strategies'])}",
        "",
        "## Aggregate Results",
        "",
        "| Split strategy | Metric | Runs | Validation mean | Test mean | Test std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy, values in summary["by_split_strategy"].items():
        lines.append(
            f"| {strategy} | {values['key_metric']} | {values['n_runs']} | "
            f"{values['mean_validation_metric']:.4f} | {values['mean_test_metric']:.4f} | "
            f"{values['std_test_metric']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Individual Runs",
            "",
            "| Split strategy | Seed | Best model | Metric | Validation | Test |",
            "| --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for run in summary["runs"]:
        lines.append(
            f"| {run['split_strategy']} | {run['seed']} | {run['best_model']} | "
            f"{run['key_metric']} | {run['validation_metric']:.4f} | "
            f"{run['test_metric']:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split_comparison(
    dataset_name: str,
    output_dir: Path,
    seeds: list[int],
    split_strategies: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    overwrite: bool = False,
) -> dict:
    """Run and summarize deterministic fingerprint benchmarks across split configurations."""
    if not seeds:
        raise ValueError("At least one seed is required")
    if not split_strategies:
        raise ValueError("At least one split strategy is required")
    invalid_strategies = sorted(set(split_strategies).difference({"random", "scaffold"}))
    if invalid_strategies:
        invalid = ", ".join(invalid_strategies)
        raise ValueError(f"Unsupported split strategies: {invalid}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for split_strategy in split_strategies:
        benchmark_root = output_dir / "runs" / split_strategy
        for seed in seeds:
            result = run_fingerprint_benchmark(
                dataset_name,
                benchmark_root,
                split_strategy=split_strategy,
                seed=seed,
                radius=radius,
                n_bits=n_bits,
                overwrite=overwrite,
            )
            runs.append(
                {
                    "dataset_name": result["dataset_name"],
                    "split_strategy": result["split_strategy"],
                    "seed": result["seed"],
                    "best_model": result["best_model"],
                    "key_metric": result["key_metric"],
                    "validation_metric": result["validation_metric"],
                    "test_metric": result["test_metric"],
                    "metrics_json": result["metrics_json"],
                    "report_md": result["report_md"],
                }
            )

    metrics_csv = output_dir / "comparison_metrics.csv"
    pd.DataFrame(runs).to_csv(metrics_csv, index=False)
    by_split_strategy = {}
    for split_strategy in split_strategies:
        strategy_runs = [run for run in runs if run["split_strategy"] == split_strategy]
        validation_values = np.asarray(
            [run["validation_metric"] for run in strategy_runs], dtype=float
        )
        test_values = np.asarray([run["test_metric"] for run in strategy_runs], dtype=float)
        by_split_strategy[split_strategy] = {
            "key_metric": strategy_runs[0]["key_metric"],
            "n_runs": len(strategy_runs),
            "mean_validation_metric": float(np.mean(validation_values)),
            "mean_test_metric": float(np.mean(test_values)),
            "std_test_metric": float(np.std(test_values)),
            "min_test_metric": float(np.min(test_values)),
            "max_test_metric": float(np.max(test_values)),
        }

    summary_json = output_dir / "comparison_summary.json"
    report_md = output_dir / "comparison_report.md"
    summary = {
        "dataset_name": dataset_name,
        "seeds": seeds,
        "split_strategies": split_strategies,
        "radius": radius,
        "n_bits": n_bits,
        "runs": runs,
        "by_split_strategy": by_split_strategy,
        "comparison_metrics_csv": str(metrics_csv),
        "comparison_summary_json": str(summary_json),
        "comparison_report_md": str(report_md),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_comparison_report(summary, report_md)
    return summary
