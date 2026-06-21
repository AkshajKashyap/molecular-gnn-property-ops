import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgnn_ops.workflows import run_gnn_benchmark

SUPPORTED_MODELS = {"gcn", "gin"}
METRIC_NAMES = ("rmse", "mae", "r2")


def _validate_inputs(model_names: list[str], seeds: list[int]) -> list[str]:
    normalized_models = [name.strip().lower() for name in model_names if name.strip()]
    if not normalized_models:
        raise ValueError("At least one GNN model name is required")
    invalid_models = sorted(set(normalized_models).difference(SUPPORTED_MODELS))
    if invalid_models:
        raise ValueError(
            "Unsupported GNN model(s): "
            f"{', '.join(invalid_models)}; choose from gcn and gin"
        )
    if len(set(normalized_models)) != len(normalized_models):
        raise ValueError("GNN model names must be unique")
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return normalized_models


def _read_metrics(run_summary: dict) -> tuple[dict, dict]:
    validation = run_summary.get("validation_metrics")
    test = run_summary.get("test_metrics")
    if validation is not None and test is not None:
        return validation, test

    metrics_path = Path(run_summary["metrics_json"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics["validation_metrics"], metrics["test_metrics"]


def _summarize_models(rows: list[dict], model_names: list[str]) -> dict[str, dict]:
    by_model: dict[str, dict] = {}
    for model_name in model_names:
        model_rows = [row for row in rows if row["model_name"] == model_name]
        values: dict[str, int | float | None] = {"n_runs": len(model_rows)}
        for metric_name in METRIC_NAMES:
            metric_values = [
                float(row[f"test_{metric_name}"])
                for row in model_rows
                if row[f"test_{metric_name}"] is not None
            ]
            values[f"mean_test_{metric_name}"] = (
                float(np.mean(metric_values)) if metric_values else None
            )
            values[f"std_test_{metric_name}"] = (
                float(np.std(metric_values, ddof=0)) if metric_values else None
            )
        by_model[model_name] = values
    return by_model


def _find_fingerprint_baseline(output_dir: Path, split_strategy: str) -> dict | None:
    split_summary_candidates = [
        output_dir.parent / "split_comparison" / "comparison_summary.json",
        output_dir / "split_comparison" / "comparison_summary.json",
    ]
    for path in split_summary_candidates:
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        aggregate = summary.get("by_split_strategy", {}).get(split_strategy)
        if aggregate is not None:
            return {
                "source": str(path),
                "scope": "repeated_seed_split_comparison",
                "split_strategy": split_strategy,
                "n_runs": aggregate.get("n_runs"),
                "mean_test_rmse": aggregate.get("mean_test_metric"),
                "std_test_rmse": aggregate.get("std_test_metric"),
            }

    single_run_candidates = [
        output_dir.parent / "seed_42" / "baseline" / "metrics.json",
        output_dir / "baseline" / "metrics.json",
    ]
    for path in single_run_candidates:
        if path.is_file():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            return {
                "source": str(path),
                "scope": "single_run",
                "split_strategy": split_strategy,
                "seed": metrics.get("seed"),
                "best_model": metrics.get("best_model"),
                "test_metrics": metrics.get("test_metrics", {}),
            }
    return None


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _write_report(summary: dict, output_path: Path) -> None:
    lines = [
        "# Repeated-Seed GNN Comparison",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Split strategy: `{summary['split_strategy']}`",
        f"- Split seed: {summary['split_seed']}",
        f"- Model seeds: {', '.join(str(seed) for seed in summary['model_seeds'])}",
        f"- Epoch limit: {summary['epochs']}",
        "",
        "## Fingerprint Baseline Results",
        "",
    ]
    fingerprint = summary["fingerprint_baseline"]
    if fingerprint is None:
        lines.append("No nearby fingerprint benchmark metrics were found.")
    elif fingerprint["scope"] == "repeated_seed_split_comparison":
        lines.extend(
            [
                f"Source: `{fingerprint['source']}`",
                "",
                "| Runs | Mean test RMSE | Std test RMSE |",
                "| ---: | ---: | ---: |",
                f"| {fingerprint['n_runs']} | "
                f"{_format_metric(fingerprint['mean_test_rmse'])} | "
                f"{_format_metric(fingerprint['std_test_rmse'])} |",
            ]
        )
    else:
        test_metrics = fingerprint["test_metrics"]
        lines.extend(
            [
                f"Source: `{fingerprint['source']}`",
                "",
                "Only a single nearby fingerprint run was available.",
                "",
                f"- Model: `{fingerprint.get('best_model')}`",
                f"- Test RMSE: {_format_metric(test_metrics.get('rmse'))}",
                f"- Test MAE: {_format_metric(test_metrics.get('mae'))}",
                f"- Test R2: {_format_metric(test_metrics.get('r2'))}",
            ]
        )

    for model_name in summary["model_names"]:
        model_summary = summary["by_model"][model_name]
        lines.extend(
            [
                "",
                f"## {model_name.upper()} Results",
                "",
                "| Runs | Mean test RMSE | Std test RMSE | Mean test MAE | "
                "Std test MAE | Mean test R2 | Std test R2 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {model_summary['n_runs']} | "
                f"{_format_metric(model_summary['mean_test_rmse'])} | "
                f"{_format_metric(model_summary['std_test_rmse'])} | "
                f"{_format_metric(model_summary['mean_test_mae'])} | "
                f"{_format_metric(model_summary['std_test_mae'])} | "
                f"{_format_metric(model_summary['mean_test_r2'])} | "
                f"{_format_metric(model_summary['std_test_r2'])} |",
            ]
        )

    best_run = summary["best_single_run"]
    lines.extend(
        [
            "",
            "## Selection Summary",
            "",
            f"- Best mean model: `{summary['best_mean_model']}`",
            f"- Best single run: `{best_run['model_name']}` seed {best_run['seed']} "
            f"(test RMSE {_format_metric(best_run['test_rmse'])})",
            "",
            "Model rankings above retain weak or unstable runs; no results are filtered.",
            "",
            "## Figures",
            "",
            "![Test RMSE by model](figures/gnn_rmse_by_model.png)",
            "",
            "![Test RMSE by seed](figures/gnn_test_rmse_by_seed.png)",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_gnn_comparison(
    dataset_name: str,
    output_dir: Path,
    model_names: list[str],
    seeds: list[int],
    split_strategy: str = "scaffold",
    epochs: int = 50,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    overwrite: bool = False,
    split_seed: int = 42,
) -> dict:
    """Run repeated GNN benchmarks and aggregate their held-out metrics."""
    normalized_models = _validate_inputs(model_names, seeds)
    if split_strategy not in {"random", "scaffold"}:
        raise ValueError("split_strategy must be 'random' or 'scaffold'")
    if epochs <= 0 or hidden_dim <= 0 or num_layers <= 0:
        raise ValueError("epochs, hidden_dim, and num_layers must be greater than 0")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in normalized_models:
        for seed in seeds:
            run_dir = output_dir / model_name / f"model_seed_{seed}"
            run_summary = run_gnn_benchmark(
                dataset_name,
                run_dir,
                split_strategy=split_strategy,
                model_name=model_name,
                seed=seed,
                split_seed=split_seed,
                model_seed=seed,
                epochs=epochs,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                overwrite=overwrite,
            )
            validation_metrics, test_metrics = _read_metrics(run_summary)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "seed": seed,
                    "split_seed": split_seed,
                    "model_seed": seed,
                    "split_strategy": split_strategy,
                    "best_epoch": run_summary["best_epoch"],
                    **{
                        f"val_{name}": validation_metrics.get(name)
                        for name in METRIC_NAMES
                    },
                    **{
                        f"test_{name}": test_metrics.get(name)
                        for name in METRIC_NAMES
                    },
                    "metrics_json": run_summary["metrics_json"],
                    "report_md": run_summary["report_md"],
                }
            )

    metrics_csv = output_dir / "gnn_comparison_metrics.csv"
    summary_json = output_dir / "gnn_comparison_summary.json"
    report_md = output_dir / "gnn_comparison_report.md"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    by_model = _summarize_models(rows, normalized_models)
    best_single_run = min(rows, key=lambda row: float(row["test_rmse"]))
    best_mean_model = min(
        normalized_models,
        key=lambda name: float(by_model[name]["mean_test_rmse"]),
    )
    summary = {
        "dataset_name": dataset_name,
        "split_strategy": split_strategy,
        "model_names": normalized_models,
        "seeds": seeds,
        "split_seed": split_seed,
        "model_seeds": seeds,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "runs": rows,
        "by_model": by_model,
        "best_single_run": best_single_run,
        "best_mean_model": best_mean_model,
        "fingerprint_baseline": _find_fingerprint_baseline(
            output_dir,
            split_strategy,
        ),
        "comparison_metrics_csv": str(metrics_csv),
        "comparison_summary_json": str(summary_json),
        "comparison_report_md": str(report_md),
        "figures": {
            "rmse_by_model": str(output_dir / "figures" / "gnn_rmse_by_model.png"),
            "metric_by_seed": str(
                output_dir / "figures" / "gnn_test_rmse_by_seed.png"
            ),
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, report_md)

    from molgnn_ops.plots import plot_gnn_metric_by_seed, plot_gnn_rmse_by_model

    plot_gnn_rmse_by_model(metrics_csv, Path(summary["figures"]["rmse_by_model"]))
    plot_gnn_metric_by_seed(
        metrics_csv,
        Path(summary["figures"]["metric_by_seed"]),
    )
    return summary
