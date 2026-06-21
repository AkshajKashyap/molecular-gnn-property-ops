from pathlib import Path


def _format_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_report(metrics: dict, output_path: Path, title: str) -> None:
    """Write a compact Markdown summary of model selection and final evaluation."""
    task_type = metrics.get("task_type", "unknown")
    best_model = metrics.get("best_model", "unknown")
    selection_metric = metrics.get("selection_metric", "unknown")
    model_results = metrics.get("models", {})
    test_metrics = metrics.get("test_metrics", {})

    lines = [
        f"# {title}",
        "",
        f"- Task type: `{task_type}`",
        f"- Selected model: `{best_model}`",
        f"- Selection metric: `{selection_metric}`",
        "",
        "## Validation Metrics",
        "",
    ]

    validation_metric_names = sorted(
        {
            metric_name
            for result in model_results.values()
            for metric_name in result.get("validation", {})
        }
    )
    if model_results and validation_metric_names:
        lines.append("| Model | " + " | ".join(validation_metric_names) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in validation_metric_names) + " |")
        for model_name, result in model_results.items():
            validation = result.get("validation", {})
            values = [_format_metric(validation.get(name)) for name in validation_metric_names]
            lines.append(f"| {model_name} | " + " | ".join(values) + " |")
    else:
        lines.append("No validation metrics available.")

    lines.extend(["", "## Test Metrics", ""])
    if test_metrics:
        for metric_name, value in test_metrics.items():
            lines.append(f"- {metric_name}: {_format_metric(value)}")
    else:
        lines.append("No test metrics available.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format_metric(value) for value in row) + " |"
        for row in rows
    )
    return lines


def write_diagnostics_report(
    diagnostics: dict,
    output_path: Path,
    title: str = "Benchmark Diagnostics Report",
) -> None:
    """Write an inspectable Markdown report from benchmark diagnostics."""
    target_summaries = diagnostics.get("target_distribution", {})
    prediction_summaries = diagnostics.get("prediction_errors", {})
    worst_rows = diagnostics.get("worst_test_predictions", [])
    scaffolds = diagnostics.get("scaffold_distribution", {})
    similarity = diagnostics.get("train_test_similarity", {})
    plots = diagnostics.get("plots", {})

    lines = [f"# {title}", "", "## Target Distribution by Split", ""]
    target_metrics = ["count", "mean", "std", "min", "max", "median", "q25", "q75"]
    lines.extend(
        _markdown_table(
            ["Split", *target_metrics],
            [
                [split_name, *(summary.get(metric) for metric in target_metrics)]
                for split_name, summary in target_summaries.items()
            ],
        )
    )

    lines.extend(["", "## Prediction Error Summary", ""])
    error_metrics = ["n", "mae", "rmse", "mean_error", "median_abs_error", "max_abs_error"]
    lines.extend(
        _markdown_table(
            ["Split", *error_metrics],
            [
                [split_name, *(summary.get(metric) for metric in error_metrics)]
                for split_name, summary in prediction_summaries.items()
            ],
        )
    )

    lines.extend(["", "## Worst Test Predictions", ""])
    worst_columns = ["smiles", "y_true", "y_pred", "error", "absolute_error"]
    lines.extend(
        _markdown_table(
            ["SMILES", "Actual", "Predicted", "Error", "Absolute error"],
            [
                [f"`{row.get('smiles')}`", *(row.get(column) for column in worst_columns[1:])]
                for row in worst_rows
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Scaffold Distribution",
            "",
            f"- Unique scaffolds: {_format_metric(scaffolds.get('n_unique_scaffolds'))}",
            "- Largest scaffold group: "
            f"{_format_metric(scaffolds.get('largest_scaffold_group_size'))}",
            "- Median scaffold group size: "
            f"{_format_metric(scaffolds.get('median_scaffold_group_size'))}",
            f"- Singleton scaffolds: {_format_metric(scaffolds.get('n_singleton_scaffolds'))}",
            "",
        ]
    )
    top_scaffolds = scaffolds.get("top_10_scaffold_groups", [])
    lines.extend(
        _markdown_table(
            ["Scaffold", "Size"],
            [[f"`{row.get('scaffold')}`", row.get("size")] for row in top_scaffolds],
        )
    )

    lines.extend(["", "## Train-Test Similarity", ""])
    for metric_name, value in similarity.items():
        lines.append(f"- {metric_name}: {_format_metric(value)}")

    lines.extend(["", "## Figures", ""])
    for plot_name, relative_path in plots.items():
        label = str(plot_name).replace("_", " ").title()
        lines.append(f"![{label}]({relative_path})")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These diagnostics describe this split and model run; they do not establish causal "
            "relationships. Differences in target distributions, scaffold frequencies, or "
            "train-test similarity may help explain performance, but should be checked across "
            "multiple seeds. Lower test-to-train similarity can make a split more challenging, "
            "while individual large errors may reflect both model limitations and unusual "
            "molecules.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gnn_report(metrics: dict, output_path: Path) -> None:
    """Write a compact report for one graph neural network regression run."""
    lines = [
        f"# {str(metrics['model_name']).upper()} Molecular Graph Baseline",
        "",
        f"- Dataset source: `{metrics['dataset_source']}`",
        f"- Device: `{metrics['device']}`",
        f"- Model seed: {metrics.get('model_seed', metrics['seed'])}",
        f"- Best epoch: {metrics['best_epoch']}",
        "",
        "## Hyperparameters",
        "",
    ]
    for name, value in metrics["hyperparameters"].items():
        lines.append(f"- {name}: {_format_metric(value)}")

    lines.extend(["", "## Validation Metrics", ""])
    for name, value in metrics["validation_metrics"].items():
        lines.append(f"- {name}: {_format_metric(value)}")
    lines.extend(["", "## Test Metrics", ""])
    for name, value in metrics["test_metrics"].items():
        lines.append(f"- {name}: {_format_metric(value)}")

    lines.extend(["", "## Fingerprint Baseline Comparison", ""])
    comparison = metrics.get("fingerprint_comparison")
    if comparison is None:
        lines.append("No nearby fingerprint baseline metrics were available for comparison.")
    else:
        difference = comparison["rmse_difference"]
        direction = "lower" if difference < 0 else "higher"
        lines.extend(
            [
                "- Fingerprint test RMSE: "
                f"{_format_metric(comparison['fingerprint_test_rmse'])}",
                f"- GNN test RMSE: {_format_metric(comparison['gnn_test_rmse'])}",
                f"- GNN RMSE is {abs(difference):.4f} {direction} than the fingerprint baseline.",
            ]
        )
    lines.extend(
        [
            "",
            "The comparison is descriptive for this split and seed. Lower RMSE is better; "
            "model conclusions should be checked across multiple seeds and hyperparameters.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gnn_uncertainty_report(
    summary: dict,
    output_path: Path,
    title: str = "GNN Ensemble Uncertainty Report",
) -> None:
    """Write an uncertainty and molecular error-analysis report."""
    test_metrics = summary["ensemble_test_metrics"]
    correlations = summary["uncertainty_error_correlations"]
    lines = [
        f"# {title}",
        "",
        f"- Ensemble members: {summary['ensemble_members']}",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "- Interval calibration split: `validation`",
        "- Interval evaluation split: `test`",
        "",
        "## Ensemble Test Metrics",
        "",
        f"- RMSE: {_format_metric(test_metrics.get('rmse'))}",
        f"- MAE: {_format_metric(test_metrics.get('mae'))}",
        f"- R2: {_format_metric(test_metrics.get('r2'))}",
        "",
        "## Validation-Calibrated Prediction Intervals",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "Target coverage",
                "Interval scale",
                "Test coverage",
                "Mean width",
                "Median width",
            ],
            [
                [
                    row["target_coverage"],
                    row["interval_scale"],
                    row["empirical_coverage"],
                    row["mean_interval_width"],
                    row["median_interval_width"],
                ]
                for row in summary["interval_results"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "These are ensemble-disagreement intervals scaled on validation residuals. They "
            "are not guaranteed confidence intervals.",
            "",
            "## Uncertainty-Error Correlation",
            "",
            f"- Pearson: {_format_metric(correlations.get('pearson'))}",
            f"- Spearman-style rank: {_format_metric(correlations.get('spearman'))}",
            "",
            "## Selective Prediction",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Retained fraction", "N", "RMSE", "MAE", "Mean uncertainty"],
            [
                [
                    row["retained_fraction"],
                    row["n_retained"],
                    row["rmse"],
                    row["mae"],
                    row["mean_uncertainty"],
                ]
                for row in summary["selective_prediction"]
            ],
        )
    )
    lines.extend(["", "## Uncertainty Buckets", ""])
    lines.extend(
        _markdown_table(
            ["Bucket", "N", "Mean uncertainty", "RMSE", "MAE", "Coverage"],
            [
                [
                    row["bucket"],
                    row["n"],
                    row["mean_uncertainty"],
                    row["rmse"],
                    row["mae"],
                    row["empirical_coverage"],
                ]
                for row in summary["uncertainty_buckets"]
            ],
        )
    )
    lines.extend(["", "## Worst Ensemble Predictions", ""])
    lines.extend(
        _markdown_table(
            [
                "SMILES",
                "Actual",
                "Mean",
                "Std",
                "Absolute error",
                "Lower",
                "Upper",
                "Covered",
                "Mol. weight",
                "Heavy atoms",
                "Rings",
                "Rotatable bonds",
                "Heteroatoms",
            ],
            [
                [
                    f"`{row['smiles']}`",
                    row["y_true"],
                    row["ensemble_mean"],
                    row["ensemble_std"],
                    row["absolute_error"],
                    row["interval_lower"],
                    row["interval_upper"],
                    row["covered"],
                    row["molecular_weight"],
                    row["heavy_atom_count"],
                    row["ring_count"],
                    row["rotatable_bond_count"],
                    row["heteroatom_count"],
                ]
                for row in summary["worst_predictions"]
            ],
        )
    )

    lines.extend(["", "## Descriptor-Based Error Summary", ""])
    for descriptor_name, groups in summary["descriptor_error_summary"].items():
        lines.extend([f"### {descriptor_name.replace('_', ' ').title()}", ""])
        lines.extend(
            _markdown_table(
                ["Quantile group", "N", "RMSE", "MAE", "Mean uncertainty", "Coverage"],
                [
                    [
                        group["group"],
                        group["n"],
                        group["rmse"],
                        group["mae"],
                        group["mean_uncertainty"],
                        group["empirical_coverage"],
                    ]
                    for group in groups
                ],
            )
        )
        lines.append("")
    lines.extend(
        [
            "Descriptor groups are descriptive associations and do not establish causal "
            "relationships.",
            "",
            "## Figures",
            "",
        ]
    )
    for plot_name, plot_path in summary["plots"].items():
        label = plot_name.replace("_", " ").title()
        lines.append(f"![{label}]({plot_path})")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Ensemble disagreement captures only part of predictive uncertainty.",
            "- Three models are a small ensemble.",
            "- Validation-calibrated intervals may not maintain nominal coverage under "
            "distribution shift.",
            "- ESOL is a small dataset.",
            "- The scaffold split is intentionally difficult.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixed_split_ensemble_report(summary: dict, output_path: Path) -> None:
    """Write a compact fixed-split ensemble integrity and performance report."""
    audit = summary["duplicate_audit"]
    uncertainty = summary["uncertainty"]
    lines = [
        "# Fixed-Split GNN Ensemble Report",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Split strategy: `{summary['split_strategy']}`",
        f"- Split seed: {summary['split_seed']}",
        f"- Model seeds: {', '.join(str(seed) for seed in summary['model_seeds'])}",
        "",
        "## Split Counts",
        "",
        f"- Train: {summary['split_counts']['train']}",
        f"- Validation: {summary['split_counts']['val']}",
        f"- Test: {summary['split_counts']['test']}",
        "",
        "## Duplicate Audit",
        "",
        f"- Duplicate canonical-SMILES groups: "
        f"{audit['duplicate_canonical_smiles_groups']}",
        f"- Rows in duplicate groups: {audit['rows_in_duplicate_groups']}",
        f"- Identical-target groups: {audit['duplicate_groups_with_identical_targets']}",
        f"- Conflicting-target groups: {audit['duplicate_groups_with_conflicting_targets']}",
        "",
        "Duplicate measurements are retained as distinct sample IDs; no targets are averaged "
        "or deleted.",
        "",
        "## Individual Models",
        "",
        "| Model seed | Best epoch | Validation RMSE | Test RMSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        lines.append(
            f"| {model['model_seed']} | {model['best_epoch']} | "
            f"{_format_metric(model['validation_metrics'].get('rmse'))} | "
            f"{_format_metric(model['test_metrics'].get('rmse'))} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble Uncertainty",
            "",
            f"- Test RMSE: {_format_metric(uncertainty['ensemble_test_metrics']['rmse'])}",
            f"- Test MAE: {_format_metric(uncertainty['ensemble_test_metrics']['mae'])}",
            "",
            "| Nominal coverage | Empirical coverage | Mean interval width |",
            "| ---: | ---: | ---: |",
        ]
    )
    for result in uncertainty["interval_results"]:
        lines.append(
            f"| {_format_metric(result['target_coverage'])} | "
            f"{_format_metric(result['empirical_coverage'])} | "
            f"{_format_metric(result['mean_interval_width'])} |"
        )
    correlations = uncertainty["uncertainty_error_correlations"]
    lines.extend(
        [
            "",
            f"- Pearson uncertainty-error correlation: "
            f"{_format_metric(correlations['pearson'])}",
            f"- Rank uncertainty-error correlation: "
            f"{_format_metric(correlations['spearman'])}",
            "",
            "All interval scales were fit on validation predictions and evaluated on test "
            "predictions from the same immutable dataset partition.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
