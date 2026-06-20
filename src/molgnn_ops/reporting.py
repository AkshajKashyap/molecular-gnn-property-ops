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
        f"- Seed: {metrics['seed']}",
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
