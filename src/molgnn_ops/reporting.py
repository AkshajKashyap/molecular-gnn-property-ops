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
