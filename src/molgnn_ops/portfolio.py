import json
from pathlib import Path
from typing import Any

from molgnn_ops import __version__

DEFAULT_DEMO_SMILES = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "[U]"]

HEADLINE_RESULTS = {
    "fingerprint_random_forest": {
        "split": "scaffold",
        "seeds": [42, 43, 44],
        "test_rmse_mean": 1.847980833926914,
        "test_rmse_std": 0.02139121055541615,
    },
    "gcn": {
        "split": "scaffold",
        "seeds": [42, 43, 44],
        "test_rmse_mean": 1.339454356739515,
        "test_rmse_std": 0.07384768754669406,
    },
    "gin": {
        "split": "scaffold",
        "seeds": [42, 43, 44],
        "test_rmse_mean": 1.4498563826784234,
        "test_rmse_std": 0.13721279936773365,
    },
    "promoted_fixed_split_gcn": {
        "split": "scaffold",
        "split_seed": 42,
        "model_seed": 43,
        "validation_rmse": 1.3420166774975013,
        "test_rmse": 1.3501974828901644,
        "test_mae": 1.0385344629368838,
        "test_r2": 0.6441207150939929,
    },
}

PROJECT_CAPABILITIES = [
    "ESOL dataset ingestion and scaffold/random splitting",
    "duplicate and conflicting-target audits",
    "RDKit graph featurization and Morgan fingerprints",
    "classical fingerprint baselines",
    "GCN and GIN regression benchmarks",
    "fixed-split ensemble uncertainty validation",
    "validation-based model promotion",
    "FastAPI inference and Streamlit molecule explorer",
    "Docker, Compose, and GitHub Actions workflows",
]


def _read_json(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} file is not valid JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} file must contain a JSON object: {path}")
    return document


def _require_mapping(document: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain object field {key!r}")
    return value


def _require_sequence(document: dict[str, Any], key: str, label: str) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label} must contain list field {key!r}")
    return value


def _require_number(document: dict[str, Any], key: str, label: str) -> float:
    value = document.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must contain numeric field {key!r}")
    return float(value)


def _format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_seeds(seeds: list[Any] | None) -> str:
    if not seeds:
        return "N/A"
    return ", ".join(str(seed) for seed in seeds)


def _table_text(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def _compact_warnings(warnings: list[Any] | None) -> str:
    if not warnings:
        return "None"
    return "<br>".join(_table_text(warning) for warning in warnings)


def _compact_neighbors(neighbors: list[dict[str, Any]] | None) -> str:
    if not neighbors:
        return "None"
    summaries = []
    for neighbor in neighbors[:3]:
        similarity = neighbor.get("tanimoto_similarity")
        similarity_text = _format_float(similarity, digits=3)
        summaries.append(
            f"{neighbor.get('sample_id', 'unknown')} "
            f"{neighbor.get('canonical_smiles', neighbor.get('smiles', 'unknown'))} "
            f"({similarity_text})"
        )
    return "<br>".join(_table_text(summary) for summary in summaries)


def _validate_benchmark_comparison(document: dict[str, Any]) -> None:
    _require_mapping(document, "by_model", "benchmark comparison")
    _require_sequence(document, "seeds", "benchmark comparison")
    for model_name, model_summary in document["by_model"].items():
        if not isinstance(model_summary, dict):
            raise ValueError(f"benchmark comparison model {model_name!r} must be an object")
        _require_number(model_summary, "mean_test_rmse", f"{model_name} summary")
        _require_number(model_summary, "std_test_rmse", f"{model_name} summary")
        _require_number(model_summary, "n_runs", f"{model_name} summary")


def _validate_fixed_split_summary(document: dict[str, Any]) -> None:
    _require_mapping(document, "split_counts", "fixed-split summary")
    _require_sequence(document, "models", "fixed-split summary")
    _require_mapping(document, "duplicate_audit", "fixed-split summary")
    _require_mapping(document, "preparation", "fixed-split summary")
    uncertainty = _require_mapping(document, "uncertainty", "fixed-split summary")
    _require_mapping(uncertainty, "ensemble_test_metrics", "fixed-split uncertainty")
    _require_sequence(uncertainty, "interval_results", "fixed-split uncertainty")
    _require_sequence(uncertainty, "selective_prediction", "fixed-split uncertainty")
    _require_mapping(
        uncertainty,
        "uncertainty_error_correlations",
        "fixed-split uncertainty",
    )


def _validate_uncertainty_summary(document: dict[str, Any]) -> dict[str, Any]:
    uncertainty = document.get("uncertainty", document)
    if not isinstance(uncertainty, dict):
        raise ValueError("uncertainty summary must contain a JSON object")
    _require_mapping(uncertainty, "ensemble_test_metrics", "uncertainty summary")
    _require_sequence(uncertainty, "interval_results", "uncertainty summary")
    _require_sequence(uncertainty, "selective_prediction", "uncertainty summary")
    _require_mapping(uncertainty, "uncertainty_error_correlations", "uncertainty summary")
    return uncertainty


def _validate_manifest(document: dict[str, Any]) -> None:
    for key in ["model_id", "model_type", "dataset_name", "split_seed", "model_seed"]:
        if key not in document:
            raise ValueError(f"promoted manifest must contain field {key!r}")
    _require_mapping(document, "validation_metrics", "promoted manifest")
    _require_mapping(document, "test_metrics", "promoted manifest")


def _benchmark_rows(
    benchmark: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if benchmark is None:
        seeds: list[Any] = []
        split = "scaffold"
    else:
        seeds = benchmark.get("seeds", [])
        split = benchmark.get("split_strategy", "scaffold")

    fingerprint = benchmark.get("fingerprint_baseline") if benchmark is not None else None
    if isinstance(fingerprint, dict):
        rows.append(
            {
                "method": "Fingerprint random forest",
                "split": fingerprint.get("split_strategy", split),
                "seeds": fingerprint.get("seeds", seeds),
                "test_rmse_mean": fingerprint.get("mean_test_rmse"),
                "test_rmse_std": fingerprint.get("std_test_rmse"),
                "notes": "Morgan fingerprint classical baseline.",
            }
        )
    if benchmark is not None:
        for model_name in sorted(benchmark.get("by_model", {})):
            values = benchmark["by_model"][model_name]
            rows.append(
                {
                    "method": model_name.upper(),
                    "split": split,
                    "seeds": seeds,
                    "test_rmse_mean": values.get("mean_test_rmse"),
                    "test_rmse_std": values.get("std_test_rmse"),
                    "notes": "Repeated-seed GNN comparison.",
                }
            )
    if manifest is not None:
        validation = manifest["validation_metrics"].get("rmse")
        rows.append(
            {
                "method": "Promoted fixed-split GCN",
                "split": manifest.get("split_strategy", "scaffold"),
                "seeds": [f"split {manifest['split_seed']}", f"model {manifest['model_seed']}"],
                "test_rmse_mean": manifest["test_metrics"].get("rmse"),
                "test_rmse_std": None,
                "notes": f"Selected by validation RMSE {_format_float(validation)}.",
            }
        )
    return rows


def _write_benchmark_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    unavailable: list[str],
) -> None:
    lines = [
        "# Portfolio Benchmark Summary",
        "",
        "This tracked snapshot records the verified ESOL scaffold-split results without "
        "copying large generated artifacts into Git.",
        "",
        "| Method | Split | Seeds | Test RMSE mean | RMSE std | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"| {row['method']} | {row['split']} | {_format_seeds(row['seeds'])} | "
                f"{_format_float(row['test_rmse_mean'])} | "
                f"{_format_float(row['test_rmse_std'])} | {row['notes']} |"
            )
    else:
        lines.append("| Unavailable | N/A | N/A | N/A | N/A | No benchmark JSON supplied. |")
    lines.extend(
        [
            "",
            "Test metrics are reported after validation-based model selection. They are not "
            "used to choose the promoted checkpoint.",
        ]
    )
    if unavailable:
        lines.extend(["", "## Unavailable Inputs", ""])
        lines.extend(f"- {item}" for item in unavailable)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_uncertainty_markdown(
    path: Path,
    fixed_split: dict[str, Any] | None,
    uncertainty: dict[str, Any] | None,
) -> None:
    lines = [
        "# Portfolio Uncertainty Summary",
        "",
        "The fixed-split ensemble experiment is retained because it is an important "
        "negative result: model disagreement did not rank ESOL prediction errors.",
    ]
    if fixed_split is None or uncertainty is None:
        lines.extend(
            [
                "",
                "No fixed-split or uncertainty summary JSON was supplied. This section is "
                "intentionally marked unavailable rather than filled with guessed values.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "",
            "## Individual Fixed-Split Models",
            "",
            "| Model seed | Validation RMSE | Test RMSE | Test MAE | Test R2 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in fixed_split["models"]:
        validation = model["validation_metrics"]
        test = model["test_metrics"]
        lines.append(
            f"| {model['model_seed']} | {_format_float(validation.get('rmse'))} | "
            f"{_format_float(test.get('rmse'))} | {_format_float(test.get('mae'))} | "
            f"{_format_float(test.get('r2'))} |"
        )

    ensemble = uncertainty["ensemble_test_metrics"]
    lines.extend(
        [
            "",
            "## Ensemble Metrics",
            "",
            f"- Ensemble test RMSE: {_format_float(ensemble.get('rmse'))}",
            f"- Ensemble test MAE: {_format_float(ensemble.get('mae'))}",
            "",
            "## Interval Calibration",
            "",
            "| Nominal coverage | Empirical coverage | Mean interval width |",
            "| ---: | ---: | ---: |",
        ]
    )
    for interval in uncertainty["interval_results"]:
        lines.append(
            f"| {_format_float(interval.get('target_coverage'))} | "
            f"{_format_float(interval.get('empirical_coverage'))} | "
            f"{_format_float(interval.get('mean_interval_width'))} |"
        )

    correlations = uncertainty["uncertainty_error_correlations"]
    pearson = _format_float(correlations.get("pearson"))
    spearman = _format_float(correlations.get("spearman"))
    lines.extend(
        [
            "",
            "## Error Ranking",
            "",
            f"- Pearson uncertainty-error correlation: {pearson}",
            f"- Rank uncertainty-error correlation: {spearman}",
            "",
            "| Retained fraction | Test RMSE | Mean uncertainty |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in uncertainty["selective_prediction"]:
        lines.append(
            f"| {_format_float(row.get('retained_fraction'))} | "
            f"{_format_float(row.get('rmse'))} | "
            f"{_format_float(row.get('mean_uncertainty'))} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Ensemble disagreement was not a useful uncertainty signal for this ESOL setup. "
            "The API and dashboard therefore expose applicability context, not unsupported "
            "confidence estimates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_system_verification_markdown(
    path: Path,
    verification_metadata: dict[str, Any] | None,
) -> None:
    metadata = verification_metadata or {}
    checks = metadata.get("checks", [])
    lines = [
        "# System Verification Summary",
        "",
        f"- Package version: {metadata.get('version', __version__)}",
        f"- Tests: {metadata.get('tests', 'unavailable')}",
        f"- Lint: {metadata.get('lint', 'unavailable')}",
        f"- Docker image: {metadata.get('docker_image', 'unavailable')}",
        f"- Compose: {metadata.get('compose', 'unavailable')}",
        f"- CPU-only behavior: {metadata.get('cpu_only', 'unavailable')}",
        f"- Registry mount design: {metadata.get('registry_mount', 'unavailable')}",
        "",
        "## Service Checks",
        "",
    ]
    if checks:
        lines.extend(f"- {check}" for check in checks)
    else:
        lines.append("- No service checks supplied.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_portfolio_reports(
    output_dir: Path,
    benchmark_comparison_json: Path | None = None,
    fixed_split_summary_json: Path | None = None,
    uncertainty_summary_json: Path | None = None,
    promoted_manifest_json: Path | None = None,
    verification_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate small deterministic portfolio summaries from explicit inputs."""
    benchmark = _read_json(benchmark_comparison_json, "benchmark comparison")
    fixed_split = _read_json(fixed_split_summary_json, "fixed-split summary")
    supplied_uncertainty = _read_json(uncertainty_summary_json, "uncertainty summary")
    manifest = _read_json(promoted_manifest_json, "promoted manifest")

    unavailable: list[str] = []
    if benchmark is None:
        unavailable.append("benchmark comparison JSON was not supplied")
    else:
        _validate_benchmark_comparison(benchmark)
    if fixed_split is None:
        unavailable.append("fixed-split summary JSON was not supplied")
    else:
        _validate_fixed_split_summary(fixed_split)
    if manifest is None:
        unavailable.append("promoted manifest JSON was not supplied")
    else:
        _validate_manifest(manifest)
    if supplied_uncertainty is not None:
        uncertainty = _validate_uncertainty_summary(supplied_uncertainty)
    elif fixed_split is not None:
        uncertainty = fixed_split["uncertainty"]
    else:
        uncertainty = None
        unavailable.append("uncertainty summary JSON was not supplied")

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows = _benchmark_rows(benchmark, manifest)
    summary = {
        "project": "molecular-gnn-property-ops",
        "version": __version__,
        "dataset": (
            benchmark.get("dataset_name")
            if benchmark is not None
            else manifest.get("dataset_name")
            if manifest is not None
            else None
        ),
        "benchmark_rows": benchmark_rows,
        "best_repeated_seed_run": benchmark.get("best_single_run")
        if benchmark is not None
        else None,
        "promoted_model": {
            "model_id": manifest.get("model_id"),
            "model_type": manifest.get("model_type"),
            "split_seed": manifest.get("split_seed"),
            "model_seed": manifest.get("model_seed"),
            "validation_metrics": manifest.get("validation_metrics"),
            "post_selection_test_metrics": manifest.get("test_metrics"),
        }
        if manifest is not None
        else None,
        "data_quality": {
            "dataset_size": fixed_split.get("preparation", {}).get("n_rows"),
            "duplicate_canonical_smiles_groups": fixed_split.get(
                "duplicate_audit", {}
            ).get("duplicate_canonical_smiles_groups"),
            "conflicting_target_groups": fixed_split.get("duplicate_audit", {}).get(
                "duplicate_groups_with_conflicting_targets"
            ),
        }
        if fixed_split is not None
        else None,
        "unavailable_sections": unavailable,
    }

    benchmark_json = output_dir / "benchmark_summary.json"
    benchmark_md = output_dir / "benchmark_summary.md"
    uncertainty_md = output_dir / "uncertainty_summary.md"
    verification_md = output_dir / "system_verification.md"
    benchmark_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_benchmark_markdown(benchmark_md, benchmark_rows, unavailable)
    _write_uncertainty_markdown(uncertainty_md, fixed_split, uncertainty)
    _write_system_verification_markdown(verification_md, verification_metadata)
    return {
        "benchmark_summary_json": str(benchmark_json),
        "benchmark_summary_md": str(benchmark_md),
        "uncertainty_summary_md": str(uncertainty_md),
        "system_verification_md": str(verification_md),
    }


def write_demo_outputs(
    predictions: list[dict[str, Any]],
    context_predictions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    """Write deterministic demo artifacts from already-computed predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    context_path = output_dir / "context_predictions.json"
    summary_path = output_dir / "demo_summary.md"
    predictions_path.write_text(
        json.dumps(predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    context_path.write_text(
        json.dumps(context_predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Molecular Solubility Demo",
        "",
        "This demo records point predictions and training-set applicability context. "
        "It does not report confidence or uncertainty.",
        "",
        "| Input SMILES | Status | Model | Canonical SMILES | Predicted logS | "
        "Predicted mol/L | Applicability warnings | Nearest training molecules |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    context_by_smiles = {item["input_smiles"]: item for item in context_predictions}
    for item in predictions:
        if not item.get("success"):
            lines.append(
                f"| {item.get('input_smiles')} | error | N/A | N/A | N/A | N/A | "
                f"{_table_text(item.get('error'))} | N/A |"
            )
            continue
        prediction = item["prediction"]
        context = context_by_smiles.get(item["input_smiles"], {})
        warnings = "N/A"
        neighbors = "N/A"
        if context.get("success"):
            applicability = context["context"]["applicability"]
            warnings = _compact_warnings(applicability.get("warnings"))
            neighbors = _compact_neighbors(context["context"]["nearest_training_molecules"])
        lines.append(
            f"| {item['input_smiles']} | ok | {prediction['model_id']} | "
            f"{prediction['canonical_smiles']} | "
            f"{_format_float(prediction['predicted_log_solubility'])} | "
            f"{_format_float(prediction['predicted_solubility_mol_per_litre'], digits=6)} | "
            f"{warnings} | {neighbors} |"
        )
    lines.extend(
        [
            "",
            "Applicability warnings are descriptive. Similarity is not confidence, and "
            "experimental validation is required for real decisions.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "predictions_json": str(predictions_path),
        "context_predictions_json": str(context_path),
        "demo_summary_md": str(summary_path),
    }


def generate_demo(
    manifest_path: Path,
    output_dir: Path,
    smiles_values: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, str]:
    """Generate a small model demo without starting a long-running service."""
    from molgnn_ops.inference import (
        load_promoted_model,
        predict_smiles,
        predict_smiles_with_context,
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Promoted model manifest does not exist: {manifest_path}")
    loaded_model = load_promoted_model(manifest_path)
    molecules = smiles_values or DEFAULT_DEMO_SMILES
    predictions: list[dict[str, Any]] = []
    context_predictions: list[dict[str, Any]] = []
    for smiles in molecules:
        try:
            prediction = predict_smiles(smiles, loaded_model)
            predictions.append(
                {
                    "input_smiles": smiles,
                    "success": True,
                    "prediction": prediction,
                    "error": None,
                }
            )
        except (RuntimeError, ValueError) as error:
            predictions.append(
                {
                    "input_smiles": smiles,
                    "success": False,
                    "prediction": None,
                    "error": str(error),
                }
            )
            context_predictions.append(
                {
                    "input_smiles": smiles,
                    "success": False,
                    "context": None,
                    "error": str(error),
                }
            )
            continue

        try:
            context = predict_smiles_with_context(smiles, loaded_model, top_k=top_k)
            context_predictions.append(
                {
                    "input_smiles": smiles,
                    "success": True,
                    "context": context,
                    "error": None,
                }
            )
        except (RuntimeError, ValueError) as error:
            context_predictions.append(
                {
                    "input_smiles": smiles,
                    "success": False,
                    "context": None,
                    "error": str(error),
                }
            )
    return write_demo_outputs(predictions, context_predictions, output_dir)
