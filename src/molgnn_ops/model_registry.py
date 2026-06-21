import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from molgnn_ops.data_sources import get_dataset_spec
from molgnn_ops.featurization import ATOM_FEATURE_CONFIG, BOND_FEATURE_CONFIG
from molgnn_ops.reference_index import build_reference_index


class ModelManifest(BaseModel):
    model_id: str
    model_type: str
    dataset_name: str
    task_type: str
    target_name: str
    checkpoint_path: str
    created_at: datetime
    split_strategy: str
    split_seed: int
    model_seed: int
    atom_feature_dim: int
    edge_feature_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    training_config: dict[str, Any]
    featurization_config: dict[str, Any]
    package_version: str
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)
    reference_index_path: str | None = None
    reference_index_radius: int | None = None
    reference_index_n_bits: int | None = None
    reference_index_size: int | None = None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required model metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_path(run_dir: Path, metrics: dict) -> Path:
    recorded = metrics.get("artifacts", {}).get("model")
    if recorded:
        recorded_path = Path(recorded)
        if recorded_path.is_file():
            return recorded_path
        relative_candidate = run_dir / recorded_path
        if relative_candidate.is_file():
            return relative_candidate
    default = run_dir / "models" / "gnn_regressor.pt"
    if default.is_file():
        return default
    raise FileNotFoundError(f"Model checkpoint not found for candidate run: {run_dir}")


def _candidate_context(run_dir: Path) -> dict:
    candidates = [
        run_dir / "gnn_benchmark_summary.json",
        run_dir / "run_config.json",
        run_dir.parent / "fixed_split_ensemble_summary.json",
    ]
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    return {}


def _candidate_summary(run_dir: Path, metric: str) -> dict:
    metrics_path = run_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    validation_metrics = metrics.get("validation_metrics")
    if not validation_metrics or validation_metrics.get(metric) is None:
        raise ValueError(
            f"Candidate {run_dir} has no validation metric {metric!r}; "
            "test metrics cannot be used for promotion"
        )
    context = _candidate_context(run_dir)
    model_seed = metrics.get("model_seed", metrics.get("seed"))
    if model_seed is None:
        raise ValueError(f"Candidate {run_dir} has no model seed")
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(_checkpoint_path(run_dir, metrics)),
        "model_type": metrics.get("model_name"),
        "dataset_name": metrics.get("dataset_source", context.get("dataset_name")),
        "split_strategy": context.get("split_strategy", "unknown"),
        "split_seed": context.get("split_seed"),
        "model_seed": int(model_seed),
        "validation_metrics": validation_metrics,
        "test_metrics": metrics.get("test_metrics", {}),
        "training_config": metrics.get("hyperparameters", {}),
        "target_normalization": metrics.get("target_normalization", {}),
        "selection_metric": metric,
        "selection_value": float(validation_metrics[metric]),
    }


def select_model_by_validation(
    candidate_run_dirs: list[Path],
    metric: str = "rmse",
    lower_is_better: bool = True,
) -> dict:
    """Rank model candidates exclusively by a validation metric."""
    if not candidate_run_dirs:
        raise ValueError("At least one candidate run directory is required")
    candidates = [_candidate_summary(path, metric) for path in candidate_run_dirs]
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate["selection_value"],
        reverse=not lower_is_better,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "selected_run": ranked[0]["run_dir"],
        "selected_candidate": ranked[0],
        "ranked_candidates": ranked,
    }


def _package_version() -> str:
    try:
        return version("molecular-gnn-property-ops")
    except PackageNotFoundError:
        return "0.1.0"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_selection_report(selection: dict, output_path: Path, model_id: str) -> None:
    selected = selection["selected_candidate"]
    lines = [
        "# Model Promotion Selection Report",
        "",
        f"- Model ID: `{model_id}`",
        f"- Selection metric: validation `{selection['metric']}`",
        f"- Direction: {'lower' if selection['lower_is_better'] else 'higher'} is better",
        f"- Selected model seed: {selected['model_seed']}",
        "",
        "Test metrics were not used for ranking. They are shown only after selection.",
        "",
        "| Rank | Model seed | Validation metric | Test metric |",
        "| ---: | ---: | ---: | ---: |",
    ]
    metric = selection["metric"]
    for candidate in selection["ranked_candidates"]:
        test_value = candidate["test_metrics"].get(metric)
        formatted_test = "N/A" if test_value is None else f"{float(test_value):.6f}"
        lines.append(
            f"| {candidate['rank']} | {candidate['model_seed']} | "
            f"{candidate['selection_value']:.6f} | {formatted_test} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_model(
    candidate_run_dirs: list[Path],
    registry_dir: Path,
    model_id: str,
    metric: str = "rmse",
    prepared_csv: Path | None = None,
    include_reference_index: bool = True,
) -> ModelManifest:
    """Select by validation performance and create a self-contained model package."""
    if not model_id.strip():
        raise ValueError("model_id must not be blank")
    selection = select_model_by_validation(candidate_run_dirs, metric=metric)
    selected = selection["selected_candidate"]

    import torch

    source_checkpoint = Path(selected["checkpoint_path"])
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyperparameters", selected["training_config"])
    required_checkpoint_keys = {
        "model_name",
        "model_state_dict",
        "input_dim",
        "target_mean",
        "target_std",
    }
    missing_checkpoint_keys = sorted(required_checkpoint_keys.difference(checkpoint))
    if missing_checkpoint_keys:
        raise ValueError(
            "Checkpoint is missing required inference fields: "
            f"{', '.join(missing_checkpoint_keys)}"
        )

    model_dir = registry_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    destination_checkpoint = model_dir / "checkpoint.pt"
    shutil.copy2(source_checkpoint, destination_checkpoint)
    registry_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = str(selected["dataset_name"] or "unknown")
    try:
        dataset_spec = get_dataset_spec(dataset_name)
        target_name = dataset_spec.target_col
        task_type = dataset_spec.task_type
    except ValueError:
        target_name = "target"
        task_type = "regression"
    split_seed = selected["split_seed"]
    if split_seed is None:
        raise ValueError(f"Candidate {selected['run_dir']} has no split_seed metadata")

    atom_feature_dim = len(ATOM_FEATURE_CONFIG.symbols) + 3 + len(
        ATOM_FEATURE_CONFIG.hybridizations
    ) + 2
    edge_feature_dim = len(BOND_FEATURE_CONFIG.bond_types) + 2
    featurization_config = {
        "atom_symbols": list(ATOM_FEATURE_CONFIG.symbols),
        "hybridizations": list(ATOM_FEATURE_CONFIG.hybridizations),
        "bond_types": list(BOND_FEATURE_CONFIG.bond_types),
        "atom_feature_dim": atom_feature_dim,
        "edge_feature_dim": edge_feature_dim,
    }
    if int(checkpoint["input_dim"]) != atom_feature_dim:
        raise ValueError(
            f"Checkpoint input dimension {checkpoint['input_dim']} does not match "
            f"current atom feature dimension {atom_feature_dim}"
        )

    relative_checkpoint = destination_checkpoint.relative_to(registry_dir).as_posix()
    reference_summary = None
    reference_index_path = None
    if include_reference_index:
        resolved_prepared_csv = prepared_csv or Path(selected["run_dir"]).parent / "prepared.csv"
        if not resolved_prepared_csv.is_file():
            raise FileNotFoundError(
                "Prepared CSV is required to build the promoted reference index: "
                f"{resolved_prepared_csv}"
            )
        reference_artifact = model_dir / "reference_index.npz"
        reference_summary = build_reference_index(
            resolved_prepared_csv,
            reference_artifact,
            split="train",
        )
        reference_index_path = reference_artifact.relative_to(registry_dir).as_posix()
    manifest = ModelManifest(
        model_id=model_id,
        model_type=str(checkpoint["model_name"]),
        dataset_name=dataset_name,
        task_type=task_type,
        target_name=target_name,
        checkpoint_path=relative_checkpoint,
        created_at=datetime.now(UTC),
        split_strategy=str(selected["split_strategy"]),
        split_seed=int(split_seed),
        model_seed=int(selected["model_seed"]),
        atom_feature_dim=atom_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        dropout=float(hyperparameters["dropout"]),
        validation_metrics=selected["validation_metrics"],
        test_metrics=selected["test_metrics"],
        training_config={
            **selected["training_config"],
            "target_normalization": {
                "mean": float(checkpoint["target_mean"]),
                "std": float(checkpoint["target_std"]),
            },
        },
        featurization_config=featurization_config,
        package_version=_package_version(),
        git_commit=_git_commit(),
        notes=[
            "Selected using validation performance only.",
            "Test metrics are post-selection reporting and were not used for promotion.",
            "Ensemble disagreement was not a useful uncertainty signal and is not exposed.",
        ],
        reference_index_path=reference_index_path,
        reference_index_radius=(
            reference_summary["radius"] if reference_summary is not None else None
        ),
        reference_index_n_bits=(
            reference_summary["n_bits"] if reference_summary is not None else None
        ),
        reference_index_size=(
            reference_summary["n_reference_molecules"]
            if reference_summary is not None
            else None
        ),
    )
    manifest_path = registry_dir / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    featurization_path = registry_dir / "featurization_config.json"
    featurization_path.write_text(
        json.dumps(featurization_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ranking_path = registry_dir / "candidate_ranking.csv"
    with ranking_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["rank", "model_seed", "validation_metric", "test_metric"],
        )
        writer.writeheader()
        for candidate in selection["ranked_candidates"]:
            writer.writerow(
                {
                    "rank": candidate["rank"],
                    "model_seed": candidate["model_seed"],
                    "validation_metric": candidate["selection_value"],
                    "test_metric": candidate["test_metrics"].get(metric),
                }
            )
    _write_selection_report(
        selection,
        registry_dir / "selection_report.md",
        model_id,
    )
    return manifest
