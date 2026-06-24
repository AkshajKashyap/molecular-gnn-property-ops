import json
from pathlib import Path

import pytest

from molgnn_ops.portfolio import generate_portfolio_reports, write_demo_outputs


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _benchmark_payload() -> dict:
    return {
        "dataset_name": "esol",
        "split_strategy": "scaffold",
        "seeds": [1, 2],
        "fingerprint_baseline": {
            "split_strategy": "scaffold",
            "mean_test_rmse": 2.0,
            "std_test_rmse": 0.1,
        },
        "by_model": {
            "gcn": {"mean_test_rmse": 1.0, "std_test_rmse": 0.2, "n_runs": 2},
            "gin": {"mean_test_rmse": 1.4, "std_test_rmse": 0.3, "n_runs": 2},
        },
        "best_single_run": {"model_name": "gcn", "seed": 2, "test_rmse": 0.9},
    }


def _fixed_split_payload() -> dict:
    uncertainty = {
        "ensemble_test_metrics": {"rmse": 1.2, "mae": 0.8},
        "interval_results": [
            {
                "target_coverage": 0.9,
                "empirical_coverage": 0.8,
                "mean_interval_width": 5.0,
            }
        ],
        "selective_prediction": [
            {"retained_fraction": 0.5, "rmse": 1.3, "mean_uncertainty": 0.2},
            {"retained_fraction": 1.0, "rmse": 1.2, "mean_uncertainty": 0.4},
        ],
        "uncertainty_error_correlations": {"pearson": -0.01, "spearman": -0.02},
    }
    return {
        "preparation": {"n_rows": 4},
        "split_counts": {"train": 2, "val": 1, "test": 1},
        "duplicate_audit": {
            "duplicate_canonical_smiles_groups": 1,
            "duplicate_groups_with_conflicting_targets": 1,
        },
        "models": [
            {
                "model_seed": 11,
                "validation_metrics": {"rmse": 1.1},
                "test_metrics": {"rmse": 1.2, "mae": 0.9, "r2": 0.3},
            }
        ],
        "uncertainty": uncertainty,
    }


def _manifest_payload() -> dict:
    return {
        "model_id": "demo-gcn-v1",
        "model_type": "gcn",
        "dataset_name": "esol",
        "split_strategy": "scaffold",
        "split_seed": 42,
        "model_seed": 11,
        "validation_metrics": {"rmse": 1.1},
        "test_metrics": {"rmse": 1.2, "mae": 0.9, "r2": 0.3},
    }


def test_generate_portfolio_reports_is_deterministic(tmp_path: Path) -> None:
    benchmark = _write_json(tmp_path / "benchmark.json", _benchmark_payload())
    fixed = _write_json(tmp_path / "fixed.json", _fixed_split_payload())
    manifest = _write_json(tmp_path / "manifest.json", _manifest_payload())

    first = generate_portfolio_reports(
        tmp_path / "reports",
        benchmark_comparison_json=benchmark,
        fixed_split_summary_json=fixed,
        promoted_manifest_json=manifest,
        verification_metadata={"tests": "ok", "lint": "ok"},
    )
    first_contents = {
        path: Path(path).read_text(encoding="utf-8") for path in first.values()
    }
    second = generate_portfolio_reports(
        tmp_path / "reports",
        benchmark_comparison_json=benchmark,
        fixed_split_summary_json=fixed,
        promoted_manifest_json=manifest,
        verification_metadata={"tests": "ok", "lint": "ok"},
    )
    second_contents = {
        path: Path(path).read_text(encoding="utf-8") for path in second.values()
    }

    assert first_contents == second_contents
    benchmark_summary = json.loads(
        Path(first["benchmark_summary_json"]).read_text(encoding="utf-8")
    )
    assert benchmark_summary["version"] == "1.0.0"
    assert benchmark_summary["promoted_model"]["model_seed"] == 11
    assert "GCN" in Path(first["benchmark_summary_md"]).read_text(encoding="utf-8")
    assert "Conclusion" in Path(first["uncertainty_summary_md"]).read_text(
        encoding="utf-8"
    )


def test_generate_portfolio_reports_marks_unavailable_sections(tmp_path: Path) -> None:
    outputs = generate_portfolio_reports(tmp_path / "reports")

    summary = json.loads(
        Path(outputs["benchmark_summary_json"]).read_text(encoding="utf-8")
    )
    assert "benchmark comparison JSON was not supplied" in summary["unavailable_sections"]
    assert "No benchmark JSON supplied" in Path(
        outputs["benchmark_summary_md"]
    ).read_text(encoding="utf-8")


def test_generate_portfolio_reports_missing_explicit_file_is_clear(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="benchmark comparison file"):
        generate_portfolio_reports(
            tmp_path / "reports",
            benchmark_comparison_json=tmp_path / "missing.json",
        )


def test_generate_portfolio_reports_invalid_schema_raises(tmp_path: Path) -> None:
    invalid = _write_json(tmp_path / "invalid.json", {"dataset_name": "esol"})

    with pytest.raises(ValueError, match="by_model"):
        generate_portfolio_reports(
            tmp_path / "reports",
            benchmark_comparison_json=invalid,
        )


def test_write_demo_outputs(tmp_path: Path) -> None:
    prediction = {
        "input_smiles": "CCO",
        "success": True,
        "prediction": {
            "canonical_smiles": "CCO",
            "predicted_log_solubility": -0.3,
            "predicted_solubility_mol_per_litre": 0.5,
            "model_id": "demo",
            "warnings": ["educational demo"],
        },
        "error": None,
    }
    context = {
        "input_smiles": "CCO",
        "success": True,
        "context": {
            "applicability": {"maximum_similarity": 1.0, "warnings": []},
            "nearest_training_molecules": [{"sample_id": "esol:1"}],
        },
        "error": None,
    }

    outputs = write_demo_outputs([prediction], [context], tmp_path / "demo")

    assert Path(outputs["predictions_json"]).is_file()
    assert Path(outputs["context_predictions_json"]).is_file()
    summary = Path(outputs["demo_summary_md"]).read_text(encoding="utf-8")
    assert "Molecular Solubility Demo" in summary
    assert "confidence" in summary
    assert "CCO" in summary

