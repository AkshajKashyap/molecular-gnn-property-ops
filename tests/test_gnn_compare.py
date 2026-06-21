import json
from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops import gnn_compare


def test_run_gnn_comparison_collects_and_summarizes_runs(
    tmp_path: Path, monkeypatch
) -> None:
    observed_calls = []

    def fake_benchmark(dataset_name, output_dir, **kwargs):
        observed_calls.append((dataset_name, output_dir, kwargs))
        model_offset = 0.0 if kwargs["model_name"] == "gcn" else 0.4
        seed_offset = (kwargs["seed"] - 42) * 0.2
        test_rmse = 1.0 + model_offset + seed_offset
        return {
            "best_epoch": 3,
            "validation_metrics": {
                "rmse": test_rmse - 0.1,
                "mae": test_rmse - 0.3,
                "r2": 0.5 - model_offset,
            },
            "test_metrics": {
                "rmse": test_rmse,
                "mae": test_rmse - 0.2,
                "r2": 0.4 - model_offset,
            },
            "metrics_json": str(output_dir / "training" / "metrics.json"),
            "report_md": str(output_dir / "training" / "report.md"),
        }

    monkeypatch.setattr(gnn_compare, "run_gnn_benchmark", fake_benchmark)
    output_dir = tmp_path / "comparison"
    summary = gnn_compare.run_gnn_comparison(
        "synthetic",
        output_dir,
        model_names=["gcn", "gin"],
        seeds=[42, 43],
        epochs=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.2,
    )

    assert len(observed_calls) == 4
    assert observed_calls[0][1] == output_dir / "gcn" / "seed_42"
    assert observed_calls[-1][1] == output_dir / "gin" / "seed_43"
    assert observed_calls[0][2]["hidden_dim"] == 16
    assert summary["by_model"]["gcn"]["mean_test_rmse"] == pytest.approx(1.1)
    assert summary["by_model"]["gcn"]["std_test_rmse"] == pytest.approx(0.1)
    assert summary["by_model"]["gin"]["mean_test_mae"] == pytest.approx(1.3)
    assert summary["best_single_run"]["model_name"] == "gcn"
    assert summary["best_single_run"]["seed"] == 42
    assert summary["best_mean_model"] == "gcn"

    metrics_csv = Path(summary["comparison_metrics_csv"])
    summary_json = Path(summary["comparison_summary_json"])
    report_md = Path(summary["comparison_report_md"])
    assert len(pd.read_csv(metrics_csv)) == 4
    assert json.loads(summary_json.read_text(encoding="utf-8"))["best_mean_model"] == "gcn"
    report = report_md.read_text(encoding="utf-8")
    assert "## Fingerprint Baseline Results" in report
    assert "## GCN Results" in report
    assert "## GIN Results" in report
    assert all(Path(path).is_file() for path in summary["figures"].values())


@pytest.mark.parametrize(
    ("model_names", "seeds", "match"),
    [
        ([], [42], "model name"),
        (["transformer"], [42], "Unsupported GNN model"),
        (["gcn"], [], "seed"),
    ],
)
def test_run_gnn_comparison_rejects_invalid_lists(
    tmp_path: Path,
    model_names: list[str],
    seeds: list[int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        gnn_compare.run_gnn_comparison(
            "synthetic",
            tmp_path,
            model_names=model_names,
            seeds=seeds,
        )
