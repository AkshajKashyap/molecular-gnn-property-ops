import json
from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops import workflows
from molgnn_ops.data_sources import DatasetSpec


def test_run_fingerprint_benchmark_with_local_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "molecule": [
                "CCO",
                "CCN",
                "CCC",
                "CCCl",
                "CCBr",
                "CCF",
                "COC",
                "CNC",
                "CC=O",
                "CC#N",
                "C1CCCCC1",
                "c1ccccc1",
            ],
            "measurement": [index + 0.25 for index in range(12)],
        }
    ).to_csv(raw_csv, index=False)
    spec = DatasetSpec(
        name="synthetic",
        url="https://example.test/synthetic.csv",
        raw_filename="synthetic.csv",
        smiles_col="molecule",
        target_col="measurement",
        task_type="regression",
        default_split_strategy="random",
        description="Synthetic regression dataset.",
    )
    monkeypatch.setattr(workflows, "get_dataset_spec", lambda name: spec)
    monkeypatch.setattr(
        workflows,
        "download_dataset",
        lambda name, overwrite=False: raw_csv,
    )

    summary = workflows.run_fingerprint_benchmark(
        "synthetic",
        tmp_path / "artifacts" / "benchmarks",
        split_strategy="random",
        seed=7,
        n_bits=64,
        overwrite=True,
    )

    assert summary["dataset_name"] == "synthetic"
    assert summary["task_type"] == "regression"
    assert summary["best_model"] in {"ridge", "random_forest"}
    assert Path(summary["prepared_csv"]).is_file()
    assert Path(summary["fingerprint_npz"]).is_file()
    assert Path(summary["metrics_json"]).is_file()
    assert Path(summary["report_md"]).is_file()
    summary_path = Path(summary["summary_json"])
    assert json.loads(summary_path.read_text(encoding="utf-8"))["seed"] == 7

    def fail_download(*args, **kwargs):
        raise AssertionError("download called for a cached benchmark")

    monkeypatch.setattr(workflows, "download_dataset", fail_download)
    cached_summary = workflows.run_fingerprint_benchmark(
        "synthetic",
        tmp_path / "artifacts" / "benchmarks",
        split_strategy="random",
        seed=7,
        n_bits=64,
    )

    assert cached_summary == summary

    with pytest.raises(ValueError, match="different configuration"):
        workflows.run_fingerprint_benchmark(
            "synthetic",
            tmp_path / "artifacts" / "benchmarks",
            split_strategy="random",
            seed=7,
            n_bits=128,
        )


def test_run_gnn_benchmark_with_local_dataset(tmp_path: Path, monkeypatch) -> None:
    from molgnn_ops import gnn_train

    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "molecule": ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "C1CCCCC1"],
            "measurement": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ).to_csv(raw_csv, index=False)
    spec = DatasetSpec(
        name="synthetic",
        url="https://example.test/synthetic.csv",
        raw_filename="synthetic.csv",
        smiles_col="molecule",
        target_col="measurement",
        task_type="regression",
        default_split_strategy="random",
        description="Synthetic regression dataset.",
    )
    monkeypatch.setattr(workflows, "get_dataset_spec", lambda name: spec)
    monkeypatch.setattr(
        workflows,
        "download_dataset",
        lambda name, overwrite=False: raw_csv,
    )

    def fake_training(graph_jsonl, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "model": str(output_dir / "model.pt"),
            "metrics": str(output_dir / "metrics.json"),
            "report": str(output_dir / "report.md"),
        }
        for path in artifacts.values():
            Path(path).write_text("artifact", encoding="utf-8")
        return {
            "artifacts": artifacts,
            "best_epoch": 1,
            "best_val_rmse": 0.5,
            "test_rmse": 0.6,
            "fingerprint_comparison": None,
        }

    monkeypatch.setattr(gnn_train, "train_gnn_regressor", fake_training)
    output_dir = tmp_path / "gnn_benchmark"
    summary = workflows.run_gnn_benchmark(
        "synthetic",
        output_dir,
        split_strategy="random",
        epochs=2,
        overwrite=True,
    )

    assert summary["model_name"] == "gcn"
    assert summary["test_rmse"] == 0.6
    assert Path(summary["graph_jsonl"]).is_file()
    assert Path(summary["summary_json"]).is_file()
