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
