from pathlib import Path

import pandas as pd

from molgnn_ops import benchmark_compare


def test_run_split_comparison_with_mocked_benchmarks(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_benchmark(
        dataset_name,
        output_dir,
        split_strategy,
        seed,
        radius,
        n_bits,
        overwrite,
    ):
        calls.append((split_strategy, seed, output_dir, overwrite))
        offset = 0.5 if split_strategy == "scaffold" else 0.0
        return {
            "dataset_name": dataset_name,
            "split_strategy": split_strategy,
            "seed": seed,
            "best_model": "random_forest",
            "key_metric": "rmse",
            "validation_metric": seed / 100 + offset,
            "test_metric": seed / 100 + offset + 0.1,
            "metrics_json": f"metrics-{split_strategy}-{seed}.json",
            "report_md": f"report-{split_strategy}-{seed}.md",
        }

    monkeypatch.setattr(benchmark_compare, "run_fingerprint_benchmark", fake_benchmark)
    summary = benchmark_compare.run_split_comparison(
        "esol",
        tmp_path,
        seeds=[42, 43],
        split_strategies=["random", "scaffold"],
        n_bits=64,
        overwrite=True,
    )

    metrics = pd.read_csv(tmp_path / "comparison_metrics.csv")
    assert len(calls) == 4
    assert len(metrics) == 4
    assert summary["by_split_strategy"]["scaffold"]["mean_test_metric"] > summary[
        "by_split_strategy"
    ]["random"]["mean_test_metric"]
    assert (tmp_path / "comparison_summary.json").is_file()
    assert (tmp_path / "comparison_report.md").is_file()
