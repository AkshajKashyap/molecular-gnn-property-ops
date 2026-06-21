from pathlib import Path

import pandas as pd

from molgnn_ops import gnn_uncertainty, workflows
from molgnn_ops.data_sources import DatasetSpec
from molgnn_ops.graph_io import load_graphs_jsonl


def test_fixed_split_workflow_reuses_graphs_and_validates_before_uncertainty(
    tmp_path: Path, monkeypatch
) -> None:
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "molecule": [
                "CCO",
                "OCC",
                "CCN",
                "CCC",
                "CCCl",
                "CCBr",
                "CCF",
                "COC",
                "CNC",
                "CC=O",
                "CC#N",
                "c1ccccc1",
            ],
            "measurement": [float(index) for index in range(12)],
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
        description="Synthetic regression data.",
    )
    monkeypatch.setattr(workflows, "get_dataset_spec", lambda name: spec)
    download_calls = []

    def fake_download(name, overwrite=False):
        download_calls.append((name, overwrite))
        return raw_csv

    monkeypatch.setattr(workflows, "download_dataset", fake_download)
    preparation_calls = []
    real_prepare = workflows.prepare_dataset

    def tracked_prepare(*args, **kwargs):
        preparation_calls.append(kwargs["split_seed"])
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(workflows, "prepare_dataset", tracked_prepare)
    featurization_calls = []
    real_featurize = workflows.featurize_records_from_csv

    def tracked_featurize(input_csv, graph_jsonl):
        featurization_calls.append(graph_jsonl)
        return real_featurize(input_csv, graph_jsonl)

    monkeypatch.setattr(workflows, "featurize_records_from_csv", tracked_featurize)

    from molgnn_ops import gnn_train

    training_graph_paths = []

    def fake_training(graph_jsonl, output_dir, **kwargs):
        training_graph_paths.append(graph_jsonl)
        graphs = [graph for graph in load_graphs_jsonl(graph_jsonl) if graph.split != "train"]
        if kwargs["model_seed"] % 2:
            graphs.reverse()
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        pd.DataFrame(
            {
                "sample_id": [graph.sample_id for graph in graphs],
                "smiles": [graph.smiles for graph in graphs],
                "canonical_smiles": [graph.canonical_smiles for graph in graphs],
                "split": [graph.split for graph in graphs],
                "y_true": [graph.target for graph in graphs],
                "y_pred": [float(graph.target) + kwargs["model_seed"] / 1000 for graph in graphs],
            }
        ).to_csv(predictions_path, index=False)
        metrics_path = output_dir / "metrics.json"
        model_path = output_dir / "model.pt"
        metrics_path.write_text("{}", encoding="utf-8")
        model_path.write_text("model", encoding="utf-8")
        return {
            "best_epoch": 1,
            "validation_metrics": {"rmse": 0.5, "mae": 0.4, "r2": 0.2},
            "test_metrics": {"rmse": 0.6, "mae": 0.5, "r2": 0.1},
            "artifacts": {
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
                "model": str(model_path),
            },
        }

    monkeypatch.setattr(gnn_train, "train_gnn_regressor", fake_training)
    validation_state = {"complete": False}
    real_alignment = gnn_uncertainty.load_ensemble_predictions

    def tracked_alignment(paths):
        aligned = real_alignment(paths)
        validation_state["complete"] = True
        return aligned

    monkeypatch.setattr(gnn_uncertainty, "load_ensemble_predictions", tracked_alignment)

    def fake_uncertainty(paths, output_dir, target_coverages=None):
        assert validation_state["complete"]
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "uncertainty_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        return {
            "ensemble_test_metrics": {"rmse": 0.55, "mae": 0.45, "r2": 0.15},
            "interval_results": [
                {
                    "target_coverage": 0.9,
                    "empirical_coverage": 1.0,
                    "mean_interval_width": 1.2,
                }
            ],
            "uncertainty_error_correlations": {"pearson": 0.2, "spearman": 0.3},
            "selective_prediction": [],
            "artifacts": {"uncertainty_summary_json": str(summary_path)},
        }

    monkeypatch.setattr(workflows, "run_gnn_uncertainty_analysis", fake_uncertainty)
    summary = workflows.run_fixed_split_gnn_ensemble(
        "synthetic",
        tmp_path / "fixed",
        split_strategy="random",
        split_seed=17,
        model_seeds=[2, 3, 5],
        epochs=2,
        overwrite=True,
    )

    assert download_calls == [("synthetic", True)]
    assert preparation_calls == [17]
    assert len(featurization_calls) == 1
    assert len(training_graph_paths) == 3
    assert len(set(training_graph_paths)) == 1
    assert summary["split_seed"] == 17
    assert summary["model_seeds"] == [2, 3, 5]
    assert [model["model_seed"] for model in summary["models"]] == [2, 3, 5]
    assert summary["duplicate_audit"]["duplicate_canonical_smiles_groups"] == 1
    assert summary["duplicate_audit"]["duplicate_groups_with_conflicting_targets"] == 1
    assert Path(summary["artifacts"]["summary_json"]).is_file()
    assert Path(summary["artifacts"]["report_md"]).is_file()
