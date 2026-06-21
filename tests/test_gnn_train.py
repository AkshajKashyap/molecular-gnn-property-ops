from pathlib import Path

import pandas as pd

from molgnn_ops.featurization import featurize_smiles
from molgnn_ops.gnn_train import train_gnn_regressor
from molgnn_ops.graph_io import save_graphs_jsonl


def test_train_gnn_regressor_tiny_dataset(tmp_path: Path) -> None:
    smiles_values = [
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
    ]
    split_values = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
    graphs = [
        featurize_smiles(
            smiles,
            target=index * 0.25,
            split=split,
            dataset_name="synthetic",
            sample_id=f"synthetic:{index}",
        )
        for index, (smiles, split) in enumerate(zip(smiles_values, split_values, strict=True))
    ]
    graph_jsonl = tmp_path / "graphs.jsonl"
    output_dir = tmp_path / "gnn"
    save_graphs_jsonl(graphs, graph_jsonl)

    summary = train_gnn_regressor(
        graph_jsonl,
        output_dir,
        model_name="gcn",
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        batch_size=3,
        epochs=2,
        patience=2,
    )

    predictions = pd.read_csv(output_dir / "predictions.csv")
    history = pd.read_csv(output_dir / "training_history.csv")
    assert summary["best_epoch"] in {1, 2}
    assert summary["test_rmse"] >= 0
    assert len(predictions) == 6
    assert predictions["sample_id"].tolist() == [
        "synthetic:6",
        "synthetic:7",
        "synthetic:8",
        "synthetic:9",
        "synthetic:10",
        "synthetic:11",
    ]
    assert "canonical_smiles" in predictions
    assert summary["model_seed"] == 42
    assert len(history) == 2
    assert (output_dir / "models" / "gnn_regressor.pt").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "report.md").is_file()
