import json
from pathlib import Path

import pandas as pd

from molgnn_ops.workflows import run_gnn_uncertainty_analysis


def test_uncertainty_workflow_writes_expected_artifacts(tmp_path: Path) -> None:
    smiles = [
        "CCO",
        "CCN",
        "c1ccccc1",
        "C1CCCCC1",
        "CC(=O)O",
        "CCCl",
        "CCCC",
        "CCBr",
    ]
    splits = ["val"] * 4 + ["test"] * 4
    targets = [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5]
    prediction_paths = []
    for run_index, offset in enumerate([-0.2, 0.0, 0.3]):
        path = tmp_path / f"seed_{42 + run_index}" / "predictions.csv"
        path.parent.mkdir(parents=True)
        predictions = [
            target + offset * (index % 3 - 1)
            for index, target in enumerate(targets)
        ]
        pd.DataFrame(
            {
                "smiles": smiles,
                "split": splits,
                "y_true": targets,
                "y_pred": predictions,
            }
        ).to_csv(path, index=False)
        prediction_paths.append(path)

    summary = run_gnn_uncertainty_analysis(
        prediction_paths,
        tmp_path / "output",
    )

    assert summary["ensemble_members"] == 3
    assert summary["seeds"] == [42, 43, 44]
    assert len(summary["interval_results"]) == 3
    assert len(summary["selective_prediction"]) == 4
    assert all(Path(path).is_file() for path in summary["artifacts"].values())
    assert all(
        (tmp_path / "output" / path).is_file() for path in summary["plots"].values()
    )
    saved = json.loads(
        Path(summary["artifacts"]["uncertainty_summary_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert saved["detailed_interval_coverage"] == 0.9
