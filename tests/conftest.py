import json
from pathlib import Path

import pytest
import torch

from molgnn_ops.gnn_models import GCNRegressor
from molgnn_ops.model_registry import promote_model


@pytest.fixture
def synthetic_candidate_dirs(tmp_path: Path) -> list[Path]:
    run_root = tmp_path / "fixed_split" / "split_seed_7"
    run_root.mkdir(parents=True)
    (run_root / "fixed_split_ensemble_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "split_strategy": "scaffold",
                "split_seed": 7,
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    configurations = [
        (11, 1.0, 0.5),
        (12, 0.8, 1.5),
        (13, 1.2, 0.4),
    ]
    for model_seed, validation_rmse, test_rmse in configurations:
        run_dir = run_root / f"model_seed_{model_seed}"
        model_dir = run_dir / "models"
        model_dir.mkdir(parents=True)
        torch.manual_seed(model_seed)
        model = GCNRegressor(input_dim=22, hidden_dim=8, num_layers=2, dropout=0.0)
        checkpoint_path = model_dir / "gnn_regressor.pt"
        torch.save(
            {
                "model_name": "gcn",
                "model_state_dict": model.state_dict(),
                "input_dim": 22,
                "hyperparameters": {
                    "hidden_dim": 8,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "epochs": 2,
                },
                "target_mean": -2.0,
                "target_std": 1.5,
                "best_epoch": 1,
                "model_seed": model_seed,
            },
            checkpoint_path,
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model_name": "gcn",
                    "dataset_source": "synthetic",
                    "model_seed": model_seed,
                    "validation_metrics": {
                        "rmse": validation_rmse,
                        "mae": validation_rmse - 0.1,
                        "r2": 0.2,
                    },
                    "test_metrics": {
                        "rmse": test_rmse,
                        "mae": test_rmse - 0.1,
                        "r2": 0.1,
                    },
                    "hyperparameters": {
                        "hidden_dim": 8,
                        "num_layers": 2,
                        "dropout": 0.0,
                        "epochs": 2,
                    },
                    "target_normalization": {"mean": -2.0, "std": 1.5},
                    "artifacts": {"model": str(checkpoint_path)},
                }
            ),
            encoding="utf-8",
        )
        candidates.append(run_dir)
    return candidates


@pytest.fixture
def promoted_manifest_path(
    tmp_path: Path,
    synthetic_candidate_dirs: list[Path],
) -> Path:
    registry_dir = tmp_path / "registry"
    promote_model(
        synthetic_candidate_dirs,
        registry_dir,
        model_id="synthetic-gcn-v1",
    )
    return registry_dir / "manifest.json"
