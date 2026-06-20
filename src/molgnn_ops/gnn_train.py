import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch_geometric.loader import DataLoader

from molgnn_ops.gnn_data import load_pyg_dataset_from_jsonl, split_pyg_dataset
from molgnn_ops.gnn_models import GCNRegressor, GINRegressor
from molgnn_ops.logging_utils import get_logger
from molgnn_ops.reporting import write_gnn_report

LOGGER = get_logger(__name__)


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def _build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    model_classes = {"gcn": GCNRegressor, "gin": GINRegressor}
    if model_name not in model_classes:
        raise ValueError("model_name must be either 'gcn' or 'gin'")
    return model_classes[model_name](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    split_name: str,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    model.eval()
    targets: list[float] = []
    predictions: list[float] = []
    smiles_values: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            normalized_predictions = model(batch)
            batch_predictions = normalized_predictions * target_std + target_mean
            targets.extend(batch.y.view(-1).detach().cpu().tolist())
            predictions.extend(batch_predictions.detach().cpu().tolist())
            batch_smiles = batch.smiles
            smiles_values.extend(
                batch_smiles if isinstance(batch_smiles, list) else [batch_smiles]
            )

    target_array = np.asarray(targets, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    frame = pd.DataFrame(
        {
            "smiles": smiles_values,
            "split": split_name,
            "y_true": target_array,
            "y_pred": prediction_array,
        }
    )
    return _regression_metrics(target_array, prediction_array), frame


def _nearby_fingerprint_metrics(
    graph_jsonl: Path,
    output_dir: Path,
    seed: int,
) -> tuple[Path, dict] | None:
    candidates = [
        graph_jsonl.parent / "baseline" / "metrics.json",
        output_dir.parent / "baseline" / "metrics.json",
        output_dir.parent.parent / f"seed_{seed}" / "baseline" / "metrics.json",
    ]
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None


def train_gnn_regressor(
    graph_jsonl: Path,
    output_dir: Path,
    model_name: str = "gcn",
    seed: int = 42,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10,
) -> dict:
    """Train a graph regressor with validation early stopping and held-out evaluation."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if patience <= 0:
        raise ValueError("patience must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay must be non-negative")

    _set_random_seeds(seed)
    data_list = load_pyg_dataset_from_jsonl(graph_jsonl)
    splits = split_pyg_dataset(data_list)
    input_dim = int(splits["train"][0].x.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training %s regressor on %s", model_name.upper(), device)

    training_targets = torch.cat([data.y for data in splits["train"]]).float()
    target_mean = float(training_targets.mean())
    target_std = float(training_targets.std(unbiased=False))
    if target_std == 0:
        target_std = 1.0

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)

    model = _build_model(model_name, input_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_metrics: dict[str, float | None] = {}
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            normalized_targets = (batch.y.view(-1) - target_mean) / target_std
            predictions = model(batch)
            loss = loss_function(predictions, normalized_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            total_graphs += batch.num_graphs

        validation_metrics, _ = _evaluate(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            "val",
        )
        train_loss = total_loss / total_graphs
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": validation_metrics["mae"],
                "val_rmse": validation_metrics["rmse"],
                "val_r2": validation_metrics["r2"],
            }
        )

        validation_rmse = float(validation_metrics["rmse"])
        if validation_rmse < best_val_rmse:
            best_val_rmse = validation_rmse
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    _, validation_predictions = _evaluate(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
        "val",
    )
    test_metrics, test_predictions = _evaluate(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        "test",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "models" / "gnn_regressor.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "report.md"
    hyperparameters = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
    }
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "hyperparameters": hyperparameters,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, model_path)
    pd.concat([validation_predictions, test_predictions], ignore_index=True).to_csv(
        predictions_path,
        index=False,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    fingerprint_comparison = None
    nearby_fingerprint = _nearby_fingerprint_metrics(graph_jsonl, output_dir, seed)
    if nearby_fingerprint is not None:
        fingerprint_path, fingerprint_metrics = nearby_fingerprint
        fingerprint_rmse = fingerprint_metrics.get("test_metrics", {}).get("rmse")
        if fingerprint_rmse is not None:
            fingerprint_comparison = {
                "metrics_path": str(fingerprint_path),
                "fingerprint_test_rmse": float(fingerprint_rmse),
                "gnn_test_rmse": float(test_metrics["rmse"]),
                "rmse_difference": float(test_metrics["rmse"]) - float(fingerprint_rmse),
            }

    dataset_source = next(
        (
            data.dataset_name
            for data in data_list
            if getattr(data, "dataset_name", None) is not None
        ),
        graph_jsonl.stem,
    )
    metrics = {
        "model_name": model_name,
        "dataset_source": dataset_source,
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "hyperparameters": hyperparameters,
        "target_normalization": {"mean": target_mean, "std": target_std},
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "fingerprint_comparison": fingerprint_comparison,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "history": str(history_path),
            "report": str(report_path),
        },
    }
    write_gnn_report(metrics, report_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **metrics,
        "best_val_rmse": float(best_validation_metrics["rmse"]),
        "test_rmse": float(test_metrics["rmse"]),
    }
