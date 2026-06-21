import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from molgnn_ops.inference import load_promoted_model
from molgnn_ops.model_registry import (
    ModelManifest,
    promote_model,
    select_model_by_validation,
)


def test_candidate_ranking_uses_validation_not_test(
    synthetic_candidate_dirs: list[Path],
) -> None:
    selection = select_model_by_validation(synthetic_candidate_dirs)

    assert Path(selection["selected_run"]).name == "model_seed_12"
    assert [row["model_seed"] for row in selection["ranked_candidates"]] == [12, 11, 13]
    assert selection["selected_candidate"]["test_metrics"]["rmse"] == 1.5


def test_candidate_without_validation_metric_raises(
    synthetic_candidate_dirs: list[Path],
) -> None:
    metrics_path = synthetic_candidate_dirs[0] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("validation_metrics")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="no validation metric"):
        select_model_by_validation(synthetic_candidate_dirs)


def test_promoted_package_is_self_contained(
    tmp_path: Path,
    synthetic_candidate_dirs: list[Path],
) -> None:
    registry_dir = tmp_path / "registry"
    manifest = promote_model(
        synthetic_candidate_dirs,
        registry_dir,
        model_id="synthetic-gcn-v1",
    )
    ranking = pd.read_csv(registry_dir / "candidate_ranking.csv")

    assert manifest.model_seed == 12
    assert manifest.split_seed == 7
    assert manifest.validation_metrics["rmse"] == 0.8
    assert manifest.test_metrics["rmse"] == 1.5
    assert (registry_dir / manifest.checkpoint_path).is_file()
    assert (registry_dir / "manifest.json").is_file()
    assert (registry_dir / "selection_report.md").is_file()
    assert ranking["model_seed"].tolist() == [12, 11, 13]

    shutil.rmtree(synthetic_candidate_dirs[0].parent)
    loaded = load_promoted_model(registry_dir / "manifest.json")
    assert loaded.manifest.model_id == "synthetic-gcn-v1"
    saved_manifest = ModelManifest.model_validate_json(
        (registry_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert not Path(saved_manifest.checkpoint_path).is_absolute()
