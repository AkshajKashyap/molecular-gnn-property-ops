from pathlib import Path

from molgnn_ops import paths


def test_ensure_project_dirs_creates_expected_directories(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {
        "RAW_DATA_DIR": tmp_path / "data" / "raw",
        "INTERIM_DATA_DIR": tmp_path / "data" / "interim",
        "PROCESSED_DATA_DIR": tmp_path / "data" / "processed",
        "REPORTS_DIR": tmp_path / "reports",
        "ARTIFACTS_DIR": tmp_path / "artifacts",
        "MODELS_DIR": tmp_path / "artifacts" / "models",
        "FIGURES_DIR": tmp_path / "reports" / "figures",
        "CONFIGS_DIR": tmp_path / "configs",
    }
    for name, directory in expected.items():
        monkeypatch.setattr(paths, name, directory)

    paths.ensure_project_dirs()

    assert all(directory.is_dir() for directory in expected.values())
