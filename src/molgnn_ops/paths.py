from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_project_dirs() -> None:
    """Create the directories used for project data, outputs, and configuration."""
    directories = (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        CONFIGS_DIR,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
