import json
import re
import tomllib
from pathlib import Path

import pytest

from molgnn_ops import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == "1.0.0"
    assert pyproject["project"]["version"] == __version__


def test_citation_cff_has_required_fields() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    for field in [
        "cff-version: 1.2.0",
        'title: "molecular-gnn-property-ops"',
        'version: "1.0.0"',
        'date-released: "2026-06-24"',
        "family-names: \"Kashyap\"",
        "given-names: \"Akshaj\"",
        "repository-code:",
        "type: software",
    ]:
        assert field in citation


def test_tracked_benchmark_json_matches_headline_values() -> None:
    summary = json.loads(
        (REPO_ROOT / "reports/portfolio/benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {row["method"]: row for row in summary["benchmark_rows"]}

    assert summary["version"] == "1.0.0"
    assert rows["Fingerprint random forest"]["test_rmse_mean"] == pytest.approx(
        1.847980833926914
    )
    assert rows["GCN"]["test_rmse_mean"] == pytest.approx(1.339454356739515)
    assert rows["GIN"]["test_rmse_mean"] == pytest.approx(1.4498563826784234)
    assert rows["Promoted fixed-split GCN"]["test_rmse_mean"] == pytest.approx(
        1.3501974828901644
    )
    assert summary["data_quality"]["conflicting_target_groups"] == 6


def test_release_docs_have_required_sections() -> None:
    model_card = (REPO_ROOT / "docs/model_card.md").read_text(encoding="utf-8")
    architecture = (REPO_ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    methodology = (REPO_ROOT / "docs/experimental_methodology.md").read_text(
        encoding="utf-8"
    )

    for heading in [
        "Model Overview",
        "Training And Selection",
        "Data Quality",
        "Uncertainty And Applicability",
        "Limitations",
        "Reproducibility",
    ]:
        assert heading in model_card
    assert "mermaid" in architecture
    assert "Promoted Registry Structure" in architecture
    assert "Why Scaffold Splits Matter" in methodology
    assert "Why The First Uncertainty Attempt Was Rejected" in methodology


def test_readme_local_links_exist() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    local_links = [
        link
        for link in links
        if not link.startswith(("http://", "https://", "#"))
    ]

    assert local_links
    for link in local_links:
        target = link.split("#", 1)[0]
        assert (REPO_ROOT / target).exists(), f"README link does not exist: {link}"

