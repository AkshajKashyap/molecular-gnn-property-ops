from pathlib import Path

from molgnn_ops import download as download_module
from molgnn_ops.data_sources import DatasetSpec
from molgnn_ops.download import download_dataset, download_file


def test_download_file_skips_existing_file(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "downloads" / "dataset.csv"
    source_path.write_text("new content", encoding="utf-8")
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing content", encoding="utf-8")

    result = download_file(source_path.as_uri(), output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "existing content"


def test_download_file_overwrites_existing_file(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "downloads" / "dataset.csv"
    source_path.write_text("new content", encoding="utf-8")
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing content", encoding="utf-8")

    download_file(source_path.as_uri(), output_path, overwrite=True)

    assert output_path.read_text(encoding="utf-8") == "new content"


def test_download_dataset_uses_registry_and_download_function(
    tmp_path: Path, monkeypatch
) -> None:
    spec = DatasetSpec(
        name="fake",
        url="https://example.test/fake.csv",
        raw_filename="fake.csv",
        smiles_col="smiles",
        target_col="target",
        task_type="regression",
        default_split_strategy="random",
        description="Fake dataset.",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(download_module, "get_dataset_spec", lambda name: spec)

    def fake_download(url: str, output_path: Path, overwrite: bool = False) -> Path:
        captured.update(url=url, output_path=output_path, overwrite=overwrite)
        return output_path

    monkeypatch.setattr(download_module, "download_file", fake_download)

    result = download_dataset("fake", output_dir=tmp_path, overwrite=True)

    assert result == tmp_path / "fake.csv"
    assert captured == {
        "url": spec.url,
        "output_path": tmp_path / "fake.csv",
        "overwrite": True,
    }
