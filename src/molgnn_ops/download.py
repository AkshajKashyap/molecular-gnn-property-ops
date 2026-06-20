import shutil
from pathlib import Path
from urllib.request import urlopen

from molgnn_ops.data_sources import get_dataset_spec
from molgnn_ops.paths import RAW_DATA_DIR


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Download a URL atomically, reusing an existing file unless asked to overwrite."""
    if output_path.is_file() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    try:
        with urlopen(url, timeout=60) as response:  # noqa: S310 - URLs come from the registry
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        temporary_path.replace(output_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} to {output_path}: {error}") from error
    return output_path


def download_dataset(
    name: str,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a registered dataset into its raw-data directory."""
    spec = get_dataset_spec(name)
    destination_dir = output_dir if output_dir is not None else RAW_DATA_DIR / spec.name
    return download_file(
        spec.url,
        destination_dir / spec.raw_filename,
        overwrite=overwrite,
    )
