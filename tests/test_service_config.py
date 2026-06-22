from pathlib import Path

import pytest

from molgnn_ops.service_config import (
    resolve_host,
    resolve_manifest_path,
    resolve_port,
)


def test_manifest_path_uses_environment() -> None:
    path = resolve_manifest_path(None, {"MOLGNN_MANIFEST_PATH": "/models/manifest.json"})

    assert path == Path("/models/manifest.json")


def test_explicit_service_configuration_takes_precedence() -> None:
    environment = {
        "MOLGNN_MANIFEST_PATH": "/environment/manifest.json",
        "API_HOST": "environment-host",
        "API_PORT": "8123",
    }

    assert resolve_manifest_path(Path("explicit.json"), environment) == Path(
        "explicit.json"
    )
    assert resolve_host("127.0.0.1", "API_HOST", "0.0.0.0", environment) == (
        "127.0.0.1"
    )
    assert resolve_port(9000, "API_PORT", 8000, environment) == 9000


def test_host_and_port_use_environment() -> None:
    environment = {"DASHBOARD_HOST": "0.0.0.0", "DASHBOARD_PORT": "8502"}

    assert resolve_host(None, "DASHBOARD_HOST", "localhost", environment) == "0.0.0.0"
    assert resolve_port(None, "DASHBOARD_PORT", 8501, environment) == 8502


@pytest.mark.parametrize("value", ["abc", "0", "65536"])
def test_invalid_environment_port_raises_clear_error(value: str) -> None:
    with pytest.raises(ValueError, match="API_PORT"):
        resolve_port(None, "API_PORT", 8000, {"API_PORT": value})
