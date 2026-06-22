import os
from collections.abc import Mapping
from pathlib import Path


def resolve_manifest_path(
    explicit_path: Path | None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve a manifest with explicit configuration taking precedence."""
    if explicit_path is not None:
        return explicit_path
    values = os.environ if environment is None else environment
    configured = values.get("MOLGNN_MANIFEST_PATH", "").strip()
    return Path(configured) if configured else None


def resolve_host(
    explicit_host: str | None,
    environment_name: str,
    default: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a service host without overriding an explicit CLI value."""
    if explicit_host is not None:
        if not explicit_host.strip():
            raise ValueError("Service host must not be blank")
        return explicit_host
    values = os.environ if environment is None else environment
    return values.get(environment_name, default).strip() or default


def resolve_port(
    explicit_port: int | None,
    environment_name: str,
    default: int,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Resolve and validate a TCP port with explicit configuration taking precedence."""
    values = os.environ if environment is None else environment
    raw_value: int | str = (
        explicit_port
        if explicit_port is not None
        else values.get(environment_name, str(default))
    )
    try:
        port = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{environment_name} must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{environment_name} must be between 1 and 65535")
    return port
