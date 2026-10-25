from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PUBLIC_CONFIGS = {"vips", "baseline"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    if config.get("name") not in PUBLIC_CONFIGS:
        raise ValueError(f"Config name must be one of {sorted(PUBLIC_CONFIGS)}")
    return config


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repository_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else repository_root() / path
