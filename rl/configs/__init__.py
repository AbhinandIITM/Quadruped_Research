"""Configuration loading for the LLM-vs-RL quadruped study.

``load_config(name)`` reads ``<this directory>/<name>.yaml`` and returns it as a plain
``dict``. Nothing here imports a simulator or a training stack, so a config can be inspected
from a bare interpreter. ``SMOKE`` is provided for convenience but is resolved lazily, so
importing this module never touches the filesystem.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["CONFIG_DIR", "SMOKE", "available_configs", "load_config"]


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a plain ``dict``.

    ``name`` may be given with or without the ``.yaml`` suffix.
    """
    stem = name[: -len(".yaml")] if name.endswith(".yaml") else name
    path = os.path.join(CONFIG_DIR, f"{stem}.yaml")
    if not os.path.isfile(path):
        known = ", ".join(available_configs()) or "none found"
        raise FileNotFoundError(f"no config {name!r} in {CONFIG_DIR} (available: {known})")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping at the top level, got {type(data).__name__}")
    return data


def available_configs() -> list[str]:
    """Stems of every ``*.yaml`` file in this directory, sorted."""
    return sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
    )


def __getattr__(name: str) -> Any:
    """Resolve ``SMOKE`` on first access so import stays lazy."""
    if name == "SMOKE":
        return load_config("smoke")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
