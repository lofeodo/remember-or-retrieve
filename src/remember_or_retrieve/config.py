"""Load the project's YAML configuration."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Return the parsed config. Paths inside it are relative to the repo root."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config path (relative to the repo root) to an absolute path."""
    return REPO_ROOT / relative
