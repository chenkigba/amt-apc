from importlib.resources import files, as_file
from pathlib import Path
import json
from typing import Any


class CustomDict(dict):
    """A dict that supports attribute-style access to nested values."""

    def __init__(self, config: dict):
        super().__init__(config)

    def __getattr__(self, name: str) -> dict | Any:
        value = self[name]
        if isinstance(value, dict):
            return CustomDict(value)
        else:
            return value

    def __getitem__(self, key: Any) -> Any:
        item = super().__getitem__(key)
        if isinstance(item, dict):
            return CustomDict(item)
        else:
            return item


def get_package_root() -> Path:
    """Get the root path of the amt_apc package.

    This is useful for accessing package resources like model files.
    """
    pkg_files = files("amt_apc")
    with as_file(pkg_files) as pkg_path:
        return Path(pkg_path)


def _load_config() -> CustomDict:
    """Load configuration from package resources."""
    config_file = files("amt_apc").joinpath("config.json")
    with config_file.open("r", encoding="utf-8") as f:
        config_json = json.load(f)
    return CustomDict(config_json)


config = _load_config()
