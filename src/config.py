"""Configuration loader with dot-access and env var resolution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Loads YAML config files, merges them, resolves ${ENV_VAR} references."""

    _ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::-[^}]*)?\}")

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self._data: Dict[str, Any] = {}

    def load(self, *filenames: str) -> "ConfigLoader":
        """Load and merge YAML files in order. Later files override earlier."""
        for filename in filenames:
            path = self.config_dir / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self._deep_merge(self._data, data)
            else:
                raise FileNotFoundError(f"Config file not found: {path}")
        self._resolve_env_vars(self._data)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-separated key, e.g. 'index.top_k'."""
        keys = key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        raise AttributeError(f"Config has no key '{name}'")

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> None:
        """Merge override into base in-place."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value

    @classmethod
    def _resolve_env_vars(cls, data: Any) -> None:
        """Replace ${VAR_NAME} with environment variable values."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = cls._resolve_str(value)
                elif isinstance(value, (dict, list)):
                    cls._resolve_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    data[i] = cls._resolve_str(item)
                elif isinstance(item, (dict, list)):
                    cls._resolve_env_vars(item)

    @classmethod
    def _resolve_str(cls, text: str) -> str:
        def _replace(match):
            full = match.group(1)
            # Handle ${VAR:-default} syntax
            if ":-" in full:
                var_name, default = full.split(":-", 1)
                return os.environ.get(var_name, default)
            # Handle ${VAR} syntax
            return os.environ.get(full, match.group(0))
        return cls._ENV_VAR_RE.sub(_replace, text)


class _DotDict:
    """Wrapper to enable dot-attribute access on nested dicts."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        raise AttributeError(f"No config key '{name}'")

    def __repr__(self):
        return repr(self._data)
