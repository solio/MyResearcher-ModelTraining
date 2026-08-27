from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ContractError


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    raw: Mapping[str, Any]
    project_root: Path
    data_root: Path

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError("CONFIG_NOT_FOUND", str(config_path)) from exc
        except yaml.YAMLError as exc:
            raise ContractError("CONFIG_INVALID", str(exc), path=str(config_path)) from exc
        if not isinstance(raw, dict):
            raise ContractError("CONFIG_INVALID", "config root must be an object")
        if raw.get("config_schema_version") != "myresearcher.semantic-baseline-config.v1":
            raise ContractError(
                "CONFIG_SCHEMA_VERSION_MISMATCH",
                "unsupported config_schema_version",
                observed=raw.get("config_schema_version"),
            )
        configured_project_root = Path(str(raw.get("project_root", ".")))
        project_root = (
            configured_project_root.resolve()
            if configured_project_root.is_absolute()
            else (config_path.parent / configured_project_root).resolve()
        )
        data = raw.get("data")
        if not isinstance(data, dict) or not data.get("root"):
            raise ContractError("CONFIG_INVALID", "data.root is required")
        data_root = Path(str(data["root"])).expanduser().resolve()
        return cls(
            path=config_path,
            raw=raw,
            project_root=project_root,
            data_root=data_root,
        )

    def repo_path(self, key: str) -> Path:
        value = self.raw.get(key)
        if not value:
            raise ContractError("CONFIG_INVALID", f"{key} is required")
        path = Path(str(value))
        return path if path.is_absolute() else self.project_root / path

    def data_path(self, key: str) -> Path:
        data = self.raw["data"]
        value = data.get(key)
        if not value:
            raise ContractError("CONFIG_INVALID", f"data.{key} is required")
        path = Path(str(value))
        return path if path.is_absolute() else self.data_root / path

    @property
    def expected(self) -> Mapping[str, Any]:
        value = self.raw["data"].get("expected", {})
        if not isinstance(value, dict):
            raise ContractError("CONFIG_INVALID", "data.expected must be an object")
        return value
