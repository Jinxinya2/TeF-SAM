from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class Config(dict):
    """Flat, attribute-accessible configuration used by the released models."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def clone(self) -> "Config":
        return Config(copy.deepcopy(dict(self)))

    def with_overrides(self, values: Mapping[str, Any]) -> "Config":
        output = self.clone()
        for key, value in values.items():
            output[key] = value
        return output


def _flatten_sections(payload: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    owners: Dict[str, str] = {}
    for section, values in payload.items():
        if not isinstance(values, Mapping):
            if section in flattened:
                raise ValueError(f"Duplicate configuration key: {section}")
            flattened[section] = values
            owners[section] = "<root>"
            continue
        for key, value in values.items():
            if key in flattened and flattened[key] != value:
                raise ValueError(
                    f"Configuration key '{key}' is defined in both "
                    f"'{owners[key]}' and '{section}'"
                )
            flattened[key] = value
            owners[key] = str(section)
    return flattened


def load_config(path: str | os.PathLike[str]) -> Config:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("The top level of a configuration file must be a mapping")
    cfg = Config(_flatten_sections(payload))
    cfg.config_path = str(config_path.resolve())
    return cfg


def apply_overrides(cfg: Config, assignments: Iterable[str]) -> Config:
    output = cfg.clone()
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected KEY=VALUE override, got: {assignment}")
        key, raw_value = assignment.split("=", 1)
        output[key] = yaml.safe_load(raw_value)
    return output
