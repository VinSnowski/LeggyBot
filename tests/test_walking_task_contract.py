"""Behavioral snapshot for the production walking task.

This deliberately fingerprints the complete nested environment and runner
configurations.  It exists to make task-definition refactors prove that they
preserve every inherited setting, not merely the handful of values that are
interesting to individual unit tests.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from mjlab.tasks.registry import load_rl_cfg

from locomotion.task import make_env_cfg, register_tasks
from locomotion.tasks import WALKING_TASK_ID


EXPECTED_CONFIG_HASHES = {
    "train": "43cab1d3877f26bdf949284ff8b13d855d9f5963ee43ee0a5ece13af7afb0156",
    "play": "d43778a8a2bb6d4b622c8ee3bd123007a3bfc224fc57a280aab02addd836c179",
    "runner": "73453a4fc1bc00b5d8880b05af79b155b5a9c1fe75422c439ef736c988736c74",
}


def _name(value: Any) -> str:
    return getattr(value, "__name__", type(value).__name__)


def _normalize(value: Any) -> Any:
    """Convert nested config objects into a stable, JSON-compatible value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, enum.Enum):
        return {"$enum": f"{type(value).__name__}.{value.name}"}
    if isinstance(value, torch.Tensor):
        return {
            "$tensor": value.detach().cpu().tolist(),
            "dtype": str(value.dtype),
        }
    if isinstance(value, type):
        return {"$type": value.__name__}
    if isinstance(value, functools.partial):
        return {
            "$partial": _normalize(value.func),
            "args": _normalize(value.args),
            "keywords": _normalize(value.keywords),
        }
    if callable(value):
        return {"$callable": _name(value)}
    if dataclasses.is_dataclass(value):
        return {
            **{
                field.name: _normalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize(item) for item in value), key=repr)
    if hasattr(value, "__dict__"):
        return {
            "$type": type(value).__name__,
            **{
                key: _normalize(item)
                for key, item in sorted(vars(value).items())
                if not key.startswith("_")
            },
        }
    return {"$repr": repr(value), "$type": type(value).__name__}


def _config_hash(value: Any) -> str:
    payload = json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_complete_walking_configuration_is_unchanged() -> None:
    register_tasks()
    configurations = {
        "train": make_env_cfg(
            WALKING_TASK_ID,
            num_envs=7,
            seed=123,
            play=False,
        ),
        "play": make_env_cfg(
            WALKING_TASK_ID,
            num_envs=7,
            seed=123,
            play=True,
        ),
        "runner": load_rl_cfg(WALKING_TASK_ID),
    }
    assert {
        name: _config_hash(configuration)
        for name, configuration in configurations.items()
    } == EXPECTED_CONFIG_HASHES
