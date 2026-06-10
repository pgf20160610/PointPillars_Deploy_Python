
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_state(path: str | Path, map_location="cpu") -> dict[str, Any]:
    obj = torch.load(str(path), map_location=map_location)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported checkpoint object: {type(obj)}")
    return obj


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_model_weights(model: torch.nn.Module, ckpt_path: str | Path, map_location="cpu", strict: bool = True):
    state = strip_module_prefix(load_checkpoint_state(ckpt_path, map_location=map_location))
    incompatible = model.load_state_dict(state, strict=strict)
    return incompatible
