#!/usr/bin/env python3
"""JSON manifest helper for PointPillars model preparation tools."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def load_manifest(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assets = data.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("manifest 'assets' must be a list")
    for i, asset in enumerate(assets):
        for key in ("name", "url", "output"):
            if key not in asset:
                raise ValueError(f"manifest asset #{i} missing key: {key}")
        asset.setdefault("type", "generic")
    return data


def get_section(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = manifest.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"manifest section '{name}' must be an object")
    return value


def require_keys(section: Dict[str, Any], keys: Iterable[str], section_name: str) -> None:
    for key in keys:
        if key not in section or section[key] in (None, ""):
            raise ValueError(f"manifest section '{section_name}' missing key: {key}")
