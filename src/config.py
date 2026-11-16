"""Configuration loading with a stable content hash (recorded next to every result)."""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.toml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "rb") as f:
        return tomllib.load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def with_overrides(cfg: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    """Deep-copied config with per-section overrides, e.g. with_overrides(cfg, decision={"lam": 5})."""
    out = copy.deepcopy(cfg)
    for sec, vals in sections.items():
        out.setdefault(sec, {}).update(vals)
    return out


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
