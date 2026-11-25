"""Steering profiles: files on disk, reloaded by mtime, pinned per request.

A profile file ``<alias>.json`` exposes one virtual model id. It names the
physical model it overlays and a directions file (``.npz`` with ``r``:
float32 ``[n_layers_total, k, hidden]``, orthonormal rows per layer, zero rows
where a layer has no direction). The effective profile of a request is the
file content plus an optional sweep override; its sha goes into the prefix
cache key, so a changed profile never shares KV with an older one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("omlx_steer")

WRITERS = ("attn", "ffn", "engram")
MTP_MODES = ("last", "own", "none")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class EffectiveProfile:
    """Immutable, request-pinned steering configuration."""

    alias: str
    sha: str
    layers: frozenset[int]
    alpha: dict[int, float]
    writers: frozenset[str]
    mtp: str
    capture: dict[str, Any] | None
    dump_last_logprobs: bool
    impl: str
    dirs: Any = field(repr=False, compare=False, default=None)  # np.ndarray [L, k, D]
    _mx: dict = field(repr=False, compare=False, default_factory=dict)

    @property
    def salt(self) -> str:
        return f"omlx-steer:{self.sha}"

    @property
    def projects(self) -> bool:
        return self.dirs is not None and bool(self.layers)

    def _rows(self, layer: int):
        if layer < 0 or layer >= self.dirs.shape[0]:
            return None
        cached = self._mx.get(layer)
        if cached is None:
            import mlx.core as mx

            rows = self.dirs[layer]
            rows = rows[np.linalg.norm(rows, axis=-1) > 0]
            cached = mx.array(rows.astype(np.float32)) if len(rows) else False
            self._mx[layer] = cached
        return cached if cached is not False else None

    def direction(self, layer: int):
        """mx.array [k, D] float32 for a backbone ``layer``, or None."""
        if not self.projects or layer not in self.layers:
            return None
        return self._rows(layer)

    def mtp_direction(self, n_layers: int, stage: int):
        """Directions for DSpark stage ``stage`` (layer id n_layers + stage)."""
        if not self.projects or self.mtp == "none":
            return None
        if self.mtp == "own":
            return self._rows(n_layers + stage)
        return self._rows(n_layers - 1)  # "last"

    def alpha_for(self, layer: int) -> float:
        return self.alpha.get(layer, self.alpha.get(-1, 1.0))


def _parse_layers(spec, n_total: int) -> frozenset[int]:
    if spec in (None, "all"):
        return frozenset(range(n_total))
    out = set()
    for item in spec:
        if isinstance(item, str) and "-" in item:
            lo, hi = item.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(item))
    return frozenset(out)


def _parse_alpha(spec) -> dict[int, float]:
    if spec is None:
        return {-1: 1.0}
    if isinstance(spec, (int, float)):
        return {-1: float(spec)}
    out = {}
    for key, value in spec.items():
        out[-1 if key == "default" else int(key)] = float(value)
    return out


class ProfileStore:
    """Profile files under ``root``; thread-safe, reloaded on mtime change."""

    OVERRIDE_KEYS = ("layers", "alpha", "writers", "mtp", "directions", "dump_last_logprobs", "impl", "tag")

    def __init__(self, root: Path, *, sweep_enabled: bool = False):
        self.root = Path(root)
        self.sweep_enabled = sweep_enabled
        self._lock = threading.Lock()
        self._files: dict[str, tuple[float, dict]] = {}
        self._dirs: dict[str, tuple[float, str, np.ndarray]] = {}
        self._effective: dict[str, EffectiveProfile] = {}

    # -- discovery ---------------------------------------------------------
    def aliases(self) -> dict[str, dict]:
        """alias -> raw profile dict for every readable profile file."""
        out = {}
        if not self.root.is_dir():
            return out
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = self._read(path)
            except Exception as exc:  # a broken file must not hide the others
                logger.error("omlx-steer: profile %s unreadable: %s", path.name, exc)
                continue
            if raw.get("sweep") and not self.sweep_enabled:
                continue
            out[raw["alias"]] = raw
        return out

    def _read(self, path: Path) -> dict:
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._files.get(str(path))
            if hit and hit[0] == mtime:
                return hit[1]
        raw = json.loads(path.read_text())
        raw.setdefault("alias", path.stem)
        if raw["alias"] != path.stem:
            raise ProfileError("alias must equal the file name")
        with self._lock:
            self._files[str(path)] = (mtime, raw)
        return raw

    def _load_dirs(self, rel: str) -> tuple[str, np.ndarray]:
        path = (self.root / rel).resolve()
        if self.root.resolve() not in path.parents:
            raise ProfileError("directions file must live under the profiles directory")
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._dirs.get(str(path))
            if hit and hit[0] == mtime:
                return hit[1], hit[2]
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        with np.load(path) as npz:
            dirs = np.asarray(npz["r"], dtype=np.float32)
        if dirs.ndim != 3:
            raise ProfileError("directions must be [layers, k, hidden]")
        with self._lock:
            self._dirs[str(path)] = (mtime, sha, dirs)
        return sha, dirs

    # -- resolution --------------------------------------------------------
    def resolve(self, alias: str, override: dict | None = None) -> EffectiveProfile | None:
        raw = self.aliases().get(alias)
        if raw is None:
            return None
        spec = copy.deepcopy(raw)
        if override:
            if not (self.sweep_enabled and raw.get("sweep")):
                raise ProfileError("steering overrides are disabled for this model")
            unknown = set(override) - set(self.OVERRIDE_KEYS)
            if unknown:
                raise ProfileError(f"unknown override keys: {sorted(unknown)}")
            spec.update(override)
        dir_sha, dirs = (None, None)
        if spec.get("directions"):
            dir_sha, dirs = self._load_dirs(spec["directions"])
        n_total = dirs.shape[0] if dirs is not None else 0
        canon = json.dumps({k: v for k, v in spec.items() if k not in ("description", "listed")}, sort_keys=True)
        sha = hashlib.sha256((canon + (dir_sha or "")).encode()).hexdigest()[:16]
        with self._lock:
            hit = self._effective.get(sha)
        if hit is not None:
            return hit
        writers = frozenset(spec.get("writers", WRITERS))
        if writers - set(WRITERS):
            raise ProfileError(f"unknown writers: {sorted(writers - set(WRITERS))}")
        mtp = spec.get("mtp", "last")
        if mtp not in MTP_MODES:
            raise ProfileError(f"unknown mtp mode {mtp!r}")
        prof = EffectiveProfile(
            alias=alias,
            sha=sha,
            layers=_parse_layers(spec.get("layers"), n_total) if dirs is not None else frozenset(),
            alpha=_parse_alpha(spec.get("alpha")),
            writers=writers,
            mtp=mtp,
            capture=self._capture_spec(spec),
            dump_last_logprobs=bool(spec.get("dump_last_logprobs", False)),
            impl=spec.get("impl", "kernel"),
            dirs=dirs,
        )
        with self._lock:
            self._effective[sha] = prof
        return prof

    def _capture_spec(self, spec: dict) -> dict | None:
        cap = spec.get("capture")
        if cap is None:
            return None
        root = self.root.resolve()
        target = (root / cap.get("dir", "capture")).resolve()
        if target != root and root not in target.parents:
            raise ProfileError("capture dir must live under the profiles directory")
        out = {"dir": str(target)}
        tag = spec.get("tag")
        if tag is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", str(tag)) or str(tag).startswith("."):
                raise ProfileError("invalid capture tag")
            out["tag"] = str(tag)
        return out

    def base_model(self, alias: str) -> str | None:
        raw = self.aliases().get(alias)
        return None if raw is None else raw.get("base_model")
