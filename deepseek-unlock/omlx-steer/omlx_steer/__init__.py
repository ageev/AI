"""omlx-steer: per-request activation steering profiles for oMLX.

``install()`` registers post-import hooks: each oMLX module is patched right
after it is first imported, and only if the sources of the functions the
patch relies on match the pinned hashes (pins.json). Any mismatch leaves the
module untouched; the steering aliases then answer 503 while the stock model
keeps working exactly as before.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
from pathlib import Path

__version__ = "0.1.0"
logger = logging.getLogger("omlx_steer")

SUPPORTED_OMLX = ("0.8.3",)
PINS_FILE = Path(os.environ.get("OMLX_STEER_PINS") or Path(__file__).with_name("pins.json"))


def _resolve_attr(module, qualname):
    obj = module
    for part in qualname.split("."):
        obj = inspect.getattr_static(obj, part) if inspect.isclass(obj) else getattr(obj, part)
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        obj = obj.fget
    return obj


def source_hash(module, qualname) -> str:
    return hashlib.sha256(inspect.getsource(_resolve_attr(module, qualname)).encode()).hexdigest()


class State:
    """Patch bookkeeping shared with the middleware."""

    def __init__(self, store):
        self.store = store
        self.reason = "model code not loaded yet"
        self._lock = threading.Lock()
        self.patched: set[str] = set()
        self.failed: dict[str, str] = {}
        self.required: dict[str, set[str]] = {}

    def ready(self, model_type) -> bool:
        if self.failed:
            return False
        need = self.required.get(model_type or "", None)
        if need is None:
            self.reason = f"no steering adapter for model type {model_type!r}"
            return False
        missing = need - self.patched
        if missing:
            self.reason = "model code not loaded yet"
            return False
        return True


    def status(self) -> dict:
        from . import rows

        aliases = {}
        for alias, raw in self.store.aliases().items():
            aliases[alias] = {"base_model": raw.get("base_model"), "ready": self.ready(raw.get("model_type"))}
        return {
            "version": __version__,
            "sweep": self.store.sweep_enabled,
            "failed": self.failed,
            "patched": sorted(self.patched),
            "aliases": aliases,
            "stats": dict(rows.stats),
        }


STATE: State | None = None


def _check_pins(module, names, pins) -> str | None:
    expected = pins.get(module.__name__, {})
    for name in names:
        want = expected.get(name)
        if want is None:
            return f"no pin for {module.__name__}:{name}"
        try:
            got = source_hash(module, name)
        except Exception as exc:
            return f"{module.__name__}:{name} not found ({exc})"
        if got != want:
            return f"{module.__name__}:{name} changed upstream"
    return None


def _plan():
    from . import integration
    from .adapters import deepseek_v41

    plan = {
        "omlx.request": (integration.PINNED["omlx.request"], integration.patch_request),
        "omlx.patches.mlx_lm_mtp.prompt_priming": (
            integration.PINNED["omlx.patches.mlx_lm_mtp.prompt_priming"],
            integration.patch_priming,
        ),
        "omlx.scheduler": (integration.PINNED["omlx.scheduler"], integration.patch_scheduler),
        "omlx.patches.mlx_lm_mtp.batch_generator": (
            integration.PINNED["omlx.patches.mlx_lm_mtp.batch_generator"],
            integration.patch_batch_generator,
        ),
        "omlx.patches.mlx_lm_mtp.fused_batch": (
            integration.PINNED["omlx.patches.mlx_lm_mtp.fused_batch"],
            integration.patch_fused_batch,
        ),
    }
    adapter_modules = list(deepseek_v41.PINNED)
    adapter = {
        # The vendor package is imported as a whole at model load; patch once
        # its last module is in sys.modules.
        "trigger": "omlx.patches.deepseek_v41.loading",
        "modules": adapter_modules,
        "pins": deepseek_v41.PINNED,
        "install": deepseek_v41.install,
        "model_type": deepseek_v41.MODEL_TYPE,
    }
    return plan, adapter


def _apply_module(state, pins, name, module, names, patch):
    err = _check_pins(module, names, pins)
    if err:
        state.failed[name] = err
        state.reason = err
        logger.error("omlx-steer: not patching %s: %s", name, err)
        return
    patch(module)
    state.patched.add(name)
    logger.info("omlx-steer: patched %s", name)


def _apply_adapter(state, pins, adapter):
    modules = {}
    for name in adapter["modules"]:
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        modules[name] = module
        err = _check_pins(module, adapter["pins"][name], pins)
        if err:
            state.failed[name] = err
            state.reason = err
            logger.error("omlx-steer: adapter %s disabled: %s", adapter["model_type"], err)
            return
    adapter["install"](modules)
    state.patched.update(adapter["modules"])


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = hooks  # module name -> callable(module)
        self._busy = set()

    def find_spec(self, fullname, path, target=None):
        hook = self.hooks.get(fullname)
        if hook is None or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module, _original=original):
            _original(module)
            hook(module)

        loader.exec_module = exec_module
        return spec


def install(profiles_dir: str | os.PathLike | None = None, *, sweep: bool | None = None) -> State:
    """Install hooks; call before ``omlx`` modules are imported."""
    global STATE
    from .profiles import ProfileStore

    root = Path(profiles_dir or os.environ.get("OMLX_STEER_PROFILES", Path.home() / "omlx-steer-data" / "profiles"))
    if sweep is None:
        sweep = os.environ.get("OMLX_STEER_SWEEP", "") == "1"
    state = State(ProfileStore(root, sweep_enabled=sweep))
    STATE = state

    from omlx._version import __version__ as omlx_version

    if omlx_version not in SUPPORTED_OMLX:
        state.failed["omlx"] = f"oMLX {omlx_version} is not supported (pinned {SUPPORTED_OMLX})"
        state.reason = state.failed["omlx"]
        logger.error("omlx-steer: %s; steering disabled", state.reason)
        _install_middleware_hook(state)
        return state
    pins = json.loads(PINS_FILE.read_text())
    plan, adapter = _plan()
    # fused_batch is imported lazily by the first shared MTP verify; it is
    # patched when that happens and is not needed before.
    state.required[adapter["model_type"]] = (set(plan) - {"omlx.patches.mlx_lm_mtp.fused_batch"}) | set(
        adapter["modules"]
    )

    hooks = {}
    for name, (names, patch) in plan.items():
        hooks[name] = lambda module, _n=name, _names=names, _patch=patch: _apply_module(
            state, pins, _n, module, _names, _patch
        )
    hooks[adapter["trigger"]] = lambda module: _apply_adapter(state, pins, adapter)
    for name, hook in list(hooks.items()):
        if name in sys.modules:
            hook(sys.modules[name])
            hooks.pop(name)
    sys.meta_path.insert(0, _Finder(hooks))
    _install_middleware_hook(state)
    logger.info("omlx-steer %s: profiles in %s, sweep %s", __version__, root, "on" if sweep else "off")
    return state


def _install_middleware_hook(state):
    def add(module):
        from .middleware import SteerMiddleware

        module.app.add_middleware(SteerMiddleware, state=state)
        logger.info("omlx-steer: middleware installed")

    if "omlx.server" in sys.modules:
        add(sys.modules["omlx.server"])
        return
    for finder in sys.meta_path:
        if isinstance(finder, _Finder):
            finder.hooks["omlx.server"] = add
            return
    sys.meta_path.insert(0, _Finder({"omlx.server": add}))
