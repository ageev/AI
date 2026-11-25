"""Which steering profile each batch row runs with.

Request -> profile: the HTTP middleware puts the effective profile into the
request task's context; ``Request`` objects created in that context pin it.
Profile -> uid: recorded when the scheduler binds a request to a
BatchGenerator uid (the same moment oMLX binds MTP priming state).
uid -> row: every model forward runs inside a scope that lists the profile of
each row it carries, in row order (prefill: the request; decode: the
GenerationBatch uids; MTP helpers: their own row batch).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Optional, Sequence

logger = logging.getLogger("omlx_steer")

# Effective profile of the HTTP request being handled (None = stock).
REQUEST_PROFILE: contextvars.ContextVar = contextvars.ContextVar("omlx_steer_request", default=None)
# Draft caches -> profile while several requests draft one after another
# (oMLX's shared DSpark draft jobs), keyed by id() of each stage cache.
DRAFT_CACHES: contextvars.ContextVar = contextvars.ContextVar("omlx_steer_draft_caches", default=None)
# Per-forward row profiles: tuple[EffectiveProfile | None, ...] or None (no scope).
ROWS: contextvars.ContextVar = contextvars.ContextVar("omlx_steer_rows", default=None)

# "prefill" while prompt tokens are processed, "decode" for generation steps.
PHASE: contextvars.ContextVar = contextvars.ContextVar("omlx_steer_phase", default=None)

_lock = threading.Lock()
_by_request: "OrderedDict[str, Any]" = OrderedDict()
_by_uid: "OrderedDict[tuple[int, int], Any]" = OrderedDict()
_MAX = 8192

stats = {"ambiguous_main": 0, "ambiguous_draft": 0, "steered_forwards": 0}


class AmbiguousRows(RuntimeError):
    """A forward mixes profiles but its rows cannot be matched to requests."""


def note_request(request_id: str, profile) -> None:
    if profile is None:
        return
    with _lock:
        _by_request[request_id] = profile
        while len(_by_request) > _MAX:
            _by_request.popitem(last=False)


def request_profile(request_id: str):
    with _lock:
        return _by_request.get(request_id)


def bind_uid(model, request_id: str, uid: int) -> None:
    with _lock:
        profile = _by_request.get(request_id)
        key = (id(model), int(uid))
        if profile is None:
            _by_uid.pop(key, None)
            return
        _by_uid[key] = profile
        while len(_by_uid) > _MAX:
            _by_uid.popitem(last=False)


def unbind_uid(model, uid: int) -> None:
    with _lock:
        _by_uid.pop((id(model), int(uid)), None)


def unbind_model(model) -> None:
    mid = id(model)
    with _lock:
        for key in [k for k in _by_uid if k[0] == mid]:
            del _by_uid[key]


def forget_request(request_id: str) -> None:
    with _lock:
        _by_request.pop(request_id, None)


def profiles_for_uids(model, uids: Sequence[int]) -> tuple:
    with _lock:
        return tuple(_by_uid.get((id(model), int(u))) for u in uids)


def any_steered_uid(model) -> bool:
    mid = id(model)
    with _lock:
        return any(k[0] == mid for k in _by_uid)


@contextlib.contextmanager
def scope(rows: Optional[tuple], phase: Optional[str] = None):
    token = ROWS.set(None if rows is None else tuple(rows))
    ptoken = PHASE.set(phase) if phase is not None else None
    try:
        yield
    finally:
        ROWS.reset(token)
        if ptoken is not None:
            PHASE.reset(ptoken)


@contextlib.contextmanager
def uid_scope(model, uids, phase: Optional[str] = None):
    if model is None or uids is None:
        yield
        return
    with scope(profiles_for_uids(model, list(uids)), phase):
        yield


def resolve_draft(n_rows: int, cache) -> tuple:
    """Row profiles of a DSpark stage forward; ``cache`` is that stage's cache."""
    mapping = DRAFT_CACHES.get()
    if mapping and n_rows == 1 and id(cache) in mapping:
        return (mapping[id(cache)],)
    return resolve(n_rows, draft=True)


def resolve(n_rows: int, *, draft: bool = False) -> tuple:
    """Row profiles for a forward of ``n_rows`` rows.

    Uniform scopes broadcast to any row count. A mixed scope must match the
    row count exactly; otherwise the main model raises (the MTP step falls
    back to the standard step on it) and a draft runs stock.
    """
    rows = ROWS.get()
    if rows is None or not rows:
        return (None,) * n_rows
    first = rows[0]
    if all(r is first for r in rows):
        return (first,) * n_rows
    if len(rows) == n_rows:
        return rows
    if draft:
        stats["ambiguous_draft"] += 1
        _debug_once("draft", n_rows, len(rows))
        return (None,) * n_rows
    stats["ambiguous_main"] += 1
    _debug_once("main", n_rows, len(rows))
    raise AmbiguousRows(f"forward has {n_rows} rows but its scope lists {len(rows)} mixed profiles")


_DEBUG = os.environ.get("OMLX_STEER_DEBUG") == "1"
_seen_debug: set = set()


def _debug_once(kind, n_rows, n_scope):
    if not _DEBUG:
        return
    import traceback

    key = (kind, n_rows, n_scope, tuple(f.name for f in traceback.extract_stack(limit=12)))
    if key in _seen_debug:
        return
    _seen_debug.add(key)
    logger.warning("omlx-steer: ambiguous %s forward rows=%d scope=%d\n%s", kind, n_rows, n_scope,
                   "".join(traceback.format_stack(limit=12)))
