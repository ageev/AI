"""Server-side wiring: request -> profile -> uid -> batch rows, plus cache keys."""

from __future__ import annotations

import functools
import inspect
import logging

from . import rows as R

logger = logging.getLogger("omlx_steer")

PINNED = {
    "omlx.request": [
        "Request.vlm_extra_keys_for_cache",
        "Request.vlm_extra_key_token_start_for_cache",
        "Request.vlm_extra_key_ranges_for_cache",
    ],
    "omlx.patches.mlx_lm_mtp.prompt_priming": ["bind_uid"],
    "omlx.scheduler": [
        "_unregister_uid_row",
        "_unregister_uid_rows_for_model",
        "Scheduler._forward_prefill_chunk",
        "Scheduler._do_external_prefill",
        "Scheduler._forward_packed_prefill",
        "Scheduler._insert_prefilled_request",
        "Scheduler._schedule_waiting",
        "_patched_ppb_prompt",
    ],
    "omlx.patches.mlx_lm_mtp.batch_generator": ["apply", "_make_row_batch", "_dspark_draft_jobs"],
    "omlx.patches.mlx_lm_mtp.fused_batch": ["_verify_rows_together", "_advance_group"],
}


# -- omlx.request ------------------------------------------------------------
def patch_request(mod) -> None:
    Request = mod.Request
    orig_init = Request.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        profile = R.REQUEST_PROFILE.get()
        object.__setattr__(self, "_omlx_steer", profile)
        if profile is not None:
            R.note_request(self.request_id, profile)
            if profile.capture is not None:
                self.skip_cache_store = True  # probe KV must not enter the shared cache

    Request.__init__ = __init__

    keys_prop = Request.vlm_extra_keys_for_cache
    start_prop = Request.vlm_extra_key_token_start_for_cache
    ranges_prop = Request.vlm_extra_key_ranges_for_cache

    def _salt(self):
        p = getattr(self, "_omlx_steer", None)
        return None if p is None else p.salt

    def extra_keys(self):
        base = keys_prop.fget(self)
        salt = _salt(self)
        if salt is None:
            return base
        return tuple(base or ()) + (salt,)

    def token_start(self):
        # A steered KV differs from the first token on, so the salt keys every block.
        return 0 if _salt(self) is not None else start_prop.fget(self)

    def key_ranges(self):
        base = ranges_prop.fget(self)
        salt = _salt(self)
        if salt is None or not base:
            return base
        out = [(start, tuple(keys) + (salt,)) for start, keys in base]
        if out[0][0] > 0:
            out.insert(0, (0, (salt,)))
        return out

    Request.vlm_extra_keys_for_cache = property(extra_keys)
    Request.vlm_extra_key_token_start_for_cache = property(token_start)
    Request.vlm_extra_key_ranges_for_cache = property(key_ranges)


# -- omlx.patches.mlx_lm_mtp.prompt_priming ------------------------------------
def patch_priming(mod) -> None:
    orig = mod.bind_uid

    @functools.wraps(orig)
    def bind_uid(model, request_id, uid):
        R.bind_uid(model, request_id, uid)
        return orig(model, request_id, uid)

    mod.bind_uid = bind_uid


# -- omlx.scheduler --------------------------------------------------------------
def patch_scheduler(mod) -> None:
    orig_unreg = mod._unregister_uid_row
    orig_unreg_all = mod._unregister_uid_rows_for_model

    def _unregister_uid_row(model, uid):
        R.unbind_uid(model, uid)
        return orig_unreg(model, uid)

    def _unregister_uid_rows_for_model(model):
        R.unbind_model(model)
        return orig_unreg_all(model)

    mod._unregister_uid_row = _unregister_uid_row
    mod._unregister_uid_rows_for_model = _unregister_uid_rows_for_model

    S = mod.Scheduler

    def _profile(request):
        return getattr(request, "_omlx_steer", None)

    orig_chunk = S._forward_prefill_chunk

    @functools.wraps(orig_chunk)
    def _forward_prefill_chunk(self, plan):
        with R.scope((_profile(plan.state.request),), "prefill"):
            return orig_chunk(self, plan)

    orig_external = S._do_external_prefill

    @functools.wraps(orig_external)
    def _do_external_prefill(self, request, *args, **kwargs):
        with R.scope((_profile(request),), "prefill"):
            return orig_external(self, request, *args, **kwargs)

    orig_packed = S._forward_packed_prefill

    @functools.wraps(orig_packed)
    def _forward_packed_prefill(self, plans):
        profiles = tuple(_profile(p.state.request) for p in plans)
        if any(p is not None for p in profiles):
            # Packed rows share one sequence; their rows cannot be told apart.
            raise RuntimeError("omlx-steer: packed prefill of steered requests is unsupported")
        return orig_packed(self, plans)

    S._forward_prefill_chunk = _forward_prefill_chunk
    S._do_external_prefill = _do_external_prefill
    S._forward_packed_prefill = _forward_packed_prefill

    PPB = mod.PromptProcessingBatch
    orig_prompt = PPB.prompt

    @functools.wraps(orig_prompt)
    def prompt(self, tokens):
        with R.uid_scope(getattr(self, "model", None), getattr(self, "uids", None), "prefill"):
            return orig_prompt(self, tokens)

    PPB.prompt = prompt
    PPB._omlx_steer_prompt = prompt


# -- MTP batch generator -------------------------------------------------------
_ROW_PARAMS = ("gen_batch", "row", "batch")


def _scope_by_first_arg(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        uids = getattr(obj, "uids", None)
        model = getattr(obj, "model", None)
        if uids is None or model is None:
            return fn(obj, *args, **kwargs)
        with R.uid_scope(model, uids):
            return fn(obj, *args, **kwargs)

    wrapper._omlx_steer = True
    return wrapper


def _wrap_row_functions(mod) -> int:
    count = 0
    for name, fn in list(vars(mod).items()):
        if not inspect.isfunction(fn) or fn.__module__ != mod.__name__ or getattr(fn, "_omlx_steer", False):
            continue
        if inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn):
            continue
        params = list(inspect.signature(fn).parameters)
        if params and params[0] in _ROW_PARAMS:
            setattr(mod, name, _scope_by_first_arg(fn))
            count += 1
    return count


def _wrap_generation_batch() -> None:
    from mlx_lm.generate import GenerationBatch

    for name in ("__init__", "next"):
        current = getattr(GenerationBatch, name)
        if getattr(current, "_omlx_steer", False):
            continue
        if name == "__init__":

            def make(orig):
                @functools.wraps(orig)
                def __init__(self, model, uids, *args, **kwargs):
                    with R.uid_scope(model, uids, "decode"):
                        return orig(self, model, uids, *args, **kwargs)

                return __init__

        else:

            def make(orig):
                @functools.wraps(orig)
                def next(self, *args, **kwargs):
                    with R.uid_scope(getattr(self, "model", None), getattr(self, "uids", None), "decode"):
                        return orig(self, *args, **kwargs)

                return next

        wrapped = make(current)
        wrapped._omlx_steer = True
        setattr(GenerationBatch, name, wrapped)


def patch_batch_generator(mod) -> None:
    orig_apply = mod.apply

    @functools.wraps(orig_apply)
    def apply(*args, **kwargs):
        result = orig_apply(*args, **kwargs)
        _wrap_generation_batch()  # outermost, around oMLX's own MTP wrappers
        return result

    apply._omlx_steer = True
    orig_jobs = mod._dspark_draft_jobs

    @functools.wraps(orig_jobs)
    def _dspark_draft_jobs(jobs):
        mapping = {}
        for job in jobs:
            gen_batch, state = job[0], job[1]
            profiles = R.profiles_for_uids(gen_batch.model, list(getattr(gen_batch, "uids", ())))
            if len(profiles) == 1:  # stock rows are recorded too (None)
                for item in state.mtp_cache:
                    mapping[id(item)] = profiles[0]
        token = R.DRAFT_CACHES.set(mapping)
        try:
            return orig_jobs(jobs)
        finally:
            R.DRAFT_CACHES.reset(token)

    _dspark_draft_jobs._omlx_steer = True
    mod._dspark_draft_jobs = _dspark_draft_jobs
    n = _wrap_row_functions(mod)
    mod.apply = apply
    if getattr(__import__("mlx_lm.generate", fromlist=["GenerationBatch"]).GenerationBatch, "_omlx_mtp_patched", False):
        _wrap_generation_batch()
    logger.info("omlx-steer: %d MTP row helpers scoped", n)


def patch_fused_batch(mod) -> None:
    orig_verify = mod._verify_rows_together
    orig_group = mod._advance_group

    @functools.wraps(orig_verify)
    def _verify_rows_together(batch, batch_state, host, indices, replacements):
        uids = [batch.uids[i] for i in indices]
        with R.uid_scope(batch.model, uids):
            return orig_verify(batch, batch_state, host, indices, replacements)

    @functools.wraps(orig_group)
    def _advance_group(batch, depth, rows, replacements, **kwargs):
        uids = [state.uid for _, _, state in rows]
        with R.uid_scope(batch.model, uids):
            return orig_group(batch, depth, rows, replacements, **kwargs)

    _verify_rows_together._omlx_steer = True
    _advance_group._omlx_steer = True
    mod._verify_rows_together = _verify_rows_together
    mod._advance_group = _advance_group
    _wrap_row_functions(mod)
