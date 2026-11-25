"""DeepSeek-V4.1 adapter: project steering directions out of every residual writer.

V4.1 carries the residual as hc_mult = 4 hyper-connection streams. Each writer
(attention, the whole MoE block, Engram) produces one [dim] vector per token
that ``hc_post`` adds into all four streams with weights ``post_j``. Projecting
that vector before ``hc_post`` is exactly equivalent to orthogonalising the
writer's output weights, independent of the stream mixing.

Hook points (oMLX 0.8.3, ``omlx/patches/deepseek_v41``):
  * ``Block.sum_ranks`` - every path (generic, fused decode, batched decode,
    MTP verify) passes the attention output and the MoE output through it,
    in that order, right before they enter the streams.
  * ``Engram.__call__`` - adds ``gate_s * value`` to each stream; ``value`` is
    projected (copy of the upstream body with one inserted line).
  * ``DSparkBlock.__call__`` - the three draft (MTP) stages call ``hc_post``
    directly; copy of the upstream body with projections inserted.
  * ``LanguageModel._forward`` / ``_forward_decode_batched`` /
    ``_forward_verify_batched`` and ``DeepseekV41Cache.extract`` - track which
    batch row the current tensors belong to.

With no steered row in a forward, every hook calls straight through to the
upstream code: stock output is bit for bit unchanged.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .. import rows as R
from ..kernel import project

logger = logging.getLogger("omlx_steer")

MODEL_TYPE = "deepseek_v41"

# Functions whose source is pinned (see pins.json). The copies below mirror
# Engram.__call__ and DSparkBlock.__call__ of this exact source.
PINNED = {
    "omlx.patches.deepseek_v41.language": [
        "Block.__call__",
        "Block.sum_ranks",
        "LanguageModel._forward",
        "LanguageModel._forward_decode_batched",
        "LanguageModel._forward_verify_batched",
        "hc_post",
    ],
    "omlx.patches.deepseek_v41.decode_fused": [
        "decode_block",
        "decode_block_nb",
        "decode_block_verify",
        "decode_block_verify_nb",
    ],
    "omlx.patches.deepseek_v41.engram": ["Engram.__call__"],
    "omlx.patches.deepseek_v41.dspark": ["DSparkBlock.__call__", "proposal_hidden"],
    "omlx.patches.deepseek_v41.cache": ["DeepseekV41Cache.extract"],
}

FWD: contextvars.ContextVar = contextvars.ContextVar("omlx_steer_fwd", default=None)


@dataclass
class _Forward:
    rows: tuple
    mode: str = "rows"  # "rows": one row at a time; "positional": [n, ...] tensors
    current: int = 0
    offset: int = 0
    n: int = 1
    cursor: int = 0
    calls: dict = field(default_factory=dict)
    capture: dict = field(default_factory=dict)

    def active_rows(self):
        if self.mode == "rows":
            return (self.rows[self.current],)
        return self.rows[self.offset : self.offset + self.n]


def _apply(st: _Forward, layer: int, y, writer: str):
    rows = st.active_rows()
    dirs, alphas, impl = [], [], "kernel"
    for p in rows:
        d = None
        if p is not None and writer in p.writers:
            d = p.direction(layer)
            impl = p.impl
        dirs.append(d)
        alphas.append(p.alpha_for(layer) if d is not None else 0.0)
    if all(d is None for d in dirs):
        return y
    return project(y, dirs, alphas, impl=impl)


def _capture_targets(st: _Forward):
    """(local row, global row, profile) of capture-profile rows among the active rows."""
    if st.mode == "rows":
        p = st.rows[st.current]
        return [(0, st.current, p)] if p is not None and p.capture is not None else []
    return [
        (i, st.offset + i, p)
        for i, p in enumerate(st.rows[st.offset : st.offset + st.n])
        if p is not None and p.capture is not None
    ]


def _record(st: _Forward, key: str, layer: int, y) -> None:
    targets = _capture_targets(st)
    if not targets:
        return
    n = 1 if st.mode == "rows" else st.n
    view = y.reshape(n, -1, y.shape[-1])
    for local, glob, p in targets:
        st.capture.setdefault(glob, {}).setdefault(key, {})[layer] = view[local, -1].astype(mx.float32)


def install(modules) -> None:
    """Patch the vendor modules. ``modules`` maps module name -> module object."""
    language = modules["omlx.patches.deepseek_v41.language"]
    engram_mod = modules["omlx.patches.deepseek_v41.engram"]
    dspark_mod = modules["omlx.patches.deepseek_v41.dspark"]
    cache_mod = modules["omlx.patches.deepseek_v41.cache"]
    try:
        from omlx.patches.mlx_lm_mtp.batch_generator import _MtpStepFallback
    except Exception:  # pragma: no cover - MTP package missing
        _MtpStepFallback = RuntimeError

    Block = language.Block
    LanguageModel = language.LanguageModel
    hc_post = language.hc_post
    hc_pre = language.hc_pre
    hc_mixes = language.hc_mixes

    # -- Block.sum_ranks: attention and MoE outputs of every path ------------
    orig_sum_ranks = Block.sum_ranks

    def sum_ranks(self, x):
        y = orig_sum_ranks(self, x)
        st = FWD.get()
        if st is None:
            return y
        layer = self.attn._layer
        count = st.calls.get(layer, 0)
        st.calls[layer] = count + 1
        writer = "attn" if count % 2 == 0 else "ffn"
        _record(st, writer, layer, y)
        return _apply(st, layer, y, writer)

    Block.sum_ranks = sum_ranks

    # -- Block.__call__: block outputs for capture only ----------------------
    orig_block_call = Block.__call__

    def block_call(self, h, pre, cache, shared, start, image_mask, ced_tail=None):
        out = orig_block_call(self, h, pre, cache, shared, start, image_mask, ced_tail=ced_tail)
        st = FWD.get()
        if st is not None and _capture_targets(st):
            _record_block(st, self.attn._layer, *out)
        return out

    Block.__call__ = block_call

    def _record_block(st, layer, h2, fp):
        _record(st, "block_mean", layer, mx.mean(h2.astype(mx.float32), axis=-2))
        _record(st, "block_pre", layer, hc_pre(h2, fp))

    # The batched paths call the fused block functions directly, not Block.__call__.
    fused = modules["omlx.patches.deepseek_v41.decode_fused"]
    for name in ("decode_block_nb", "decode_block_verify_nb"):
        original = getattr(fused, name)

        def wrapped(blk, h, pre, *args, _original=original, **kwargs):
            out = _original(blk, h, pre, *args, **kwargs)
            st = FWD.get()
            if st is not None and _capture_targets(st):
                _record_block(st, blk.attn._layer, *out)
            return out

        wrapped.__wrapped__ = original
        setattr(fused, name, wrapped)

    # -- Engram: project the value written into every stream -----------------
    _engram_call = _make_engram_call(engram_mod)
    orig_engram_call = engram_mod.Engram.__call__

    def engram_call(self, h, ids, image_mask=None):
        st = FWD.get()
        if st is None or all(p is None for p in st.active_rows()):
            return orig_engram_call(self, h, ids, image_mask)
        layer = getattr(self, "_steer_layer", None)
        if layer is None:
            raise RuntimeError("omlx-steer: Engram layer id unknown")
        return _engram_call(self, h, ids, image_mask, lambda v: _apply(st, layer, v, "engram"))

    engram_mod.Engram.__call__ = engram_call

    # -- DSpark draft stages ---------------------------------------------------
    DSparkBlock = dspark_mod.DSparkBlock
    orig_dspark_call = DSparkBlock.__call__

    def dspark_call(self, h, pre, cache):
        rows = R.resolve_draft(h.shape[0], cache)
        if all(p is None for p in rows):
            return orig_dspark_call(self, h, pre, cache)
        c = self._config
        stage = self.attn._layer - c.n_layers

        def proj(y, writer):
            dirs, alphas, impl = [], [], "kernel"
            for p in rows:
                d = p.mtp_direction(c.n_layers, stage) if p is not None and writer in p.writers else None
                dirs.append(d)
                alphas.append(p.alpha_for(c.n_layers + stage) if d is not None else 0.0)
                if d is not None:
                    impl = p.impl
            return project(y, dirs, alphas, impl=impl)

        # Upstream DSparkBlock.__call__ with the two writer outputs projected.
        ap, ao, ac = hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self._config)
        h = hc_post(proj(self.attn(self.attn_norm(hc_pre(h, pre)), cache), "attn"), h, ao, ac)
        fp, fo, fc = hc_mixes(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self._config)
        h = hc_post(proj(self.ffn(self.ffn_norm(hc_pre(h, ap)), None), "ffn"), h, fo, fc)
        return h, fp

    DSparkBlock.__call__ = dspark_call

    # -- row tracking -----------------------------------------------------------
    Cache = cache_mod.DeepseekV41Cache
    orig_extract = Cache.extract

    def extract(self, idx):
        st = FWD.get()
        if st is not None and st.mode == "rows":
            st.current = idx % len(st.rows) if idx < 0 else idx
        return orig_extract(self, idx)

    Cache.extract = extract

    orig_forward = LanguageModel._forward

    def _forward(self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs):
        if FWD.get() is not None:  # nested (e.g. replay): keep the outer state
            return orig_forward(self, input_ids, cache, inputs_embeds, token_types, **kwargs)
        try:
            rows = R.resolve(int(input_ids.shape[0]))
        except R.AmbiguousRows as exc:
            raise _MtpStepFallback(str(exc)) from exc
        if all(p is None for p in rows):
            return orig_forward(self, input_ids, cache, inputs_embeds, token_types, **kwargs)
        R.stats["steered_forwards"] += 1
        if not getattr(self, "_steer_engram_ids", False):
            for i, layer in enumerate(self.layers):
                if "engram" in layer:
                    object.__setattr__(layer.engram, "_steer_layer", i)
            object.__setattr__(self, "_steer_engram_ids", True)
        st = _Forward(rows=rows)
        token = FWD.set(st)
        try:
            out = orig_forward(self, input_ids, cache, inputs_embeds, token_types, **kwargs)
        finally:
            FWD.reset(token)
        _finish_capture(st, out, self._config.vocab_size)
        return out

    LanguageModel._forward = _forward

    orig_fdb = LanguageModel._forward_decode_batched

    def _forward_decode_batched(self, input_ids, cache, skip_logits=False):
        st = FWD.get()
        if st is None:
            return orig_fdb(self, input_ids, cache, skip_logits)
        b = int(input_ids.shape[0])
        if st.mode == "positional":  # a MAX_BATCH chunk of an outer batched call
            off = st.cursor
            st.cursor += b
        else:
            off = 0
        saved = (st.mode, st.offset, st.n, st.cursor)
        st.mode, st.offset, st.n, st.cursor = "positional", off, b, off
        try:
            return orig_fdb(self, input_ids, cache, skip_logits)
        finally:
            st.mode, st.offset, st.n, st.cursor = saved

    LanguageModel._forward_decode_batched = _forward_decode_batched

    orig_fvb = LanguageModel._forward_verify_batched

    def _forward_verify_batched(self, input_ids, cache, verify_states, capture, skip_logits=False):
        st = FWD.get()
        if st is None:
            return orig_fvb(self, input_ids, cache, verify_states, capture, skip_logits)
        saved = (st.mode, st.offset, st.n, st.cursor)
        st.mode, st.offset, st.n, st.cursor = "positional", 0, int(input_ids.shape[0]), 0
        try:
            return orig_fvb(self, input_ids, cache, verify_states, capture, skip_logits)
        finally:
            st.mode, st.offset, st.n, st.cursor = saved

    LanguageModel._forward_verify_batched = _forward_verify_batched
    logger.info("omlx-steer: deepseek_v41 hooks installed")


def _make_engram_call(engram_mod):
    """Engram.__call__ of oMLX 0.8.3 with ``value`` passed through ``proj``."""
    import math

    import numpy as np

    _project_each = engram_mod._project_each
    dedupe_tuple_ids = engram_mod.dedupe_tuple_ids
    DECODE_ROWS = engram_mod.DECODE_ROWS

    def call(self, h, ids, image_mask, proj):
        c = self._config
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_host, inverse = dedupe_tuple_ids(ids_np)
        rows = self.embed(mx.array(ids_host)).flatten(-2)
        count = math.prod(ids_np.shape[:-1]) if ids_np.ndim >= 2 else 1
        requests = ids_np.shape[0] if ids_np.ndim >= 2 else 1
        if 1 < count <= 8 or (requests > 1 and count <= DECODE_ROWS):
            kv = _project_each(self.wkv, rows.reshape(count, -1))
            if kv is None:
                flat = rows.reshape(count, 1, -1)
                kv = mx.concatenate([self.wkv(flat[r : r + 1]) for r in range(count)])
            kv = kv.reshape(*rows.shape[:2], -1)
        else:
            kv = self.wkv(rows)
        if inverse is not None:
            kv = kv[mx.array(inverse, dtype=mx.int32)]
        key, value = mx.split(kv, [c.dim * c.hc_mult], axis=-1)
        value = proj(value)  # omlx-steer: the only change
        key = key.astype(mx.float32).reshape(*h.shape)
        x = h.astype(mx.float32)
        inv = mx.rsqrt(mx.mean(x * x, -1) + c.norm_eps) * mx.rsqrt(mx.mean(key * key, -1) + c.norm_eps)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        dot = mx.sum(x * weight * key, -1) * inv * c.dim**-0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)
        if image_mask is not None:
            gate = mx.where(image_mask[..., None], 0, gate)
        return (x + gate[..., None] * value.astype(mx.float32)[..., None, :]).astype(h.dtype)

    return call


def _finish_capture(st: _Forward, out, vocab: int) -> None:
    """Write each capture row once, from its first decode-phase forward.

    oMLX prefills prompt tokens 0..N-2 and hands the last prompt token to the
    first generation step, so the first forward of the decode phase is the one
    at the last gen-prompt token. Later decode forwards (the generated token,
    MTP warm-up) must not overwrite it.
    """
    if not st.capture or R.PHASE.get() != "decode":
        return
    logits = out[0] if isinstance(out, tuple) else out
    has_logits = logits is not None and logits.ndim == 3 and logits.shape[-1] == vocab
    for glob, keys in st.capture.items():
        p = st.rows[glob]
        target = Path(p.capture["dir"]) / f"{p.capture.get('tag') or p.sha}.npz"
        if target.exists():
            continue
        record = {}
        for key, layers in keys.items():
            order = sorted(layers)
            record[key] = mx.stack([layers[i] for i in order])
            record[key + "_layers"] = mx.array(order, mx.int32)
        if p.dump_last_logprobs and has_logits:
            last = logits[glob, -1].astype(mx.float32)
            record["logprobs"] = (last - mx.logsumexp(last)).astype(mx.float16)
        target.parent.mkdir(parents=True, exist_ok=True)
        mx.eval(list(record.values()))
        tmp = target.with_name(target.stem + ".tmp.npz")
        mx.savez(str(tmp), **record)
        tmp.replace(target)
