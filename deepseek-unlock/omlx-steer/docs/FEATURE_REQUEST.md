# Feature request: per-request activation steering profiles (virtual models over one loaded model)

## Summary

Allow several behaviour variants of one loaded model to be served side by
side, selected per request through the `model` field, without loading a second
copy of the weights. A variant ("steering profile") applies a runtime
directional projection to the residual writers of chosen layers:
`out -= alpha * R^T (R out)`. This is the runtime form of directional
ablation / weight orthogonalisation (Arditi et al., 2024; "Surgical, Cheap,
and Flexible", arXiv 2410.03415): numerically it is the same as editing the
writer weights, but it needs no re-quantised checkpoint, no reload, and it can
differ between rows of one continuous batch.

Typical uses: an abliterated or domain-specific variant next to the stock
model on a machine that can only hold one copy (e.g. a 400 GB checkpoint on a
512 GB Mac), A/B evaluation, and cheap sweeps (layer range / strength become
request parameters instead of hours of re-quantisation).

## Reference implementation

[omlx-steer](https://github.com/ageev/AI/tree/main/deepseek-unlock/omlx-steer) is an out-of-tree plugin for oMLX 0.8.4 (first written for 0.8.3) with an adapter for
DeepSeek-V4.1. It patches at import time and pins the source of every
upstream function it relies on, so it disables itself (503 on the virtual ids)
when oMLX changes. It is meant to show the integration points, not as the
proposed final shape.

Measured on DeepSeek-V4.1-Flash oQ4e (Mac Studio M5 Ultra, 512 GB):

* stock requests are bit-for-bit unchanged with the plugin installed (greedy
  outputs and top-20 logprobs over 18 prompts incl. thinking, tool calls,
  20k context);
* decode speed with a 32-layer, two-direction profile: 97.2% of stock at one
  request (50.4 vs 51.8 tok/s), 97.5% at eight concurrent requests;
* MTP (DSpark) acceptance 0.635 steered vs 0.642 stock;
* no engine reload, eviction or second engine entry while serving stock and
  steered requests in one continuous batch (rows stay isolated: a stock row in
  a mixed batch is bit-identical to the same row in an all-stock batch).

## What a native implementation needs

1. **Virtual model ids that resolve to a loaded engine without
   `runtime_settings`.** Exposed profiles today carry engine-construction
   settings, and a signature change unloads and reloads the model
   (`engine_pool.get_engine`). Steering profiles must never do that.
2. **A per-request field that survives to the batch row.** The plugin binds
   request -> uid at `prompt_priming.bind_uid` and opens a row scope around
   prefill (`Scheduler._forward_prefill_chunk`, `_do_external_prefill`),
   `PromptProcessingBatch.prompt`, `GenerationBatch.__init__/next` and the MTP
   helpers that work on row subsets (`fused_batch._verify_rows_together`,
   `_advance_group`, `_dspark_draft_jobs`). A native design could carry the
   profile on the per-uid row state the scheduler already keeps for samplers
   (`_uid_row_registry`).
3. **The profile in the prefix / SSD cache key.** Appending a salt to the
   request's cache `extra_keys` from token 0 (the plugin reuses the
   `vlm_extra_keys_*` properties) covers prefix cache, paged SSD cache,
   boundary snapshots and MTP priming sidecars.
4. **Model hook points.** For DeepSeek-V4.1 a single hook on
   `Block.sum_ranks` sees both writer outputs (attention, whole MoE) in all
   four forward paths (generic, fused decode, batched decode, MTP verify)
   before `hc_post` adds them to the hyper-connection streams. Engram and the
   DSpark stages need small hooks of their own. A generic `writer_output`
   hook per architecture would make this reusable.
5. **Row masks in batched kernels.** The plugin applies one fused Metal kernel
   per writer with a per-row alpha (0 = untouched copy), so stock rows in a
   mixed batch are bit-identical to an all-stock batch of the same shape.
6. **Optional capture mode** for direction finding inside the serving
   process (last-token writer / block outputs and next-token logprobs), so no
   second model instance is needed.

## Not proposed

Shipping refusal directions or any particular profile. The feature is
model-agnostic infrastructure; profiles are user files.
