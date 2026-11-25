# omlx-steer design

## Goal

Serve several behaviour variants of one model from a single copy of its
weights. A variant is a *steering profile*: per-layer directions that are
projected out of the residual writers at runtime (directional ablation), with
a per-layer strength. Clients pick a variant through the OpenAI `model` field.
Weights in memory never change, the stock model stays bit-for-bit stock, and
requests of different variants share one continuous batch.

Directional ablation of a writer `W` (`W' = (I - r rᵀ) W`) gives the output
`W'x = Wx - r (rᵀ W x)`. Projecting the writer's output at runtime is
mathematically the same operation as orthogonalising its weights, without
requantising a 4-bit checkpoint, and the projection can differ per request.

## Request path

```
HTTP body {"model": "<alias>"}            middleware.py
  -> profile resolved (file on disk, sha)  profiles.py
  -> body rewritten to base model id; profile in request context
  -> oMLX resolves / leases the loaded engine exactly as for the base model
  -> Request.__init__ pins the profile      integration.patch_request
  -> scheduler binds request -> uid         integration.patch_priming (bind_uid)
  -> every forward runs in a row scope      rows.py
       prefill: the request being prefilled
       decode: GenerationBatch.uids
       MTP helpers: their own row batch / subset
  -> model hooks read the row profiles      adapters/deepseek_v41.py
```

Because the engine pool only ever sees the base model id, there is no second
engine entry, no memory accounting change, no reload and no eviction. oMLX's
own exposed profiles are deliberately not used: they carry engine-construction
settings and a signature change unloads and reloads the model.

## Per-row semantics

A forward's rows come from a scope listing one profile (or `None`) per row.

* All rows stock: every hook calls straight through; output is bit-identical.
* All rows one profile: the profile applies to the whole tensor.
* Mixed: batched paths use a positional per-row mask (`[B]`, or `[B*R]` for
  MTP verify blocks); the generic per-row path learns the current row from
  `DeepseekV41Cache.extract(idx)`. Stock rows inside a mixed batch are copied
  untouched by the kernel.
* A mixed scope whose length does not match the forward's rows raises
  `_MtpStepFallback`, so the MTP step falls back to the standard step; it
  never silently runs stock. DSpark drafts of shared draft jobs are matched to
  their request through the request's own draft cache.

oMLX itself is not batch-invariant (a request alone and in a batch can differ
slightly), so isolation is defined against a batch of the same composition.

## Prefix / SSD cache

A steered KV differs from the first token on. The profile sha (file content +
directions file sha + any sweep override) is appended to the request's cache
`extra_keys` from token 0, through the same `Request` properties that carry
VLM image hashes, so prefix cache, paged SSD cache, boundary snapshots and MTP
priming sidecars are all keyed by it. Stock requests produce exactly the keys
they produced before.

## Hook points (DeepSeek-V4.1, oMLX 0.8.3)

The residual is `hc_mult = 4` hyper-connection streams. Each writer produces
one `[dim]` vector per token and `hc_post` adds `post_j * x` to stream `j`;
projecting `x` before `hc_post` equals weight orthogonalisation regardless of
stream mixing.

| Writer | Where | Hook |
|---|---|---|
| attention (`wo_a` grouped + `wo_b`) | generic, fused decode, batched decode, MTP verify | `Block.sum_ranks` (1st call) |
| MoE (shared + routed) | same four paths | `Block.sum_ranks` (2nd call) |
| Engram value | `Engram.__call__` | copy with one inserted line |
| DSpark stages 40-42 | `DSparkBlock.__call__` | copy with projections |

All four main-model paths pass both writer outputs through `blk.sum_ranks`
(a no-op without tensor parallelism) right before they enter the streams.

## Fail-closed upstream drift

`pins.json` holds the sha256 of the source of every upstream function the
hooks rely on. A module is patched only if all its pins match; any mismatch
keeps the stock path and makes every steering alias answer 503 (oMLX's
`model_fallback` would otherwise serve the stock model under the alias name).

## Kernel

`kernel.py` does `y - alpha * Rᵀ (R y)` for one token per threadgroup in one
dispatch, `R` up to 4 orthonormal rows in fp32, alpha per row (0 = untouched
copy). `impl: "mlx"` in a profile selects the MLX-ops reference.

## Capture

A capture profile records, at the last processed token of each forward, the
attention and MoE outputs, the mean of the four streams after each block and
`hc_pre(h, pre)`, plus optionally the full next-token log-probabilities.
Captures run through the ordinary scheduler (`max_tokens 1`), so no second
model instance is needed.
