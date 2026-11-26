# omlx-steer

Per-request activation steering profiles for [oMLX](https://github.com/jundot/omlx):
several behaviour variants of one loaded model, chosen per request through the
`model` field, from a single copy of the weights.

A profile projects per-layer directions out of every residual writer at
runtime (directional ablation, Arditi et al. 2024). It is exactly equivalent
to orthogonalising the writer weights, but needs no re-quantised checkpoint,
no reload, and can differ between rows of the same continuous batch.

Status: works with oMLX 0.8.4 (ported from 0.8.3 on 2026-11-26) and the DeepSeek-V4.1 model family
(hyper-connections, Engram, DSpark MTP). Other architectures need an adapter.

## Guarantees

* Stock requests are bit-for-bit unchanged (verified on the release
  checkpoint with greedy outputs and top-20 logprobs).
* Steering is per batch row, never a global flag.
* The profile is part of the prefix / SSD cache key.
* Virtual model ids never reach the engine pool: no second engine, no reload.
* Upstream source changes disable steering (503 on the aliases) instead of
  running it wrongly.

## Use

```
deploy/install.sh [--sweep]   # edits the launchd job, saves the original
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/local.omlx.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.omlx.plist
deploy/uninstall.sh [--purge] # restores the launchd job exactly
```

A profile is a file `~/omlx-steer-data/profiles/<alias>.json`:

```json
{"alias": "mymodel-steered", "base_model": "mymodel", "model_type": "deepseek_v41",
 "directions": "dirs/both.npz", "layers": ["14-39"], "alpha": 1.0,
 "writers": ["attn", "ffn", "engram"], "mtp": "last"}
```

`directions` is an `.npz` with `r: float32 [layers + mtp_layers, k, hidden]`
(orthonormal rows per layer, k <= 4). Profiles are re-read on change; the
weights are untouched. `GET /v1/omlx-steer/status` reports readiness.

Responses of steered requests carry `X-Omlx-Steer: <alias>@<sha>`.

Sweep mode (`OMLX_STEER_SWEEP=1`) enables profiles marked `"sweep": true` and
per-request overrides in the body field `omlx_steer`. Without it any request
with `omlx_steer` gets 400.

## Tools

* `tools/make_tiny_v41.py` - random-weight 6-layer DeepSeek-V4.1 in oMLX's
  converted format (MXFP8 / MXFP4, Engram, 3 DSpark stages) for tests.
* `tools/capture.py`, `tools/directions.py` - activations through a live
  server, then refusal / domain directions with biprojection.
* `tools/sweep.py` - layer / alpha screening via overrides.
* `tools/golden.py` - greedy reference outputs to prove a deploy changed nothing.
* `tests/` - model-level and server-level checks against the tiny checkpoint.

See [docs/DESIGN.md](docs/DESIGN.md).

## License

Apache-2.0, as oMLX.
