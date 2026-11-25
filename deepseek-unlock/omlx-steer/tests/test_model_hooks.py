"""Model-level checks on the tiny DeepSeek-V4.1 checkpoint (tools/make_tiny_v41.py).

Run twice:
    python tests/test_model_hooks.py MODEL_DIR OUT.npz --baseline   # stock oMLX, no hooks
    python tests/test_model_hooks.py MODEL_DIR OUT.npz              # hooks installed

The second run compares against the baseline and checks:
  1. stock forwards with hooks installed are bit-identical to stock oMLX
     (prefill, decode B=1, batched decode B=2);
  2. a steered row with alpha 0 (Engram copy, row tracking, kernels on) is
     bit-identical to stock;
  3. in a mixed batch the stock row is bit-identical to stock, and the steered
     row matches the same request decoded alone;
  4. writer outputs of a steered row have no component along the directions;
  5. the Metal kernel matches the MLX reference.
"""

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL, OUT = sys.argv[1], sys.argv[2]
BASELINE = "--baseline" in sys.argv

if not BASELINE:
    tmp = Path(tempfile.mkdtemp())
    import omlx_steer

    state = omlx_steer.install(tmp)

from omlx.patches.deepseek_v41 import apply_patch  # noqa: E402

apply_patch()
from omlx.patches.deepseek_v41 import loading  # noqa: E402
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache  # noqa: E402

model, _ = loading.load(MODEL)
lm = model.language_model
cfg = lm._config
P1 = [1, 400, 500, 600, 700, 800, 900, 1000] * 5 + [42]
P2 = [1, 1200, 1300, 77, 1500, 1600, 1700] * 6 + [43]


def prefill(ids):
    cache = lm.make_cache()
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    return logits[:, -1], cache


def decode(tok, cache):
    out = model(mx.array([[tok]]), cache=cache)
    mx.eval(out)
    return out[:, -1]


def batched(toks, caches):
    merged = [DeepseekV41Cache.merge([c[i] for c in caches]) for i in range(len(caches[0]))]
    out = model(mx.array([[t] for t in toks]), cache=merged)
    mx.eval(out)
    return out[:, -1]


def run_all():
    res = {}
    l1, c1 = prefill(P1)
    l2, c2 = prefill(P2)
    res["prefill1"], res["prefill2"] = l1, l2
    t1, t2 = int(mx.argmax(l1)), int(mx.argmax(l2))
    res["decode1"] = decode(t1, c1)
    l1b, c1b = prefill(P1)
    l2b, c2b = prefill(P2)
    res["batched"] = batched([t1, t2], [c1b, c2b])
    return res


if BASELINE:
    res = run_all()
    mx.savez(OUT, **res)
    print("baseline saved", {k: v.shape for k, v in res.items()})
    sys.exit(0)

from omlx_steer import rows as R  # noqa: E402
from omlx_steer.kernel import _reference, project  # noqa: E402

base = dict(mx.load(OUT))
fails = []


def same(name, a, b):
    ok = bool(mx.array_equal(a, b))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" max|d|={float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))):.3g}"))
    if not ok:
        fails.append(name)


print("ready:", state.ready("deepseek_v41"), state.failed, sorted(state.patched))

# 1. stock with hooks
res = run_all()
for k in base:
    same(f"stock-with-hooks {k}", res[k], base[k])

# profiles on disk
rng = np.random.default_rng(1)
n_total = cfg.n_layers + cfg.n_mtp_layers
dirs = np.zeros((n_total, 2, cfg.dim), np.float32)
for layer in range(n_total):
    q, _ = np.linalg.qr(rng.standard_normal((cfg.dim, 2)))
    dirs[layer] = q.T
(tmp / "dirs").mkdir()
np.savez(tmp / "dirs" / "t.npz", r=dirs)
for alias, alpha in (("tiny-steer", 1.0), ("tiny-zero", 0.0)):
    (tmp / f"{alias}.json").write_text(json.dumps({
        "alias": alias, "base_model": "tiny", "model_type": "deepseek_v41",
        "directions": "dirs/t.npz", "layers": "all", "alpha": alpha}))
steer = state.store.resolve("tiny-steer")
zero = state.store.resolve("tiny-zero")

# 2. alpha 0 == stock
with R.scope((zero,)):
    res0 = run_all()
for k in base:
    same(f"alpha0 {k}", res0[k], base[k])

# 3. mixed batch
with R.scope((steer,)):
    ls1, cs1 = prefill(P1)
    s_tok = int(mx.argmax(ls1))
    solo = decode(s_tok, cs1)
    ls1b, cs1b = prefill(P1)
_, c2m = prefill(P2)
t2 = int(mx.argmax(base["prefill2"]))
with R.scope((steer, None)):
    mixed = batched([s_tok, t2], [cs1b, c2m])
same("mixed: stock row == stock", mixed[1], base["batched"][1])
d = float(mx.max(mx.abs(mixed[0].astype(mx.float32) - solo[0].astype(mx.float32))))
print(f"{'PASS' if d < 1e-2 else 'FAIL'} mixed: steered row vs alone, max|d|={d:.3g}")
if d >= 1e-2:
    fails.append("mixed steered")
changed = float(mx.max(mx.abs(ls1.astype(mx.float32) - base["prefill1"].astype(mx.float32))))
print(f"{'PASS' if changed > 0 else 'FAIL'} steering changes logits, max|d|={changed:.3g}")

# 4. writer outputs orthogonal to directions (capture of a steered row)
from omlx_steer.adapters import deepseek_v41 as A  # noqa: E402

seen = []
orig_apply = A._apply


def spy(st, layer, y, writer):
    out = orig_apply(st, layer, y, writer)
    p = st.active_rows()[0]
    if p is not None and p.direction(layer) is not None:
        coef = out.astype(mx.float32).reshape(-1, out.shape[-1]) @ p.direction(layer).T
        ref = y.astype(mx.float32).reshape(-1, y.shape[-1]) @ p.direction(layer).T
        seen.append(float(mx.max(mx.abs(coef))) / max(1e-9, float(mx.max(mx.abs(ref)))))
    return out


A._apply = spy
with R.scope((steer,)):
    prefill(P1)
A._apply = orig_apply
worst = max(seen) if seen else 1.0
print(f"{'PASS' if worst < 2e-2 else 'FAIL'} residual component after projection / before: max {worst:.3g} over {len(seen)} writer calls")
if worst >= 2e-2:
    fails.append("orthogonality")

# 5. kernel vs reference
y = (mx.random.normal((3, 5, cfg.dim)) * 3).astype(mx.bfloat16)
ds = [mx.array(dirs[3]), None, mx.array(dirs[4][:1])]
k = project(y, ds, [1.0, 0.0, 0.7])
r = _reference(y, ds, [1.0, 0.0, 0.7])
kd = float(mx.max(mx.abs(k.astype(mx.float32) - r.astype(mx.float32))))
same("kernel stock row untouched", k[1], y[1])
print(f"{'PASS' if kd < 0.05 else 'FAIL'} kernel vs reference max|d|={kd:.3g}")
print("stats", R.stats)
print("RESULT", "OK" if not fails else f"FAILED {fails}")
