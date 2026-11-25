"""End-to-end checks against a running oMLX + omlx-steer (the dev stand).

    python tests/test_server.py http://127.0.0.1:8099 PROFILES_DIR BASE_ALIAS SERVER_LOG

Creates test profiles in PROFILES_DIR (random directions), then checks model
listing, stock invariance, per-row isolation in concurrent batches, override
policy, cache salting, capture files and that the engine was never reloaded.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

URL, PROFILES, BASE, LOG = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
fails = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}")
    if not ok:
        fails.append(name)


def call(path, body=None, timeout=600):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read() or b"{}")


def chat(model, content, max_tokens=24, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 5, "stream": False}
    if extra:
        body.update(extra)
    return call("/v1/chat/completions", body)


def sig(resp):
    choice = resp["choices"][0]
    lp = choice.get("logprobs") or {}
    return choice["message"].get("content"), json.dumps(lp.get("content"), sort_keys=True)


# -- profiles ------------------------------------------------------------------
n_total, dim = 9, 1024
rng = np.random.default_rng(7)
dirs = np.zeros((n_total, 2, dim), np.float32)
for layer in range(n_total):
    q, _ = np.linalg.qr(rng.standard_normal((dim, 2)))
    dirs[layer] = q.T
(PROFILES / "dirs").mkdir(parents=True, exist_ok=True)
np.savez(PROFILES / "dirs" / "test.npz", r=dirs)
common = {"base_model": BASE, "model_type": "deepseek_v41", "directions": "dirs/test.npz", "layers": "all"}
for alias, extra in (
    ("tiny-steer", {"alpha": 1.0}),
    ("tiny-zero", {"alpha": 0.0}),
    ("tiny-sweep", {"alpha": 1.0, "sweep": True}),
    ("tiny-capture", {"alpha": 0.0, "sweep": True, "capture": {"dir": "capture"}, "dump_last_logprobs": True}),
):
    (PROFILES / f"{alias}.json").write_text(json.dumps(dict(common, alias=alias, **extra)))
log_start = LOG.stat().st_size

# -- listing / status --------------------------------------------------------------
_, _, models = call("/v1/models")
ids = [m["id"] for m in models.get("data", [])]
check("models lists aliases", {"tiny-steer", "tiny-zero"} <= set(ids), ids)
_, _, status = call("/v1/omlx-steer/status")
check("status ready", all(a["ready"] for a in status["aliases"].values()), status["aliases"])

# -- stock invariance -------------------------------------------------------------
P = "Write three short facts about rivers."
s, h, base = chat(BASE, P)
check("stock 200 without steer header", s == 200 and "x-omlx-steer" not in {k.lower() for k in h}, s)
s, h, zero = chat("tiny-zero", P)
check("alpha0 header", s == 200 and any(k.lower() == "x-omlx-steer" for k in h), h.get("x-omlx-steer"))
check("alpha0 == stock (tokens + logprobs)", sig(zero) == sig(base))
s, h, steer = chat("tiny-steer", P)
check("steered differs from stock", s == 200 and sig(steer) != sig(base))
s, _, steer2 = chat("tiny-steer", P)
check("steered deterministic", sig(steer2) == sig(steer))

# -- concurrency: rows stay isolated ---------------------------------------------------
# oMLX itself is not batch-invariant (a request alone and the same request in a
# batch can differ), so a row is compared with the same row of an all-stock /
# all-steered batch of the same size.
def batch(models, prompt=P):
    out = {}
    threads = [threading.Thread(target=lambda i=i, m=m: out.__setitem__(i, chat(m, prompt))) for i, m in enumerate(models)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [sig(out[i][2]) for i in range(len(models))]


ref_stock = batch([BASE, BASE, BASE])
ref_steer = batch(["tiny-steer"] * 3)
for trial in range(3):
    mixed = batch([BASE, "tiny-steer", "tiny-zero"])
    check(f"concurrent {trial}: stock row == all-stock batch row", mixed[0] == ref_stock[0])
    check(f"concurrent {trial}: steered row == all-steered batch row", mixed[1] == ref_steer[1])
    check(f"concurrent {trial}: alpha0 row == all-stock batch row", mixed[2] == ref_stock[2])

# -- override policy -----------------------------------------------------------------
s, _, _ = chat("tiny-steer", P, extra={"omlx_steer": {"alpha": 0.5}})
check("override rejected on prod profile", s == 400, s)
s, _, _ = chat(BASE, P, extra={"omlx_steer": {"alpha": 0.5}})
check("override rejected on stock model", s == 400, s)
s, h1, half = chat("tiny-sweep", P, extra={"omlx_steer": {"alpha": 0.5}})
s2, h2, full = chat("tiny-sweep", P)
check("sweep override accepted, new sha", s == 200 and h1.get("x-omlx-steer") != h2.get("x-omlx-steer"),
      (h1.get("x-omlx-steer"), h2.get("x-omlx-steer")))
check("sweep alpha1 == tiny-steer", sig(full) == sig(steer))

# -- cache salt -------------------------------------------------------------------
long = f"run {time.time_ns()}: " + " ".join(f"river{i} flows to sea{i % 97}." for i in range(1800))
usage = []
for model in (BASE, "tiny-steer", BASE, "tiny-steer"):
    s, _, r = chat(model, long, max_tokens=4)
    details = (r.get("usage") or {}).get("prompt_tokens_details") or {}
    usage.append(details.get("cached_tokens"))
print("cached_tokens per call (stock, steered, stock, steered):", usage)
if all(u is not None for u in usage):
    check("salt: steered first call reuses nothing", usage[1] == 0, usage)
    check("salt: stock and steered each reuse their own", usage[2] > 0 and usage[3] > 0, usage)

# -- capture -------------------------------------------------------------------------
(PROFILES / "capture" / "probe1.npz").unlink(missing_ok=True)
FRESH = f"nonce {time.time_ns()}: " + P
s, _, base = chat(BASE, FRESH, max_tokens=3)
s, _, _ = chat("tiny-capture", FRESH, max_tokens=1, extra={"omlx_steer": {"tag": "probe1"}})
cap = PROFILES / "capture" / "probe1.npz"
time.sleep(0.5)
if cap.exists():
    data = dict(np.load(cap))
    shapes = {k: v.shape for k, v in data.items()}
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(sys.argv[5]) if len(sys.argv) > 5 else None
    if tok is not None:
        content = base["choices"][0]["logprobs"]["content"]
        top = tok.decode([int(np.argmax(data["logprobs"]))])
        check("capture logprobs predict the first generated token (not the second)",
              top == content[0]["token"], (top, content[0]["token"], content[1]["token"]))
    check("capture file complete", all(k in data for k in ("attn", "ffn", "block_mean", "block_pre", "logprobs")) and data["block_mean"].shape[0] == 6, shapes)
else:
    check("capture file written", False, str(cap))
s, _, _ = chat("tiny-capture", P, max_tokens=1, extra={"omlx_steer": {"tag": "../evil"}})
check("capture tag sanitized", s == 400, s)

# -- no engine reloads -----------------------------------------------------------------
with LOG.open() as f:
    f.seek(log_start)
    tail = f.read()
bad = [ln for ln in tail.splitlines() if any(w in ln for w in ("Unloading model", "Loading model", "variant changed"))]
check("no engine reload or variant switch", not bad, bad[:3])
_, _, status = call("/v1/omlx-steer/status")
print("stats", status["stats"])
check("no ambiguous main-model forwards", status["stats"]["ambiguous_main"] == 0, status["stats"])
print("RESULT", "OK" if not fails else f"FAILED {fails}")
