"""Load the tiny checkpoint with oMLX's own loader and run every forward path once."""
import sys
import mlx.core as mx

from omlx.patches.deepseek_v41 import apply_patch
apply_patch()
from omlx.patches.deepseek_v41 import decode_fused, loading

model, proc = loading.load(sys.argv[1])
lm = model.language_model
print("fused per layer:", [decode_fused.supported(l, lm._config) for l in lm.layers])
ids = mx.array([[1, 400, 500, 600, 700, 800, 900, 1000] * 4])
cache = lm.make_cache()
logits = model(ids, cache=cache)
mx.eval(logits)
nxt = mx.argmax(logits[:, -1:], -1)
out = model(nxt, cache=cache)
mx.eval(out)
print("prefill", logits.shape, "decode", out.shape, "ok")
