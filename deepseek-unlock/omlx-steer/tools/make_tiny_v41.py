"""Build a tiny random-weight DeepSeek-V4.1 checkpoint in oMLX's converted format.

The checkpoint keeps every architectural feature the steering hooks touch:
hyper-connections (hc_mult 4), compressed / indexed attention, Engram layers,
MoE with shared + routed experts, and three embedded DSpark (MTP) stages. The
projections use the same MXFP8 / MXFP4 layouts as the released oQ4e checkpoint,
so oMLX's fused decode, batched decode and MTP verify kernels all run on it.

Usage (with oMLX's bundled interpreter):
    python tools/make_tiny_v41.py --tokenizer-from REAL_MODEL_DIR --out OUT_DIR
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

QUANTIZED = [
    (r"\.attn\.(wq_a|wq_b|wkv|wo_b)$", "mxfp8"),
    (r"\.attn\.indexer\.wq_b$", "mxfp8"),
    (r"\.ffn\.shared_experts\.(w1|w2|w3)$", "mxfp8"),
    (r"\.ffn\.experts\.(w1|w2|w3)$", "mxfp4"),
    (r"\.engram\.wkv$", "mxfp8"),
    (r"^language_model\.mtp\.0\.main_proj$", "mxfp8"),
]


def _prime(n):
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def engram_rows(layer_ids, ngram, heads, vocab):
    """Table rows per Engram layer, following NgramHash's prime layout."""
    rows, seen = [], set()
    for _ in layer_ids:
        total = 0
        for _ in range(ngram - 1):
            current = vocab - 1
            for _ in range(heads):
                current += 1
                while not _prime(current) or current in seen:
                    current += 1
                seen.add(current)
                total += current
        rows.append(total)
    return rows


def tiny_text_config(real_text):
    layers, mtp = 6, 3
    engram_ids, ngram, heads, evocab = [1, 3], 4, 8, 1000
    text = dict(real_text)
    text.update(
        hidden_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=16,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=256,
        o_lora_rank=256,
        o_groups=8,
        n_routed_experts=16,
        num_experts_per_tok=4,
        compress_ratios=[0, 0, 2, 2, 1, 1] + [0] * mtp,
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4],
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        candidate_source_layer_id=4,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        engram_layer_ids=engram_ids,
        engram_num_embeddings=engram_rows(engram_ids, ngram, heads, evocab),
        engram_max_ngram_size=ngram,
        engram_vocab_size=evocab,
        engram_n_heads=heads,
        engram_head_dim=64,
        num_nextn_predict_layers=mtp,
        dspark_target_layer_ids=[3, 4, 5],
        dspark_markov_rank=64,
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=2,
    )
    return text


def kind_for(module):
    for pattern, mode in QUANTIZED:
        if re.search(pattern, module):
            return mode
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from omlx.patches.deepseek_v41 import apply_patch

    apply_patch()
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.model import Model

    src, out = Path(args.tokenizer_from), Path(args.out)
    real = json.loads((src / "config.json").read_text())
    cfg = {k: v for k, v in real.items() if k not in ("vision_config",)}
    cfg["text_config"] = tiny_text_config(real["text_config"])
    cfg["omlx_deepseek_v41"] = {"version": 1, "preserve_mtp": True, "quantized_modules": {}}

    config = ModelConfig.from_dict(cfg)
    model = Model(config)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tensors, quantized = {}, {}
    for name, value in tree_flatten(model.parameters()):
        module, _, leaf = name.rpartition(".")
        mode = kind_for(module) if leaf == "weight" else None
        if "norm" in module.split(".")[-1] and leaf == "weight":
            tensors[name] = mx.ones(value.shape, mx.bfloat16)
            continue
        scale = 1.0 / max(1.0, float(value.shape[-1]) ** 0.5) if value.ndim >= 2 else 0.02
        rand = mx.random.normal(value.shape) * scale
        if mode is not None:
            bits = 8 if mode == "mxfp8" else 4
            w, s = mx.quantize(rand.astype(mx.bfloat16), group_size=32, bits=bits, mode=mode)[:2]
            tensors[name] = w
            tensors[module + ".scales"] = s
            quantized[module] = {"bits": bits, "mode": mode}
        elif module.endswith(".attn.wo_a") and leaf == "weight":
            # The fused decode needs a bf16 wo_a that is exactly MXFP8-representable.
            w, s = mx.quantize(rand.astype(mx.bfloat16), group_size=32, bits=8, mode="mxfp8")[:2]
            tensors[name] = mx.dequantize(w, s, group_size=32, bits=8, mode="mxfp8").astype(mx.bfloat16)
        elif value.dtype in (mx.float32, mx.bfloat16, mx.float16):
            # Same dtypes as the release: fp32 for sinks, hyper-connection mixers
            # and router biases, bf16 for everything else.
            fp32 = re.search(r"(attn_sink|\.hc_\w+|\.gate\.bias\w*)$", name)
            tensors[name] = rand.astype(mx.float32 if fp32 else mx.bfloat16)
        else:
            tensors[name] = value
    cfg["omlx_deepseek_v41"]["quantized_modules"] = quantized

    out.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(out / shard), tensors)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}}, indent=1)
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    total = sum(v.nbytes for v in tensors.values())
    print(f"wrote {out} ({total / 2**20:.1f} MiB, {len(tensors)} tensors, {len(quantized)} quantized)")


if __name__ == "__main__":
    main()
