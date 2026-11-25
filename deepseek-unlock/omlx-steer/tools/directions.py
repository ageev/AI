"""Refusal and hedging directions from captured activations.

    directions.py CAPTURE_DIR OUT_DIR [--source block_mean] [--shift 0] [--layers 40] [--mtp 3]

Per layer l (recipe of the earlier Qwen abliterations):
  d_ref = sum over think/nothink of mean(harm) - mean(harmless) at the last
          gen-prompt token;
  biprojection: remove its component in span(harmless means of both modes);
  d_hedge = same with advice requests vs reference questions, biprojected on
          the reference means, then orthogonalised against r_ref.
--shift -1 takes the direction from the output of the previous block (the
residual the layer's writers read) instead of the layer's own output.

Writes <source><shift>.both.npz (refusal + hedging rows, the profile file) and
<source><shift>.ref.npz (refusal only, for sweeps), r: [layers + mtp, k, hidden]
(MTP rows left zero; profiles use mtp "last"), and a stats JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(cap, prefix, source):
    files = sorted(Path(cap).glob(f"{prefix}.*.npz"))
    if not files:
        raise SystemExit(f"no captures for {prefix}")
    acc, n = None, 0
    for f in files:
        with np.load(f) as d:
            x = d[source].astype(np.float64) if source != "writers" else (d["attn"] + d["ffn"]).astype(np.float64)
        acc = x if acc is None else acc + x
        n += 1
    return acc / n, n


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def biproject(d, basis):
    q, _ = np.linalg.qr(np.stack(basis, 1))
    return d - q @ (q.T @ d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--source", default="block_mean", choices=["block_mean", "block_pre", "writers"])
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--mode", default="sum", choices=["sum", "span"],
                    help="sum: one refusal vector (think + nothink); span: both, orthonormalised")
    a = ap.parse_args()

    m = {}
    for s in ("harm_train", "harmless_train", "advice_train", "reference_train"):
        for mode in ("think", "nothink"):
            m[s, mode], n = load(a.capture, f"{s}.{mode}", a.source)
            print(f"{s}.{mode}: {n} prompts")
    L, D = a.layers, m["harm_train", "think"].shape[-1]
    ref = np.zeros((L, D))
    ref2 = np.zeros((L, D))  # span mode: second refusal vector
    hedge = np.zeros((L, D))
    stats = []
    for layer in range(L):
        src = layer + a.shift
        if src < 0:
            stats.append({"layer": layer, "skipped": True})
            continue
        dt = m["harm_train", "think"][src] - m["harmless_train", "think"][src]
        dn = m["harm_train", "nothink"][src] - m["harmless_train", "nothink"][src]
        h = np.linalg.norm(m["harmless_train", "nothink"][src])
        hb = [m["harmless_train", "think"][src], m["harmless_train", "nothink"][src]]
        if a.mode == "span":
            q2, _ = np.linalg.qr(np.stack([biproject(dt, hb), biproject(dn, hb)], 1))
            r, r2 = q2[:, 0], q2[:, 1]
        else:
            r, r2 = unit(biproject(dt + dn, hb)), None
        mt = m["advice_train", "think"][src] - m["reference_train", "think"][src]
        mn = m["advice_train", "nothink"][src] - m["reference_train", "nothink"][src]
        q = biproject(mt + mn, [m["reference_train", "think"][src], m["reference_train", "nothink"][src]])
        cos_raw = float(unit(q) @ unit(dt + dn))
        q = q - (q @ r) * r
        if r2 is not None:
            q = q - (q @ r2) * r2
            ref2[layer] = r2
        q = unit(q)
        ref[layer], hedge[layer] = r, q
        stats.append({
            "layer": layer,
            "ref_think_rel": float(np.linalg.norm(dt) / h),
            "ref_nothink_rel": float(np.linalg.norm(dn) / h),
            "hedge_rel": float(np.linalg.norm(mt + mn) / h),
            "cos_think_nothink": float(unit(dt) @ unit(dn)),
            "cos_ref_hedge": cos_raw,
        })
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.source}{a.shift:+d}" if a.shift else a.source
    if a.mode == "span":
        name += "-span"
    total = L + a.mtp

    def save(suffix, rows):
        r = np.zeros((total, len(rows), D), np.float32)
        for k, arr in enumerate(rows):
            r[:L, k] = arr
        np.savez(out / f"{name}.{suffix}.npz", r=r)

    refs = [ref] if a.mode == "sum" else [ref, ref2]
    save("ref", refs)
    save("both", refs + [hedge])
    (out / f"{name}.stats.json").write_text(json.dumps(stats, indent=1))
    for s in stats:
        if not s.get("skipped"):
            print(f"L{s['layer']:02d} |d_think|/|h|={s['ref_think_rel']:.3f} |d_nothink|/|h|={s['ref_nothink_rel']:.3f} "
                  f"|d_hedge|/|h|={s['hedge_rel']:.3f} cos(think,nothink)={s['cos_think_nothink']:.2f} cos(ref,hedge)={s['cos_ref_hedge']:.2f}")


if __name__ == "__main__":
    main()
