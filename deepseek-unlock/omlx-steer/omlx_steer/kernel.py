"""Row-masked directional projection: y_i <- y_i - alpha_i * R_i^T (R_i y_i).

``project`` is the only entry point. Rows with alpha 0 (stock rows in a mixed
batch) are returned bit for bit. The Metal kernel does the whole projection of
one token in one threadgroup (one dispatch per writer); ``impl="mlx"`` uses
plain MLX ops instead and serves as the reference.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

MAX_K = 4

_SOURCE = r"""
    // One threadgroup per token row. y: [rows, D] (T), r: [groups, K, D] f32,
    // alpha: [groups] f32, group of a row = row / M (M tokens per batch row).
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint g = row / M;
    const float a = alpha[g];
    const device T* yr = y + (size_t)row * D;
    device T* outr = out + (size_t)row * D;
    if (a == 0.0f) {
        for (uint d = tid; d < D; d += TPG) outr[d] = yr[d];
        return;
    }
    const device float* rg = r + (size_t)g * K * D;
    float part[K];
    for (int k = 0; k < K; ++k) part[k] = 0.0f;
    for (uint d = tid; d < D; d += TPG) {
        const float v = float(yr[d]);
        for (int k = 0; k < K; ++k) part[k] = fma(v, rg[k * D + d], part[k]);
    }
    threadgroup float sums[TPG / 32][K];
    threadgroup float coef[K];
    for (int k = 0; k < K; ++k) {
        const float s = simd_sum(part[k]);
        if (lane == 0) sums[sg][k] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {
        float total = 0.0f;
        for (uint i = 0; i < TPG / 32; ++i) total += sums[i][tid];
        coef[tid] = a * total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < D; d += TPG) {
        float v = float(yr[d]);
        for (int k = 0; k < K; ++k) v = fma(-coef[k], rg[k * D + d], v);
        outr[d] = T(v);
    }
"""

_TPG = 256


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_steer_masked_project",
        input_names=["y", "r", "alpha"],
        output_names=["out"],
        source=_SOURCE,
    )


def _stack(dirs, n):
    """[n, K, D] float32 from per-group [k_i, D] arrays (None = no direction)."""
    width = next(d.shape[-1] for d in dirs if d is not None)
    k = max(d.shape[0] for d in dirs if d is not None)
    if k > MAX_K:
        raise ValueError(f"at most {MAX_K} directions per layer")
    parts = []
    for d in dirs:
        if d is None:
            parts.append(mx.zeros((k, width), mx.float32))
        elif d.shape[0] < k:
            parts.append(mx.concatenate([d, mx.zeros((k - d.shape[0], width), mx.float32)]))
        else:
            parts.append(d)
    return mx.stack(parts) if n > 1 else parts[0][None]


def project(y, dirs, alphas, impl="kernel"):
    """Project rows of ``y`` (shape [n, ..., D]); ``dirs[i]`` / ``alphas[i]`` per row i."""
    n = len(dirs)
    if all(d is None or a == 0.0 for d, a in zip(dirs, alphas)):
        return y
    shape = y.shape
    width = shape[-1]
    alphas = [0.0 if d is None else float(a) for d, a in zip(dirs, alphas)]
    if impl == "mlx":
        return _reference(y.reshape(n, -1, width), dirs, alphas).reshape(shape)
    rows = y.size // width
    per_row = rows // n
    r = _stack(dirs, n)
    alpha = mx.array(alphas, mx.float32)
    return _kernel()(
        inputs=[mx.contiguous(y.reshape(rows, width)), r, alpha],
        template=[("T", y.dtype), ("D", width), ("K", r.shape[1]), ("M", per_row), ("TPG", _TPG)],
        grid=(rows * _TPG, 1, 1),
        threadgroup=(_TPG, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[y.dtype],
    )[0].reshape(shape)


def _reference(y, dirs, alphas):
    """MLX-ops reference on [n, m, D]."""
    outs = []
    for i, (d, a) in enumerate(zip(dirs, alphas)):
        row = y[i]
        if d is None or a == 0.0:
            outs.append(row)
            continue
        f = row.astype(mx.float32)
        coef = f @ d.T
        outs.append((f - a * (coef @ d)).astype(y.dtype))
    return mx.stack(outs)
