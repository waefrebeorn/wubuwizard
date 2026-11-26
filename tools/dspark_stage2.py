#!/usr/bin/env python3
"""dspark_stage2.py - reference implementation of DSpark stages 1+2 (context fusion + 3 block-attention layers).

Source of truth: strata engine/src/inkling2/cuda/ink_dspark.cuh
  k_dspark_mix_taps (:52)        layer il mixes the 4 taps with softmax(fuse_w[il]) and normalises by
                                 **dspark.layers.il.dflare_norm.weight** - per-layer, NOT dspark.norm.weight
                                 (call site :1445-1449)
  k_dspark_split_kv (:394)       [M, 2048] fused kv_ctx -> k [M,1024], v [M,1024]
  k_dspark_rmsnorm_rope (:19)    per-head RMSNorm over 128, then NeoX half-split RoPE, theta = 1e6, pairs (d, d+64)
  k_dspark_block_attn (:127)     Q [B,24,128]; K/V from ctx (fp16 cache) + own block [B,8,128];
                                 GQA kh = h/3; scale = 1/sqrt(128); NON-causal; online softmax over
                                 ctx[0..apos) + block[0..B)
  k_dspark_swiglu (:246)         SwiGLU, hidden 9216

Block size B = 8, H = 2304, 24 q heads, 8 kv heads, head_dim 128, theta 1e6.

Usage:  python dspark_stage2.py --weights DIR --taps DIR [--block 8]
"""
import argparse, glob, os, sys
import numpy as np

H, HQ, HKV, HD, THETA, B_DEF = 2304, 24, 8, 128, 1e6, 8
KV_DIM, Q_DIM = HKV * HD, HQ * HD
EPS = 1e-6
TAP_VISITS = (5, 23, 41, 47)


def load(w, name):
    p = os.path.join(w, name + '.f32')
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return np.fromfile(p, dtype='<f4').astype(np.float32)


def mat(w, name, rows, cols):
    a = load(w, name)
    if a.size != rows * cols:
        sys.exit(f"{name}: {a.size} != {rows}x{cols}")
    return a.reshape(rows, cols)


def rmsnorm(x, wt, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * wt


def rope(x, pos):
    """x: [N, heads, 128], pos: [N]. NeoX half-split: pair (d, d+64)."""
    N, nh, _ = x.shape
    d = np.arange(HD // 2, dtype=np.float32)
    inv = 1.0 / (THETA ** (2.0 * d / HD))
    ang = np.asarray(pos, dtype=np.float32)[:, None] * inv[None, :]      # [N, 64]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    o = np.empty_like(x)
    o[..., :HD // 2] = x1 * c - x2 * s
    o[..., HD // 2:] = x1 * s + x2 * c
    return o


def mix_taps(taps, fuse, dflare, row):
    """taps: [4, M, H] -> [M, H] using softmax(fuse[row]) and dflare_norm as the norm weight."""
    fw = fuse[row * 4:(row + 1) * 4]
    e = np.exp(fw - fw.max())
    w = (e / e.sum()).astype(np.float32)
    v = np.tensordot(w, taps, axes=(0, 0))
    return rmsnorm(v, dflare), w


def attn(q, kd, vd, ctx_k, ctx_v, block_pos):
    """q [B,24,128]; kd/vd [B,8,128]; ctx_k/ctx_v [M,8,128]. Non-causal over ctx ++ block."""
    B = q.shape[0]
    K = np.concatenate([ctx_k, kd], axis=0)      # [M+B, 8, 128]
    V = np.concatenate([ctx_v, vd], axis=0)
    scale = 1.0 / np.sqrt(HD)
    out = np.empty((B, HQ, HD), dtype=np.float32)
    for h in range(HQ):
        kh = h // (HQ // HKV)
        s = (q[:, h, :] @ K[:, kh, :].T) * scale          # [B, M+B]
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, h, :] = p @ V[:, kh, :]
    return out


def forward(x, taps, fuse, W, applied_pos, ctx_pos, B):
    """x: [B, H] block state. taps: [4, M, H]. Returns [B, H]."""
    h = x.astype(np.float32).copy()
    for il in range(3):
        p = f'dspark.layers.{il}'
        ctx, w = mix_taps(taps, fuse, load(W, p + '.dflare_norm.weight'), il)
        M = ctx.shape[0]
        kc = ctx @ mat(W, p + '.self_attn.k_proj_ctx.weight', KV_DIM, H).T        # [M,1024]
        vc = ctx @ mat(W, p + '.self_attn.v_proj_ctx.weight', KV_DIM, H).T
        kc = rmsnorm(kc.reshape(M, HKV, HD), load(W, p + '.self_attn.k_norm_ctx.weight'))
        vc = vc.reshape(M, HKV, HD)
        kc = rope(kc, ctx_pos)

        xn = rmsnorm(h, load(W, p + '.input_layernorm.weight'))
        q = (xn @ mat(W, p + '.self_attn.q_proj.weight', Q_DIM, H).T).reshape(B, HQ, HD)
        k = (xn @ mat(W, p + '.self_attn.k_proj.weight', KV_DIM, H).T).reshape(B, HKV, HD)
        v = (xn @ mat(W, p + '.self_attn.v_proj.weight', KV_DIM, H).T).reshape(B, HKV, HD)
        q = rope(rmsnorm(q, load(W, p + '.self_attn.q_norm.weight')), applied_pos)
        k = rope(rmsnorm(k, load(W, p + '.self_attn.k_norm.weight')), applied_pos)

        a = attn(q, k, v, kc, vc, applied_pos).reshape(B, Q_DIM)
        h = h + a @ mat(W, p + '.self_attn.o_proj.weight', H, Q_DIM).T

        xn = rmsnorm(h, load(W, p + '.post_attention_layernorm.weight'))
        g = xn @ mat(W, p + '.mlp.gate_proj.weight', 9216, H).T
        u = xn @ mat(W, p + '.mlp.up_proj.weight', 9216, H).T
        act = (g / (1.0 + np.exp(-g))) * u
        h = h + act @ mat(W, p + '.mlp.down_proj.weight', H, 9216).T
        print(f"  layer {il}: fuse w={np.round(w,3)} ctx={M}x{H} out_rms={np.sqrt((h**2).mean()):.4f}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--taps', required=True)
    ap.add_argument('--block', type=int, default=B_DEF)
    a = ap.parse_args()

    taps = []
    for v in TAP_VISITS:
        hits = sorted(glob.glob(os.path.join(a.taps, f'V{v:03d}_*.bin')))
        if not hits:
            sys.exit(f"no dump for visit {v}")
        t = np.fromfile(hits[0], dtype='<f4').reshape(-1, H)
        taps.append(t)
    M = min(t.shape[0] for t in taps)
    taps = np.stack([t[:M] for t in taps])
    fuse = load(a.weights, 'dspark.fuse_weight')
    print(f"taps: {taps.shape}   block B={a.block}")

    ctx_pos = np.arange(M)
    applied_pos = np.arange(M, M + a.block)
    # Real block init, from emit_dspark_backbone (:1484-1489): slot 0 is the anchor token's embedding passed
    # through model.llm.embed_norm, slots 1..B-1 are dspark.mask_embed. We do not have the anchor embedding
    # here, so use mask_embed for all slots - still the real vector, unlike the old taps[3] placeholder.
    mask = load(a.weights, 'dspark.mask_embed')
    print(f"mask_embed rms={np.sqrt((mask**2).mean()):.4f}")
    x = np.repeat(mask[None, :], a.block, axis=0)
    y = forward(x, taps, fuse, a.weights, applied_pos, ctx_pos, a.block)
    print(f"\nout {y.shape}  rms={np.sqrt((y**2).mean()):.4f}  finite={np.isfinite(y).all()}  "
          f"min={y.min():.3f} max={y.max():.3f}")


if __name__ == '__main__':
    main()
