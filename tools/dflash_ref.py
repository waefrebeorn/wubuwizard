#!/usr/bin/env python3
"""dflash_ref.py - reference implementation of the Inkling-2 DSpark drafter as upstream llama.cpp defines it.

Follows src/models/dspark.cpp (build_dspark_inp_block, build_dspark_markov_head) and src/models/dflash.cpp
(build_inject_ctx_mix). Weights come from the dequantised drafter GGUF; the trunk supplies only the anchor
embedding and the unembed matrix.

Architecture, from the drafter's own metadata:
    n_embd 2304, n_layer 3, n_head 24, n_head_kv 8, head_dim 128, ffn 9216, rms eps 1e-6,
    rope theta 1e6, rope dim 128, block_size 8, softcap 30, has_markov_context True, sample_from_anchor False

  x0   = [ rms(embd[anchor], token_embd_norm) ; mask_embd[:,0..6] ]      # 8 slots, slot 0 = anchor
  3 x  ( attn_norm -> GQA attn over ctx++block -> attn_output -> +residual
         ffn_norm -> SwiGLU -> ffn_down -> +residual )
  ctx K/V per layer = attn_k_ctx @ rms(sum_i ctx_mix[i]*tap_i, ctx_norm)  (+ k_ctx_norm + RoPE for K)
  h      = rms(x, output_norm);  base = h @ unembed^T;  base = 30*tanh(base/30)
  markov: prev = anchor; state = 0
          for i in 1..7 :  w1p = markov_w1[:,prev]
                           j   = joint_proj @ [state; w1p; h_i; f_rnn] + b     (5120 -> 768)
                           g, c = sigmoid(j[0:256]), tanh(j[256:512])
                           state = c + g*(state - c);  markov = tanh(j[512:768])
                           col = base[:,i] + markov_w2 @ markov + w2b
                           prev = argmax(col)
"""
import argparse, os, struct, sys

import numpy as np

N_EMBD, N_LAYER, N_HEAD, N_KV, HD, FF, R = 2304, 3, 24, 8, 128, 9216, 256
EPS = 1e-6
THETA = 1e6


def rms(x, w=None, eps=EPS):
    y = x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    return (y * w if w is not None else y).astype(np.float32)


def rope(x, pos, theta=THETA):
    """x: [T, H, HD], pos: [T] absolute positions. DFLASH_ROPE=interleaved switches to GPT-J pairing."""
    T, H, D = x.shape
    half = D // 2
    inv = theta ** (-np.arange(half, dtype=np.float64) * 2.0 / D)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]     # [T, half]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if os.environ.get('DFLASH_ROPE') == 'interleaved':
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out


class Drafter:
    def __init__(self, d, unembed, gguf=None, verbose=True):
        self.d = d
        self.t = {}
        self.unembed = unembed
        self.shapes = {}
        if gguf:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from gguf_extract import read_header
            _, _, tensors, _ = read_header(gguf)
            self.shapes = {n: tuple(dm) for n, dm, _ty, _off in tensors}
        if verbose:
            print(f"[dflash] unembed {unembed.shape} dtype {unembed.dtype}, {len(self.shapes)} shapes")

    def get(self, name):
        a = self.t.get(name)
        if a is None:
            shape = self.shapes[name]
            a = np.fromfile(os.path.join(self.d, name + '.f32'), dtype='<f4').reshape(shape)
            self.t[name] = a
        return a


    def ctx_kv(self, taps, layer, ctx_pos):
        """taps: [4, T, 2304] -> K,V for this layer over ctx positions."""
        mix = self.get(f'blk.{layer}.ctx_mix.weight')          # [4]
        cn = self.get(f'blk.{layer}.ctx_norm.weight')          # [2304]
        mixed = np.tensordot(mix, taps, axes=(0, 0))           # [T, 2304]
        mixed = rms(mixed, cn)
        k = mixed @ self.get(f'blk.{layer}.attn_k_ctx.weight')  # [T, 1024]
        v = mixed @ self.get(f'blk.{layer}.attn_v_ctx.weight')
        T = k.shape[0]
        k = k.reshape(T, N_KV, HD)
        kn = self.get(f'blk.{layer}.attn_k_ctx_norm.weight')   # [128]
        k = rms(k, kn)
        k = rope(k, ctx_pos)
        return k, v.reshape(T, N_KV, HD)

    def block(self, x, taps, ctx_pos, blk_pos, layer):
        an = self.get(f'blk.{layer}.attn_norm.weight')
        h = rms(x, an)
        B = h.shape[0]
        q = (h @ self.get(f'blk.{layer}.attn_q.weight')).reshape(B, N_HEAD, HD)
        k = (h @ self.get(f'blk.{layer}.attn_k.weight')).reshape(B, N_KV, HD)
        v = (h @ self.get(f'blk.{layer}.attn_v.weight')).reshape(B, N_KV, HD)
        q = rms(q, self.get(f'blk.{layer}.attn_q_norm.weight'))
        k = rms(k, self.get(f'blk.{layer}.attn_k_norm.weight'))
        q = rope(q, blk_pos)
        k = rope(k, blk_pos)

        if os.environ.get('DFLASH_NO_CTX'):
            kc = np.zeros((0, N_KV, HD), np.float32)
            vc = np.zeros((0, N_KV, HD), np.float32)
        else:
            kc, vc = self.ctx_kv(taps, layer, ctx_pos)
        K = np.concatenate([kc, k], axis=0)      # [T+B, KV, HD]
        V = np.concatenate([vc, v], axis=0)
        T = kc.shape[0]
        g = N_HEAD // N_KV
        K = np.repeat(K, g, axis=1)              # GQA expand
        V = np.repeat(V, g, axis=1)

        scale = 1.0 / np.sqrt(HD)
        att = np.einsum('bhd,khd->bhk', q.astype(np.float64), K.astype(np.float64)) * scale
        if os.environ.get('DFLASH_TRACE'):
            print(f'   [L{layer}] x rms {np.sqrt((x.astype(np.float64)**2).mean()):.3f} q rms {np.sqrt((q.astype(np.float64)**2).mean()):.3f} kc rms {np.sqrt((kc.astype(np.float64)**2).mean()):.3f} vc rms {np.sqrt((vc.astype(np.float64)**2).mean()):.3f} logit max {att.max():.2f} min {att.min():.2f}')
        # non-causal over ctx+block; block positions see each other unconditionally
        att = att - att.max(axis=-1, keepdims=True)
        p = np.exp(att)
        p /= p.sum(axis=-1, keepdims=True)
        o = np.einsum('bhk,khd->bhd', p, V.astype(np.float64)).astype(np.float32)
        o = o.reshape(B, N_HEAD * HD) @ self.get(f'blk.{layer}.attn_output.weight')
        x = x + o

        h2 = rms(x, self.get(f'blk.{layer}.ffn_norm.weight'))
        gate = h2 @ self.get(f'blk.{layer}.ffn_gate.weight')
        up = h2 @ self.get(f'blk.{layer}.ffn_up.weight')
        act = (gate / (1.0 + np.exp(-gate))) * up        # silu(gate) * up
        x = x + act @ self.get(f'blk.{layer}.ffn_down.weight')
        return x


def forward(dr, anchor_emb, taps, ctx_pos, blk_pos, taps_frnn=None):
    tok = dr.get('token_embd_norm.weight')
    x0 = np.concatenate([rms(anchor_emb[None, :], tok), dr.get('mask_embd.weight').T], axis=0)  # [8, 2304]
    x = x0
    for L in range(N_LAYER):
        x = dr.block(x, taps, ctx_pos, blk_pos, L)
        print(f"  layer {L}: rms {np.sqrt((x.astype(np.float64)**2).mean()):.4f}")
    h = rms(x, dr.get('output_norm.weight'))
    base = h @ dr.unembed.T                                     # [8, vocab]
    base = 30.0 * np.tanh(base / 30.0)                          # softcap
    # f_rnn: the markov-side context mix
    mixm = dr.get('markov_ctx_mix.weight')
    mixn = dr.get('markov_ctx_norm.weight')
    src = taps if taps_frnn is None else taps_frnn
    f_rnn = rms(np.tensordot(mixm, src, axes=(0, 0)), mixn)[-1]  # most recent ctx row
    return h, base, f_rnn


def markov(dr, base, h, f_rnn, anchor, n_draft, verbose=True):
    w1 = dr.get('markov_w1.weight')      # [256, vocab]
    w2 = dr.get('markov_w2.weight')      # [256, vocab]
    w2b = dr.get('markov_w2.bias')       # [vocab]
    jp = dr.get('joint_proj.weight')     # [5120, 768]
    jb = dr.get('joint_proj.bias')       # [768]
    state = np.zeros(R, dtype=np.float32)
    prev = int(anchor)
    out = []
    for i in range(1, n_draft + 1):
        w1p = w1[:, prev]
        j = np.concatenate([state, w1p, h[i], f_rnn]) @ jp + jb
        g = 1.0 / (1.0 + np.exp(-j[:R]))
        c = np.tanh(j[R:2 * R])
        state = (c + g * (state - c)).astype(np.float32)
        mk = np.tanh(j[2 * R:3 * R]).astype(np.float32)
        col = base[i] + mk @ w2 + w2b
        prev = int(np.argmax(col))
        out.append((prev, float(col[prev])))
        if verbose:
            print(f"  draft {i}: {prev}  score {col[prev]:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='C:/Users/eman5/models/inkling-2-drafter/f32')
    ap.add_argument('--gguf', default='C:/Users/eman5/models/inkling-2-drafter/dspark-Inkling-2-Mini-Q8_0.gguf')
    ap.add_argument('--unembed', default=os.path.join(os.environ.get('TEMP', '.'), 'unembed.f32'))
    ap.add_argument('--emb', required=True, help='anchor embedding, raw f32 2304')
    ap.add_argument('--anchor', type=int, required=True)
    ap.add_argument('--taps', required=True, help='comma list of 4 tap .bin files (T x 2304 f32)')
    ap.add_argument('--ctx-pos', default='auto')
    ap.add_argument('--oracle', default=None, help='comma list of expected ids')
    a = ap.parse_args()

    un = np.memmap(a.unembed, dtype='<f4', mode='r')
    print(f"[dflash] unembed file {un.size} floats = {un.size // N_EMBD} rows")
    un = un.reshape(-1, N_EMBD)[:200058]
    dr = Drafter(a.dir, un, gguf=a.gguf)

    taps = []
    for p in a.taps.split(','):
        t = np.fromfile(p, dtype='<f4')
        taps.append(t.reshape(-1, N_EMBD))
    T = taps[0].shape[0]
    print(f"[dflash] {len(taps)} taps x {T} ctx positions")
    taps = np.stack(taps)                      # [4, T, 2304]

    emb = np.fromfile(a.emb, dtype='<f4')[:N_EMBD]
    anchor_pos = T - 1                        # slot 0 IS the anchor, at its own position
    ctx_pos = np.arange(T)
    blk_pos = anchor_pos + np.arange(8)
    h, base, f_rnn = forward(dr, emb, taps, ctx_pos, blk_pos)
    print('[base] per-slot top-5 (before markov):')
    for i in range(8):
        t5 = np.argsort(-base[i])[:5]
        print(f'  slot {i}: ' + '  '.join(f'{int(t)}:{base[i][t]:.2f}' for t in t5))
    print(f'[base] rms {np.sqrt((base.astype(np.float64)**2).mean()):.4f}  h rms {np.sqrt((h.astype(np.float64)**2).mean()):.4f}  f_rnn rms {np.sqrt((f_rnn.astype(np.float64)**2).mean()):.4f}')
    if a.oracle:
        oracle = [int(x) for x in a.oracle.split(',')]
        res = markov(dr, base, h, f_rnn, a.anchor, min(7, len(oracle)))
        got = [d for d, _ in res]
        hit = sum(1 for i, d in enumerate(got) if i < len(oracle) and d == oracle[i])
        print(f"  oracle {oracle[:len(got)]}")
        print(f"  MATCH {hit}/{len(got)}")
    else:
        markov(dr, base, h, f_rnn, a.anchor, 7)


if __name__ == '__main__':
    main()
