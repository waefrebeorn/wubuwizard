#!/usr/bin/env python3
"""mix_check.py -- verify one layer's depth mix (Inkling2LoopBlockAttentionResidual) independently.

The mix is attention, not scoring: entries are unweighted-RMS-normed, the query is dotted against them with
hidden_size**-0.5 scaling, softmax over entries, and the output is the weighted sum of those SAME normed
entries. Without the 1/sqrt(d) factor the softmax saturates and the mix becomes a hard pick of one state,
so the scaling is load-bearing rather than cosmetic.

Inputs: the engine's dumped mixed state for the layer (IN2_DUMP_MIX), the layer input (IN2_DUMP_LAYERS),
and the block state banked by the previous boundary layer - here blocks[0] for blk.1, which is the normed
embedding of the token (IQ4_XS row + token_embd_norm).

    IN2_DUMP_MIX=mix1.bin IN2_DUMP_LAYERS=dir ./inkling_logits.exe 1234
    python tools/mix_check.py <model.gguf> <token> dir/L00.bin mix1.bin <attn_res_query_off> <token_embd_off>
"""
import struct, sys
import numpy as np

KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

def iq4_xs_row(f, at, ne=2304):
    f.seek(at); raw = f.read((ne // 256) * 136)
    out = np.zeros(ne, dtype=np.float32)
    for b in range(ne // 256):
        blk = raw[b * 136:(b + 1) * 136]
        d = np.float32(np.frombuffer(blk[0:2], dtype=np.float16)[0])
        sh = struct.unpack("<H", blk[2:4])[0]; sl = blk[4:8]; qs = blk[8:136]
        for ib in range(8):
            ls = ((sl[ib // 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4)
            dl = np.float32(d * np.float32(ls - 32))
            for j in range(16):
                q = qs[ib * 16 + j]
                out[b * 256 + ib * 32 + j]      = dl * np.float32(KV[q & 0xf])
                out[b * 256 + ib * 32 + j + 16] = dl * np.float32(KV[q >> 4])
    return out

def rms(x, w=None, eps=1e-6):
    s = np.sqrt((x.astype(np.float64) ** 2).mean() + eps)
    y = (x / s).astype(np.float32)
    return y * w if w is not None else y

if __name__ == "__main__":
    model, token, lpath, mpath, aq_off, emb_off, tn_off = sys.argv[1:8]
    base = 11978688                      # data section: tensor dir end aligned to 32
    f = open(model, "rb")
    emb = iq4_xs_row(f, base + int(emb_off) + int(token) * 1024)
    f.seek(base + int(tn_off)); tn = np.frombuffer(f.read(2304 * 4), dtype=np.float32)
    f.seek(base + int(aq_off)); aq = np.frombuffer(f.read(2304 * 4), dtype=np.float32)
    e0 = rms(emb, tn)                    # blocks[0]: the normed embedding blk.0 banked
    h = np.fromfile(lpath, dtype=np.float32)
    eng = np.fromfile(mpath, dtype=np.float32)
    E = np.stack([rms(e0), rms(h)])
    sc = (E @ aq) / np.float32(np.sqrt(2304))
    w = np.exp(sc - sc.max()); w /= w.sum()
    mine = (w[:, None] * E).sum(0).astype(np.float32)
    print(f"weights {w}  (saturated pick would be 1/0)")
    print(f"rms engine {np.sqrt((eng**2).mean()):.6f}  python {np.sqrt((mine**2).mean()):.6f}")
    print(f"max abs diff {np.abs(mine-eng).max():.3e}  corr {np.corrcoef(mine,eng)[0,1]:.12f}")
