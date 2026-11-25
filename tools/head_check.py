#!/usr/bin/env python3
"""head_check.py -- verify the LM head independently of the engine.

Reads a dumped final hidden state (IN2_DUMP_H) and the engine's logits (IN2_DUMP_LOGITS), dequantises the
matching rows of output.weight from the GGUF with a hand-written IQ4_XS decoder, and dots them. The engine
applies softcap(30) to logits, so the comparison is against atanh(y/30)*30.

If these agree the output projection, final norm, row indexing and IQ4_XS dequantiser are all correct and
any wrong logits come from upstream. That is the point: it splits the graph in half with one comparison.

    IN2_DUMP_H=h.bin IN2_DUMP_LOGITS=logits.bin ./inkling_logits.exe 1234
    python tools/head_check.py <model.gguf> h.bin logits.bin 87 779 11
"""
import struct, sys
import numpy as np

# kvalues_iq4nl from ggml-common.h (GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)) - the format authority
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

def find_tensor(path, want):
    f = open(path, "rb")
    f.read(4); struct.unpack("<I", f.read(4))[0]
    nt, nkv = struct.unpack("<QQ", f.read(16))
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")
    def rv(t):
        if t in (0, 1): return struct.unpack("<B", f.read(1))[0]
        if t in (2, 3): return struct.unpack("<H", f.read(2))[0]
        if t in (4, 5): return struct.unpack("<I", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0]
        if t == 8: return rs()
        if t in (10, 11): return struct.unpack("<Q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        if t == 9:
            at, an = struct.unpack("<IQ", f.read(12)); return [rv(at) for _ in range(an)]
        raise ValueError(t)
    for _ in range(nkv): rs(); rv(struct.unpack("<I", f.read(4))[0])
    base = (f.tell() + 31) // 32 * 32          # data section starts after the 32-byte-aligned tensor dir
    info = None
    for _ in range(nt):
        name = rs(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
        if name == want: info = (ty, dims, off)
    f.close()
    return base, info

def iq4_xs_row(f, at, ne):
    """One row of IQ4_XS: 256-element blocks of 136 bytes."""
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
                q = qs[ib * 16 + j]              # 16 bytes per sub-block, as dequantize_row_iq4_xs does
                out[b * 256 + ib * 32 + j]      = dl * np.float32(KV[q & 0xf])
                out[b * 256 + ib * 32 + j + 16] = dl * np.float32(KV[q >> 4])
    return out

def unsoftcap(y, cap=30.0):
    return cap * np.arctanh(np.clip(y / cap, -0.999999, 0.999999))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__); raise SystemExit(2)
    model, hpath, lpath = sys.argv[1:4]
    ids = [int(x) for x in sys.argv[4:]] or [0]
    base, info = find_tensor(model, "output.weight")
    if not info: print("output.weight not found"); raise SystemExit(1)
    ty, dims, off = info
    ne, nrow = dims[0], dims[1]
    h = np.fromfile(hpath, dtype=np.float32)
    lg = np.fromfile(lpath, dtype=np.float32)
    print(f"output.weight type={ty} dims={dims}  h={h.shape} logits={lg.shape}")
    f = open(model, "rb")
    worst = 0.0
    for row in ids:
        w = iq4_xs_row(f, base + off + (row * ne // 256) * 136, ne)
        mine = float(np.dot(w, h)); raw = float(unsoftcap(lg[row]))
        worst = max(worst, abs(mine - raw))
        print(f"  id {row:7d}  python {mine:+13.6f}  engine {raw:+13.6f}  diff {mine - raw:+.2e}")
    print(f"worst abs diff {worst:.2e}")
