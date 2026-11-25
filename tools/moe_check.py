#!/usr/bin/env python3
"""moe_check.py -- reproduce in2_moe independently, for a layer whose expert types are known.

Verified bit-exact against tools/expert_probe.c output, which pinned the packing: this GGUF's type-17 is
grid = iq2xs_grid[q & 511], signs = ksigns_iq2xs[(q >> 9) & 127] - a 9-bit grid plus a SEVEN-bit sign index,
not the 9+3 of canonical ggml IQ2_XS. That is why the bytes carry full 16-bit entropy and why the canonical
12-bit reading fails. db = d * (0.5 + ls) * 0.25 per 4-bit sub-scale.

    IN2_DUMPX=1 ./inkling_logits.exe 1234        # writes d_moein.bin / d_moe.bin for the last sliding layer
    python tools/moe_check.py <gguf> d_moein.bin d_moe.bin <grid> <dn> <picks...> <weights...>
"""
import sys, re
import numpy as np

def ksigns(src_path):
    src = open(src_path, encoding="utf-8", errors="replace").read()
    body = re.search(r"ksigns_iq2xs\[128\]\s*=\s*\{(.*?)\};", src, re.S).group(1)
    return np.array([int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", re.sub(r"//[^\n]*", "", body))], dtype=np.uint16)

def iq2xs(raw, ne, grid, ks):
    km = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint16)
    nb = ne // 256
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 74)
    d = np.frombuffer(a[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    qs = np.frombuffer(a[:, 2:66].tobytes(), dtype=np.uint16).reshape(nb, 32)
    sc = a[:, 66:74]
    out = np.zeros((nb, 256), dtype=np.float32)
    for ib in range(8):
        db0 = d * (0.5 + (sc[:, ib] & 0xf).astype(np.float32)) * 0.25
        db1 = d * (0.5 + (sc[:, ib] >> 4).astype(np.float32)) * 0.25
        for l in range(4):
            q = qs[:, 4 * ib + l]
            g = grid[(q & 511)]
            sg = ks[((q >> 9) & 127).astype(np.int64)]
            neg = ((sg[:, None] & km[None, :]) != 0)
            db = db0 if l < 2 else db1
            out[:, ib * 32 + l * 8:ib * 32 + l * 8 + 8] = (g * np.where(neg, np.float32(-1), np.float32(1))) * db[:, None]
    return out.reshape(-1)

def q2_0(raw, ne):
    nb = ne // 64
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 18)
    d = np.frombuffer(a[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    vals = ((a[:, 2:18][:, :, None] >> (np.arange(4) * 2)[None, None, :]) & 0x03).astype(np.int16) - 1
    return (vals.reshape(nb, -1).astype(np.float32) * d[:, None]).reshape(-1)

if __name__ == "__main__":
    model, mpath, epath, gu_off, dn_off = sys.argv[1:6]
    args = [float(x) for x in sys.argv[6:]]
    half = len(args) // 2
    picks = [int(x) for x in args[:half]]; weights = np.array(args[half:], dtype=np.float32)
    base = 11978688
    grid = np.load("iq2xs_grid.npy").astype(np.float32)      # from ggml-common.h
    ks = ksigns("src/gguf_reader.c")
    GUE = 2304 * 2304 * 74 // 256; DNE = 1152 * 2304 * 18 // 64
    f = open(model, "rb")
    moein = np.fromfile(mpath, dtype=np.float32); eng = np.fromfile(epath, dtype=np.float32)
    acc = np.zeros(2304, dtype=np.float64)
    for wi, e in zip(weights, picks):
        f.seek(base + int(gu_off) + e * GUE); gu = iq2xs(f.read(GUE), 2304 * 2304, grid, ks).reshape(2304, 2304)
        pre = gu @ moein
        act = (np.tanh(pre[:1152] / 4.0) * 4.0) * (1.0 / (1.0 + np.exp(-pre[:1152]))) * (np.tanh(pre[1152:] / 25.0) * 25.0)
        f.seek(base + int(dn_off) + e * DNE); dn = q2_0(f.read(DNE), 1152 * 2304).reshape(2304, 1152)
        acc += wi * (dn @ act.astype(np.float32))
    mine = acc.astype(np.float32)
    print(f"max|diff| {np.abs(mine-eng).max():.4e}  mean {np.abs(mine-eng).mean():.4e}  corr {np.corrcoef(mine,eng)[0,1]:.9f}")
