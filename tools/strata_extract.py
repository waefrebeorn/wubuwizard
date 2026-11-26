#!/usr/bin/env python3
"""strata_extract.py - read tensors out of a .strata pack to raw f32.

Formats verified against real data (see docs/research/strata-container.md):
  F32      rows*cols*4, row-major
  BF16     rows*cols*2
  Q8_ROW   [rows] f32 scales, then [rows][cols] int8   (NOT interleaved: ink_cpu_backend.cpp:50)
  S5_K/S3_K/S2_K/S4_K/S6_K/S8_K
           166/102/70/134/198/262 bytes per 256-weight superblock:
             2 B fp16 D, 4 B u[0..3] subgroup multipliers (/255), then packed q values
             value = (q - half) * D * (u[g]/255), half = (2^bits - 1)/2  (15.5 for 5-bit)
             q packed LSB-first, 5 bits each for S5_K

Usage:
  python strata_extract.py --list                     # every tensor, name/fmt/shape/bytes
  python strata_extract.py --dump NAME OUT.f32        # one tensor as little-endian f32
  python strata_extract.py --dump-dspark DIR          # all 54 DSpark tensors into DIR
"""
import argparse, json, os, struct, sys, zstandard as zstd

SF4 = None
SUPER = {"S1_25": 48, "S1_5": 58, "S2_K": 70, "S3_K": 102, "S4_K": 134, "S5_K": 166, "S6_K": 198, "S8_K": 262}
HALF = {"S2_K": 1.5, "S3_K": 3.5, "S4_K": 7.5, "S5_K": 15.5, "S6_K": 31.5}
BITS = {"S2_K": 2, "S3_K": 3, "S4_K": 4, "S5_K": 5, "S6_K": 6}
QK_S, SUB_G, NUM_SUB = 256, 64, 4
BASE = 4096   # tensor offsets are relative to section 1's start


def load_manifest(path):
    f = open(path, 'rb'); h = f.read(512)
    assert h[:8] == b'STRATA\x00\x01', "not a .strata file"
    nsec = struct.unpack_from('<H', h, 14)[0]
    s0_off = s0_len = None
    for i in range(nsec):
        o = 16 + i * 64
        fo, ln = struct.unpack_from('<QQ', h, o)
        if i == 0:
            s0_off, s0_len = fo, ln
    f.seek(s0_off); raw = f.read(s0_len)
    f.close()
    return json.loads(zstd.ZstdDecompressor().decompress(raw, max_output_size=1 << 26).decode('utf-8'))


def f16(b):
    return struct.unpack('<e', struct.pack('<H', b))[0]


def dequant_sk(blk, fmt):
    sb = SUPER[fmt]
    d_bits = struct.unpack_from('<H', blk, 0)[0]
    D = f16(d_bits)
    u = blk[2:6]
    pl = blk[6:]
    out = [0.0] * QK_S
    if fmt == "S8_K":
        q = struct.unpack_from('<256b', pl, 0)
        for g in range(NUM_SUB):
            eff = D * (u[g] / 255.0)
            for i in range(SUB_G):
                out[g * SUB_G + i] = q[g * SUB_G + i] * eff
        return out
    bits = BITS[fmt]
    half = HALF[fmt]
    mask = (1 << bits) - 1
    for g in range(NUM_SUB):
        eff = D * (u[g] / 255.0)
        for i in range(SUB_G):
            idx = g * SUB_G + i
            bp = idx * bits
            by = bp >> 3
            sh = bp & 7
            v = pl[by] if by < len(pl) else 0
            if by + 1 < len(pl):
                v |= pl[by + 1] << 8
            if by + 2 < len(pl):
                v |= pl[by + 2] << 16
            q = (v >> sh) & mask
            out[idx] = (q - half) * eff
    return out


def read_tensor(path, meta, t):
    rows, cols = (t['shape'] + [1, 1])[:2]
    n = rows * cols
    f = open(path, 'rb'); f.seek(BASE + t['offset']); raw = f.read(t['bytes']); f.close()
    fmt = t['format']
    if fmt == 'F32':
        return list(struct.unpack_from('<%df' % n, raw, 0))
    if fmt == 'BF16':
        u16 = struct.unpack_from('<%dH' % n, raw, 0)
        return [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in u16]
    if fmt == 'F16':
        return [f16(v) for v in struct.unpack_from('<%dH' % n, raw, 0)]
    if fmt == 'Q8_ROW':
        sc = struct.unpack_from('<%df' % rows, raw, 0)
        qs = struct.unpack_from('<%db' % n, raw, rows * 4)
        out = [0.0] * n
        for r in range(rows):
            s = sc[r]
            for c in range(cols):
                out[r * cols + c] = qs[r * cols + c] * s
        return out
    if fmt in SUPER:
        sb = SUPER[fmt]
        nsb = n // QK_S
        assert nsb * QK_S == n, f"{fmt} tensor not a whole number of superblocks"
        out = []
        for s in range(nsb):
            out.extend(dequant_sk(raw[s * sb:(s + 1) * sb], fmt))
        return out
    raise SystemExit(f"unhandled format {fmt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dump', nargs=2, metavar=('NAME', 'OUT'))
    ap.add_argument('--dump-dspark', metavar='DIR')
    a = ap.parse_args()
    j = load_manifest(a.pack)
    allt = j['trunk_tensors'] + j['dspark_tensors']
    if a.list:
        for t in allt:
            print(f"{t['name']:<60} {t['format']:<8} {str(t['shape']):<16} {t['bytes']:>11}")
        print(f"\n{j['format']}  trunk={len(j['trunk_tensors'])} dspark={len(j['dspark_tensors'])}")
        return
    if a.dump:
        name, out = a.dump
        t = next((x for x in allt if x['name'] == name), None)
        if not t:
            raise SystemExit(f"no tensor named {name}")
        vals = read_tensor(a.pack, j, t)
        open(out, 'wb').write(struct.pack('<%df' % len(vals), *vals))
        print(f"{name}: {len(vals)} floats -> {out}")
        return
    if a.dump_dspark:
        os.makedirs(a.dump_dspark, exist_ok=True)
        for t in j['dspark_tensors']:
            vals = read_tensor(a.pack, j, t)
            p = os.path.join(a.dump_dspark, t['name'] + '.f32')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(struct.pack('<%df' % len(vals), *vals))
        print(f"dumped {len(j['dspark_tensors'])} DSpark tensors to {a.dump_dspark}")


if __name__ == '__main__':
    main()
