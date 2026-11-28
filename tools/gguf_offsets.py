#!/usr/bin/env python3
"""Print a GGUF's data-section origin and per-tensor absolute offsets, matching src/gguf_reader.c exactly.

WHY THIS EXISTS: the first version of this script computed data_start BEFORE reading the tensor-info section, so it reported
11931808 instead of 11978688 - 46880 bytes early - and every "real bytes" comparison built on it read the wrong tensor data
while producing plausible-magnitude garbage that looked like a decode disagreement. The ordering is the whole bug: metadata
kv entries, THEN the tensor-info entries, THEN align to `general.alignment` (default 32).
Verified against the engine: data_blob_offset 11978688, blk.5.ffn_gate_up_exps.weight abs 2003254464.
"""
import struct, sys

def offsets(path):
    f = open(path, 'rb')
    def rd(n): return f.read(n)
    def u32(): return struct.unpack('<I', rd(4))[0]
    def u64(): return struct.unpack('<Q', rd(8))[0]
    def s():
        n = u64(); return rd(n).decode('utf-8', 'replace')
    SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    align = 32
    def skip(t):
        nonlocal align
        if t == 8: return s()
        if t == 9:
            et = u32(); n = u64()
            if et == 8: return [s() for _ in range(n)]
            f.seek(SZ[et] * n, 1); return None
        f.seek(SZ[t], 1); return None
    assert rd(4) == b'GGUF'
    u32(); nt = u64(); nkv = u64()
    for _ in range(nkv):
        k = s(); t = u32(); v = skip(t)
        if k == 'general.alignment' and isinstance(v, int): align = v
    tens = []
    for _ in range(nt):
        nm = s(); nd = u32(); dims = [u64() for _ in range(nd)]
        ty = u32(); off = u64(); tens.append((nm, ty, off, dims))
    end = f.tell()
    data_start = end + (align - (end % align)) % align
    return data_start, align, tens

if __name__ == '__main__':
    ds, al, tens = offsets(sys.argv[1])
    print(f"data_start={ds} alignment={al} tensors={len(tens)}")
    pat = sys.argv[2] if len(sys.argv) > 2 else None
    for nm, ty, off, dims in tens:
        if pat is None or pat in nm:
            print(f"  {nm} type={ty} dims={dims} abs={ds + off}")
