#!/usr/bin/env python3
"""gguf_extract.py - dequantise a GGUF's tensors to flat .f32 files, one per tensor.

Supports the types the Inkling-2 DSpark drafter actually uses: F32, F16, BF16, Q8_0.
Deliberately NOT the Strata Q8_ROW plane layout - ggml Q8_0 is a 34-byte block per 32 weights:
    fp16 d  +  32 x int8 q        value = d * q
which is why a shared decoder between the two formats would be wrong.

Usage:
  python gguf_extract.py FILE --list
  python gguf_extract.py FILE --dump NAME OUT.f32
  python gguf_extract.py FILE --dump-all DIR
"""
import argparse, os, struct, sys

import numpy as np

DTYPE_SIZES = {0: 4, 1: 2, 2: 2, 3: 2}   # f32, f16, bf16, (f32 again for completeness)


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
STRING, ARRAY = 8, 9


def skip_value(f, t):
    if t in SCALAR:
        f.read(struct.calcsize(SCALAR[t]))
    elif t == STRING:
        read_str(f)
    elif t == ARRAY:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        if et in SCALAR:
            f.read(struct.calcsize(SCALAR[et]) * cnt)
        elif et == STRING:
            for _ in range(cnt):
                read_str(f)
        else:
            raise SystemExit(f"unsupported array elem {et}")
    else:
        raise SystemExit(f"unsupported value type {t}")


def read_header(path):
    """Return (kv, tensors) where tensors = [(name, dims, type, offset)] and data_start."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit("not a GGUF")
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            if t in SCALAR:
                v = struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]
                kv[k] = v
            elif t == STRING:
                kv[k] = read_str(f)
            else:
                skip_value(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, dims, ty, off))
        align = kv.get('general.alignment', 32)
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    return ver, kv, tensors, data_start


def dequant(raw, dims, ty):
    """Return float32 array shaped dims (ggml order: ne[0] fastest)."""
    n = 1
    for d in dims:
        n *= d
    if ty == 0:
        a = np.frombuffer(raw, dtype='<f4', count=n)
    elif ty == 1:
        a = np.frombuffer(raw, dtype='<f2', count=n).astype(np.float32)
    elif ty == 30:
        u = np.frombuffer(raw, dtype='<u2', count=n).astype(np.uint32) << 16
        a = u.view(np.float32) if u.dtype == np.uint32 else None
        a = (np.frombuffer(raw, dtype='<u2', count=n).astype(np.uint32) << 16).view(np.float32)
    elif ty == 8:   # Q8_0: 34-byte block per 32 weights
        nb = n // 32
        if n % 32:
            raise SystemExit(f"Q8_0 length {n} not a multiple of 32")
        blocks = np.frombuffer(raw, dtype=np.uint8, count=nb * 34).reshape(nb, 34)
        d = blocks[:, :2].copy().view('<f2').astype(np.float32).reshape(nb, 1)
        q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
        a = (q * d).reshape(-1)
    else:
        raise SystemExit(f"unsupported ggml type {ty}")
    return a.reshape([d for d in dims])


def tensor_bytes(dims, ty):
    n = 1
    for d in dims:
        n *= d
    if ty == 0:
        return n * 4
    if ty in (1, 30):
        return n * 2
    if ty == 8:
        return (n // 32) * 34
    raise SystemExit(f"unsupported type {ty}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dump')
    ap.add_argument('--out')
    ap.add_argument('--dump-all')
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()

    ver, kv, tensors, data_start = read_header(a.file)
    total = sum(tensor_bytes(d, t) for _, d, t, _ in tensors)

    if a.list or not (a.dump or a.dump_all):
        for name, dims, ty, off in tensors:
            print(f"{name:<50} {str(dims):<18} type={ty:<3} {tensor_bytes(dims, ty):>10} B")
        print(f"# {len(tensors)} tensors, {total} B of data, data_start={data_start}")
        return

    size = os.path.getsize(a.file)
    if a.verify:
        last = max(tensors, key=lambda r: r[3])
        end = data_start + last[3] + tensor_bytes(last[1], last[2])
        print(f"# data region {data_start}..{end} of file size {size}  (delta {size - end})")
        if size - end < 0:
            raise SystemExit("tensor table overruns the file")

    with open(a.file, 'rb') as f:
        if a.dump:
            hit = [t for t in tensors if t[0] == a.dump]
            if not hit:
                sys.exit(f"no tensor {a.dump}")
            name, dims, ty, off = hit[0]
            f.seek(data_start + off)
            arr = dequant(f.read(tensor_bytes(dims, ty)), dims, ty)
            arr.astype('<f4').tofile(a.out)
            print(f"{name} {dims} type={ty} -> {a.out} {arr.size * 4} B  rms={float(np.sqrt((arr.astype(np.float64)**2).mean())):.6f}")
        else:
            os.makedirs(a.dump_all, exist_ok=True)
            for name, dims, ty, off in tensors:
                f.seek(data_start + off)
                arr = dequant(f.read(tensor_bytes(dims, ty)), dims, ty)
                p = os.path.join(a.dump_all, name + '.f32')
                arr.astype('<f4').tofile(p)
            print(f"dumped {len(tensors)} tensors to {a.dump_all}, {total} B f32")


if __name__ == '__main__':
    main()
