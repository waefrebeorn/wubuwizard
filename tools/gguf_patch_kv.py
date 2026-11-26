#!/usr/bin/env python3
"""gguf_patch_kv.py - rewrite a scalar integer metadata value in a GGUF copy, in place.

The trunk GGUF declares `inkling2.block_count = 48` (the visit count used by the loop-core schedule) while physically
holding blk.0..blk.23, i.e. 24 layers. Our engine reads 48 as visits; upstream llama.cpp reads block_count as the layer
count and refuses to load ("tensor 'blk.24.attn_q.weight' not found"). Upstream cannot be used as a reference without
a copy whose block_count is the physical layer count.

Only same-size rewrites are allowed, so no offsets shift and no tensor moves. The original file is never modified.

Usage:
  python gguf_patch_kv.py SRC DST KEY VALUE
"""
import argparse, os, shutil, struct, sys


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
            sz = struct.calcsize(SCALAR[et])
            f.read(sz * cnt)
        elif et == STRING:
            for _ in range(cnt):
                read_str(f)
        else:
            raise SystemExit(f"unsupported array elem {et}")
    else:
        raise SystemExit(f"unsupported value type {t}")


def find_value_offset(path, key):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit("not a GGUF")
        ver = struct.unpack('<I', f.read(4))[0]
        struct.unpack('<Q', f.read(8))          # n_tensors
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            if k == key:
                if t not in SCALAR:
                    raise SystemExit(f"{key} is not a scalar (type {t})")
                return ver, t, f.tell(), SCALAR[t]
            skip_value(f, t)
    raise SystemExit(f"key {key} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('key')
    ap.add_argument('value', type=int)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    ver, ty, off, fmt = find_value_offset(a.src, a.key)
    with open(a.src, 'rb') as f:
        f.seek(off)
        old = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    new = struct.pack(fmt, a.value)
    print(f"{a.key}: {old} -> {a.value}   (type {ty} {fmt}, at offset {off}, v{ver}, {len(new)} bytes)")

    if a.dry_run:
        print("dry run, nothing written")
        return

    print(f"copying {a.src} -> {a.dst} ({os.path.getsize(a.src)} bytes)")
    shutil.copyfile(a.src, a.dst)
    with open(a.dst, 'r+b') as f:
        f.seek(off)
        f.write(new)
    with open(a.dst, 'rb') as f:
        f.seek(off)
        check = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    print(f"verify: {a.key} = {check}")
    if check != a.value:
        raise SystemExit("patch did not take")


if __name__ == '__main__':
    main()
