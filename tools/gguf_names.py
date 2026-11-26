#!/usr/bin/env python3
"""gguf_names.py - dump a GGUF's tensor table (names, shapes, types) and optionally diff against a name list.

Reads only the metadata and tensor-info section, never the data, so it is fast and memory-light on multi-GB files.
GGUF v3 layout: header, KV pairs (typed, arrays are elem_type + count + elements), then tensor infos.

Usage:
  python gguf_names.py FILE                 # print every tensor: name, dims, type
  python gguf_names.py FILE --grep dspark   # filter by substring
  python gguf_names.py FILE --skeleton      # normalise indices to N and print the sorted unique set
"""
import argparse, struct, sys

# gguf type sizes for the metadata values
SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?',
          10: '<Q', 11: '<q', 12: '<d'}
STRING, ARRAY = 8, 9
GGML_TYPE_NAMES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
    10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
    16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
    22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64',
    30: 'BF16',
}


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


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
        elif et == ARRAY:
            raise SystemExit("nested array in metadata - unsupported")
        else:
            raise SystemExit(f"bad array elem type {et}")
    else:
        raise SystemExit(f"bad value type {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--grep', default=None)
    ap.add_argument('--skeleton', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    with open(a.file, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            sys.exit(f"not a GGUF: {magic!r}")
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            read_str(f)                     # key
            t = struct.unpack('<I', f.read(4))[0]
            skip_value(f, t)
        rows = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack('<I', f.read(4))[0]
            f.read(8)                       # offset
            rows.append((name, dims, GGML_TYPE_NAMES.get(ty, str(ty))))

    print(f"# {a.file}: v{ver}, {n_tensors} tensors, {n_kv} kv pairs")
    sel = [r for r in rows if not a.grep or a.grep in r[0]]
    if a.limit:
        sel = sel[:a.limit]
    if a.skeleton:
        import re
        sk = sorted({re.sub(r'(?<![0-9])\d+(?![0-9])', 'N', n) for n, _, _ in sel})
        for s in sk:
            print(s)
        print(f"# {len(sk)} distinct skeletons")
    else:
        for n, d, t in sel:
            print(f"{n:<58} {str(d):<18} {t}")
    print(f"# {len(sel)} shown of {len(rows)}")


if __name__ == '__main__':
    main()
