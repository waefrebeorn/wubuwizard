#!/usr/bin/env python3
"""gguf_vocab.py - compare the tokenizer vocab of two GGUFs, array fields included.

The drafter's markov projections are 200058 wide against the trunk's 201024, and no d2t mapping tensor exists.
Before any logit alignment is assumed, this prints enough of both vocabs to show whether one is a prefix of the
other, a reordering, or a genuinely different set.
"""
import argparse, struct, sys


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
STRING, ARRAY = 8, 9


def read_value(f, t):
    if t in SCALAR:
        return struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]
    if t == STRING:
        return read_str(f)
    if t == ARRAY:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        if et in SCALAR:
            sz, fm = struct.calcsize(SCALAR[et]), SCALAR[et]
            return list(struct.unpack('<%d%s' % (cnt, fm[1]), f.read(sz * cnt)))
        if et == STRING:
            return [read_str(f) for _ in range(cnt)]
        raise SystemExit(f"unsupported array elem {et}")
    raise SystemExit(f"unsupported value type {t}")


def load(path, want=('tokenizer.ggml.tokens', 'tokenizer.ggml.merges', 'tokenizer.ggml.model',
                     'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.mask_token_id')):
    out = {}
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit("not a GGUF")
        struct.unpack('<I', f.read(4))
        n_t = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            v = read_value(f, t)
            if k in want:
                out[k] = v
        out['_n_tensors'] = n_t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    ta, tb = A.get('tokenizer.ggml.tokens'), B.get('tokenizer.ggml.tokens')
    print(f"A {a.a}\n  model={A.get('tokenizer.ggml.model')} tokens={len(ta)} eos={A.get('tokenizer.ggml.eos_token_id')} mask={A.get('tokenizer.ggml.mask_token_id')}")
    print(f"B {a.b}\n  model={B.get('tokenizer.ggml.model')} tokens={len(tb)} eos={B.get('tokenizer.ggml.eos_token_id')} mask={B.get('tokenizer.ggml.mask_token_id')}")
    n = min(len(ta), len(tb))
    same = sum(1 for i in range(n) if ta[i] == tb[i])
    print(f"  common prefix length by equality: ", end='')
    cp = 0
    while cp < n and ta[cp] == tb[cp]:
        cp += 1
    print(cp)
    print(f"  identical at same index over first {n}: {same}")
    print(f"  A[{len(ta)-4}:] = {ta[-4:]}")
    print(f"  B[{len(tb)-4}:] = {tb[-4:]}")
    for probe in (0, 1, 10, 11, 220, 1377, 32329, 199999, 200006, 200054, 200057, 200058, 201023):
        va = ta[probe] if probe < len(ta) else '<oob>'
        vb = tb[probe] if probe < len(tb) else '<oob>'
        flag = '' if va == vb else '   <-- DIFFERS'
        print(f"  id {probe:>7}: A={va!r:<24} B={vb!r}{flag}")


if __name__ == '__main__':
    main()
