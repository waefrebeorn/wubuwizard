#!/usr/bin/env python3
"""gguf_reindex_layers.py - rewrite a visit-indexed trunk GGUF into physical-layer indexing.

The Inkling-2 trunk declares 48 visits (`inkling2.block_count = 48`) with 48-entry per-layer arrays, but physically
holds blk.0..blk.23. Our engine wants that (48 visits over 24 physical layers); upstream llama.cpp wants physical
indexing and refuses to load it:
    check_tensor_dims: tensor 'blk.24.attn_q.weight' not found
    key inkling2.attention.head_count has wrong array length; expected 24, got 48

GGUF tensor offsets are relative to the data section start, so shortening the KV section moves nothing inside the
data: the whole data region can be copied verbatim after adjusting where it begins. That is what this does.

Which entries survive: the three per-layer arrays are all structurally periodic with period 6 (= `attn_res.block_size`):
    attention.head_count          [24] * 48                  uniform
    attention.head_count_kv       [8]  * 48                  uniform
    attention.sliding_window_pattern   [T,T,T,T,T,F] * 8     period 6
so the first `n_physical` entries are the correct physical pattern, and truncation is lossless. This is asserted below
rather than assumed: the script refuses to run if an array is not periodic with the declared period.

Usage:
  python gguf_reindex_layers.py SRC DST [--n-physical 24] [--dry-run]
"""
import argparse, os, struct, sys

SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
STRING, ARRAY = 8, 9
ARRAY_KEYS = ('inkling2.attention.head_count',
              'inkling2.attention.head_count_kv',
              'inkling2.attention.sliding_window_pattern')


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n)          # raw bytes, including the length prefix handled by caller


def read_key(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def parse_header(path):
    """Return dict with everything needed to rebuild the header and locate the data section."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit("not a GGUF")
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        kvs = []
        for _ in range(n_kv):
            k = read_key(f)
            t = struct.unpack('<I', f.read(4))[0]
            if t in SCALAR:
                sz = struct.calcsize(SCALAR[t])
                raw = f.read(sz)
                val = struct.unpack(SCALAR[t], raw)[0]
            elif t == STRING:
                raw = read_str(f)
                val = raw
            elif t == ARRAY:
                et = struct.unpack('<I', f.read(4))[0]
                cnt = struct.unpack('<Q', f.read(8))[0]
                if et in SCALAR:
                    sz = struct.calcsize(SCALAR[et])
                    body = f.read(sz * cnt)
                    val = list(struct.unpack('<%d%s' % (cnt, SCALAR[et][1]), body))
                elif et == STRING:
                    val = []
                    for _ in range(cnt):
                        n = struct.unpack('<Q', f.read(8))[0]
                        val.append(f.read(n))
                else:
                    raise SystemExit(f"unsupported array elem {et}")
                raw = (et, cnt)
            else:
                raise SystemExit(f"unsupported type {t}")
            kvs.append({'key': k, 'type': t, 'val': val, 'raw': raw, 'elem': (raw[0] if t == ARRAY else None)})
        tensor_bytes = f.read(0)          # placeholder; tensor infos rebuilt below
        tinfos = []
        for _ in range(n_tensors):
            name = read_key(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tinfos.append((name, dims, ty, off))
        hdr_end = f.tell()

    def enc_str(s):
        b = s.encode('utf-8') if isinstance(s, str) else s
        return struct.pack('<Q', len(b)) + b

    def enc_kv(e):
        out = enc_str(e['key']) + struct.pack('<I', e['type'])
        if e['type'] in SCALAR:
            return out + struct.pack(SCALAR[e['type']], e['val'])
        if e['type'] == STRING:
            return out + enc_str(e['raw'])
        et, cnt = e['raw']
        vals = e['val']
        out += struct.pack('<I', et) + struct.pack('<Q', len(vals))
        if et in SCALAR:
            fmt = '<%d%s' % (len(vals), SCALAR[et][1])
            return out + struct.pack(fmt, *vals)
        for v in vals:
            out += enc_str(v)
        return out

    def enc_tinfo(t):
        name, dims, ty, off = t
        return enc_str(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) \
            + struct.pack('<I', ty) + struct.pack('<Q', off)

    return dict(ver=ver, n_tensors=n_tensors, n_kv=n_kv, kvs=kvs, tinfos=tinfos,
                hdr_end=hdr_end, enc_kv=enc_kv, enc_tinfo=enc_tinfo, enc_str=enc_str)


def align_up(x, a):
    return (x + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--n-physical', type=int, default=24)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    H = parse_header(a.src)
    print(f"{a.src}: v{H['ver']}, {H['n_tensors']} tensors, {H['n_kv']} kv, header ends {H['hdr_end']}")

    period = None
    for e in H['kvs']:
        if e['key'] == 'inkling2.attn_res.block_size':
            period = e['val']
    print(f"period (attn_res.block_size) = {period}")

    changed = 0
    for e in H['kvs']:
        if e['key'] == 'inkling2.block_count':
            print(f"  block_count {e['val']} -> {a.n_physical}")
            e['val'] = a.n_physical
            changed += 1
        elif e['key'] in ARRAY_KEYS:
            vals = e['val']
            if len(vals) <= a.n_physical:
                print(f"  {e['key']}: already {len(vals)} entries, left alone")
                continue
            # refuse unless the array is periodic with the declared period, so truncation is provably lossless
            if period and all(vals[i] == vals[i % period] for i in range(len(vals))):
                print(f"  {e['key']}: {len(vals)} -> {a.n_physical}  (verified periodic, period {period})")
            else:
                raise SystemExit(f"  {e['key']} is NOT periodic with period {period}; truncation would be lossy - refusing")
            e['val'] = vals[:a.n_physical]
            changed += 1

    if not changed:
        raise SystemExit("nothing changed")

    align = 32
    for e in H['kvs']:
        if e['key'] == 'general.alignment' and isinstance(e['val'], int):
            align = e['val']

    hdr = b'GGUF' + struct.pack('<I', H['ver']) + struct.pack('<Q', H['n_tensors']) + struct.pack('<Q', H['n_kv'])
    for e in H['kvs']:
        hdr += H['enc_kv'](e)
    for t in H['tinfos']:
        hdr += H['enc_tinfo'](t)
    new_data_start = align_up(len(hdr), align)
    hdr += b'\x00' * (new_data_start - len(hdr))

    old_size = os.path.getsize(a.src)
    old_data_start = align_up(H['hdr_end'], align)
    delta = new_data_start - old_data_start
    print(f"  header {H['hdr_end']} -> {len(hdr)}  data start {old_data_start} -> {new_data_start}  (delta {delta})")
    print(f"  expected dst size {new_data_start + (old_size - old_data_start)}")

    if a.dry_run:
        print("dry run, nothing written")
        return

    with open(a.src, 'rb') as fi, open(a.dst, 'wb') as fo:
        fo.write(hdr)
        fi.seek(old_data_start)
        remaining = old_size - old_data_start
        while remaining > 0:
            chunk = fi.read(min(64 << 20, remaining))
            if not chunk:
                raise SystemExit("unexpected EOF while copying data")
            fo.write(chunk)
            remaining -= len(chunk)
    print(f"  wrote {os.path.getsize(a.dst)} bytes")

    # verify by re-parsing the result
    V = parse_header(a.dst)
    bc = [e['val'] for e in V['kvs'] if e['key'] == 'inkling2.block_count'][0]
    lens = [(e['key'], len(e['val'])) for e in V['kvs'] if e['key'] in ARRAY_KEYS]
    print(f"  verify: block_count={bc} tensors={V['n_tensors']} arrays={lens}")
    ok = bc == a.n_physical and all(n == a.n_physical for _, n in lens) and V['n_tensors'] == H['n_tensors']
    print("  VERIFY OK" if ok else "  VERIFY FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
