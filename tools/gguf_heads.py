"""What does the file already give us for free? MTP/draft heads, a tied output head, and the metadata that names them.

Front A (one pass -> many tokens) is much cheaper if the checkpoint ships multi-token-prediction tensors, and front B
(the 469 MB head, 31% of a decode step's traffic) is free if output.weight is literally token_embd.weight.
"""
import struct
import sys

P = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf'
TT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 0, 9: 0, 10: 8, 11: 8, 12: 8}

f = open(P, 'rb')
magic, ver, nt, nkv = struct.unpack('<IIQQ', f.read(24))


def rs():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def rv(t, depth=0):
    if t == 8:
        return rs()
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if n > 8:
            for _ in range(n):
                rv(et, depth + 1)
            return f'<array of {n}>'
        return [rv(et, depth + 1) for _ in range(n)]
    fmt = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[t]
    return struct.unpack('<' + fmt, f.read(TT[t]))[0]


md = {}
for _ in range(nkv):
    k = rs()
    t = struct.unpack('<I', f.read(4))[0]
    md[k] = rv(t)

print('=== metadata, everything that names architecture or a head ===')
KEY = ('nextn', 'draft', 'mtp', 'arch', 'expert', 'attention', 'context', 'rope', 'head', 'tie', 'link', 'layer',
       'feed_forward', 'embedding', 'sliding', 'window', 'shared', 'group', 'visit')
for k in sorted(md):
    if any(x in k.lower() for x in KEY):
        v = md[k]
        s = str(v)
        print(f'  {k:52s} {s[:70]}')

names = []
for _ in range(nt):
    nm = rs()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
    ty = struct.unpack('<I', f.read(4))[0]
    names.append((nm, dims, ty))
    struct.unpack('<Q', f.read(8))[0]

print(f'\n=== {len(names)} tensors; anything outside blk.N.* ===')
for nm, dims, ty in names:
    if not nm.startswith('blk.'):
        print(f'  {nm:34s} dims={dims} type={ty}')

print('\n=== MTP / draft / nextn tensors? ===')
hits = [n for n, d, t in names if any(x in n.lower() for x in ('nextn', 'draft', 'mtp', 'eh_proj', 'enorm', 'hnorm'))]
print('  ' + ('\n  '.join(hits) if hits else 'NONE - no MTP head in this checkpoint'))

print('\n=== non-blk tensor names for head/embed comparison ===')
for nm, dims, ty in names:
    if 'token_embd' in nm or 'output' in nm:
        print(f'  {nm:34s} dims={dims} type={ty}')

print('\n=== per-layer tensor name set (layer 0) ===')
for nm, dims, ty in names:
    if nm.startswith('blk.0.'):
        print(f'  {nm:40s} {str(dims):22s} type={ty}')
