"""Extract real trained vectors from a GGUF (BF16/F16 only, no dequant needed) for the retrieval harness."""
import struct, sys, os

FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def rd(f, n):
    b = f.read(n)
    if len(b) != n: raise EOFError
    return b

def u32(f): return struct.unpack('<I', rd(f,4))[0]
def u64(f): return struct.unpack('<Q', rd(f,8))[0]

def rstr(f):
    n = u64(f)
    return rd(f,n).decode('utf-8','replace')

def skip_val(f, t):
    if t == 8: rstr(f); return
    if t == 9:
        et = u32(f); cnt = u64(f)
        if et == 8:
            for _ in range(cnt): rstr(f)
        else:
            f.seek(FIXED[et]*cnt, 1)
        return
    f.seek(FIXED[t], 1)

path = sys.argv[1]
f = open(path,'rb')
assert rd(f,4) == b'GGUF'
ver = u32(f); n_tensors = u64(f); n_kv = u64(f)
print('gguf v%d, %d tensors, %d kv' % (ver, n_tensors, n_kv))

align = 32
for _ in range(n_kv):
    k = rstr(f)
    t = u32(f)
    if k == 'general.alignment':
        v = struct.unpack('<I', rd(f,4))[0]; align = v; print('alignment', v); continue
    skip_val(f, t)

tensors = []
for _ in range(n_tensors):
    name = rstr(f)
    nd = u32(f)
    dims = [u64(f) for _ in range(nd)]
    tt = u32(f)
    off = u64(f)
    tensors.append((name, dims, tt, off))

# data section starts at the next alignment boundary
pos = f.tell()
f.seek((pos + align - 1)//align*align)
data0 = f.tell()

TYPES = {0:'F32',1:'F16',2:'Q4_0',8:'Q8_0',30:'BF16'}
cands = [(n,d,t,o) for (n,d,t,o) in tensors if len(d)==2 and t in (0,1,30) and d[0]>=64 and d[1]>=64]
print('BF16/F16/F32 2-D tensors:')
for n,d,t,o in sorted(cands, key=lambda x:-(x[1][0]*x[1][1]))[:6]:
    print('  %-42s %6d x %-6d %s' % (n, d[0], d[1], TYPES.get(t,'?')))

if not cands:
    print('none found'); sys.exit(1)

name, dims, tt, off = sorted(cands, key=lambda x:-(x[1][0]*x[1][1]))[0]
n_in, n_out = dims[0], dims[1]
N = int(sys.argv[3]) if len(sys.argv)>3 else 20000
N = min(N, n_out)
print('chosen: %s  [%d x %d] type %s, taking %d rows' % (name, n_in, n_out, TYPES[tt], N))

f.seek(data0 + off)
rows = []
for r in range(N):
    raw = rd(f, n_in * 2)
    if tt == 30:
        v = struct.unpack('<%de' % n_in, raw)
    else:
        v = struct.unpack('<%de' % n_in, raw)
    rows.append(v)

# normalise to the unit sphere, which is what the quantiser assumes
out = sys.argv[2]
with open(out,'wb') as g:
    g.write(struct.pack('<II', N, n_in))
    for v in rows:
        nrm = sum(x*x for x in v) ** 0.5
        if nrm < 1e-12: nrm = 1.0
        g.write(struct.pack('<%df' % n_in, *[x/nrm for x in v]))
print('wrote %s  (%d x %d f32, %.1f MB)' % (out, N, n_in, os.path.getsize(out)/1e6))
