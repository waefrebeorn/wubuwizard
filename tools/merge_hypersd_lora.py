#!/usr/bin/env python3
"""merge_hypersd_lora.py — merge a Hyper-SD / LCM / any SD1.5 diffusers LoRA
into a base GGUF, producing a NEW GGUF the wizard engine can load directly.

Why: wizard txt2img_sd loads GGUF only. Hyper-SD15-1step-lora gives 1-step
inference (with TCD cfg=0) but is a diffusers-format LoRA. This tool
dequantizes the LoRA-touched UNet tensors, applies
W += (alpha/rank)*scale*(lora_up @ lora_down), re-quantizes them to Q8_0
(the engine fully supports Q8_0), and writes a merged GGUF with the KV
section preserved and tensor offsets rebuilt.

Usage:
  merge_hypersd_lora.py base.gguf lora.safetensors out.gguf [scale]
"""
import struct, json, sys
import numpy as np

# ---------- GGML dequant / quant (types the base GGUF uses) ----------
def dequant_q4_0(b, n):
    nb = n // 32
    bb = np.frombuffer(b, np.uint8)[:nb * 18].reshape(nb, 18)
    d = bb[:, :2].copy().view(np.float16).astype(np.float32)[:, 0]
    qbytes = bb[:, 2:18]                                   # [nb, 16]
    j = np.arange(32)
    nib = (qbytes[:, j // 2] >> (4 * (j % 2))) & 0x0F
    return ((nib.astype(np.int16) - 8).astype(np.float32) * d[:, None]).reshape(-1)

def dequant_q8_0(b, n):
    nb = n // 32
    bb = np.frombuffer(b, np.uint8)[:nb * 34].reshape(nb, 34)
    d = bb[:, :2].copy().view(np.float16).astype(np.float32)[:, 0]
    q = bb[:, 2:34].astype(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1)

def dequant_f16(b, n):
    return np.frombuffer(b, np.float16, n).astype(np.float32)

def quant_q8_0(x):
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    amax = np.max(np.abs(xb), axis=1)
    d = np.where(amax > 0, amax / 127.0, 0.0)
    q = np.clip(np.round(np.where(d[:, None] > 0, xb / d[:, None], 0.0)),
                -128, 127).astype(np.int8)
    out = np.empty(nb * 34, np.uint8)
    out[:, ] = 0
    # interleave: 2-byte fp16 scale + 32 int8
    out = bytearray()
    for i in range(nb):
        out += np.float16(d[i]).tobytes()
        out += q[i].tobytes()
    return bytes(out)

DEQ = {1: dequant_f16, 2: dequant_q4_0, 8: dequant_q8_0}
Q8_0 = 8

# ---------- GGUF low-level ----------
GGUF_TYPES = {0:'UINT8',1:'INT8',2:'UINT16',3:'INT16',4:'UINT32',5:'INT32',
              6:'FLOAT32',7:'BOOL',8:'STRING',9:'ARRAY',10:'UINT64',11:'INT64',
              12:'FLOAT64'}
GGML_TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',
              9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',
              15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',
              20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS'}

def w_str(s):
    b = s.encode(); return struct.pack('<Q', len(b)) + b

def w_value(typ, val):
    if typ == 0: return struct.pack('<B', val)
    if typ == 1: return struct.pack('<b', val)
    if typ == 2: return struct.pack('<H', val)
    if typ == 3: return struct.pack('<h', val)
    if typ == 4: return struct.pack('<I', val)
    if typ == 5: return struct.pack('<i', val)
    if typ == 6: return struct.pack('<f', val)
    if typ == 7: return struct.pack('<?', val)
    if typ == 8: return w_str(val)
    if typ == 10: return struct.pack('<Q', val)
    if typ == 11: return struct.pack('<q', val)
    if typ == 12: return struct.pack('<d', val)
    if typ == 9:
        at = val[0]; n = val[1]
        out = struct.pack('<i', at) + struct.pack('<Q', len(n))
        for e in n: out += w_value(at, e)
        return out
    raise ValueError(typ)

def read_gguf(path):
    """Return (tensors, kv_bytes, nt, nk, header_end, alignment). Does NOT
    read the data blob (1.5GB) — callers read it once and keep a single
    reference to bound memory on the 5.8GB box."""
    f = open(path, 'rb')
    magic = f.read(4); assert magic == b'GGUF', magic
    ver = struct.unpack('<I', f.read(4))[0]
    nt = struct.unpack('<Q', f.read(8))[0]
    nk = struct.unpack('<Q', f.read(8))[0]
    kv_start = f.tell()
    for _ in range(nk):
        k = read_str(f)
        typ = struct.unpack('<i', f.read(4))[0]
        skip_value(f, typ)
    kv_end = f.tell()
    kv_bytes = open(path, 'rb').read()[kv_start:kv_end]
    tensors = []
    for _ in range(nt):
        name = read_str(f)
        nd = struct.unpack('<I', f.read(4))[0]
        dims = tuple(struct.unpack('<q', f.read(8))[0] for _ in range(nd))
        gtype = struct.unpack('<i', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        tensors.append({'name': name, 'dims': dims, 'type': gtype, 'offset': off})
    header_end = f.tell()
    f.close()
    return tensors, kv_bytes, nt, nk, header_end, 32

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode()

def skip_value(f, typ):
    if typ == 8: read_str(f)
    elif typ == 9:
        at = struct.unpack('<i', f.read(4))[0]; al = struct.unpack('<Q', f.read(8))[0]
        for _ in range(al):
            if at == 8: read_str(f)
            elif at in (0,1,2,3,7): f.read(1)
            elif at in (4,5): f.read(4)
            elif at in (10,11): f.read(8)
            elif at in (6,12): f.read(4)
            else: raise ValueError(f"arr elem {at}")
    elif typ in (0,1,2,3,7): f.read(1)
    elif typ in (4,5): f.read(4)
    elif typ in (10,11): f.read(8)
    elif typ in (6,12): f.read(4)
    else: raise ValueError(f"KV type {typ}")

# ---------- safetensors LoRA ----------
def read_safetensors(path):
    d = open(path, 'rb').read()
    n = struct.unpack('<Q', d[:8])[0]
    hdr = json.loads(d[8:8+n].decode())
    base = 8 + n; out = {}
    for k, m in hdr.items():
        if k == '__metadata__': continue
        a, b = m['data_offsets']; shape = tuple(m['shape']); t = m['dtype']
        raw = d[base+a:base+b]
        if t == 'F32': out[k] = np.frombuffer(raw, np.float32).reshape(shape)
        elif t == 'F16': out[k] = np.frombuffer(raw, np.float16).reshape(shape).astype(np.float32)
        elif t == 'BF16':
            out[k] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        else: raise ValueError(t)
    return out

# ---------- diffusers -> LDM name mapping (SD1.5) ----------
def load_lora_deltas(lora, scale):
    from collections import defaultdict
    ups = defaultdict(dict)
    for k, v in lora.items():
        if not k.startswith('lora_unet_'): continue
        if '.lora_up.weight' in k: ups[k.replace('.lora_up.weight','')]['up'] = v
        elif '.lora_down.weight' in k: ups[k.replace('.lora_down.weight','')]['down'] = v
        elif '.alpha' in k: ups[k.replace('.alpha','')]['alpha'] = float(v)
    deltas = {}
    for base, d in ups.items():
        if 'up' not in d or 'down' not in d: continue
        up, down = d['up'], d['down']           # up [out,r,...], down [r,in,...]
        alpha = d.get('alpha', up.shape[0] if up.ndim >= 2 else 64)
        rank = down.shape[0]
        # generic einsum over the rank axis, keeping spatial dims of `down`
        # up: [out, r, 1, 1] (squeeze to [out,r]); down: [r, in, kh, kw]
        # -> delta [out, in, kh, kw] = sum_r up[out,r]*down[r,in,kh,kw]
        upm = up.reshape(up.shape[0], up.shape[1]) if up.ndim >= 3 else up
        down_nd = down.ndim
        # build labels: down dims = r, i, s0, s1, ... ; up = o, r
        dow = ['r', 'i'] + [chr(ord('a') + j) for j in range(down_nd - 2)]
        out_lbl = ['o', 'i'] + [chr(ord('a') + j) for j in range(down_nd - 2)]
        delta = np.einsum('or,' + ''.join(dow) + '->' + ''.join(out_lbl),
                          upm, down) * (alpha / rank) * scale
        gg = map_base_to_gguf(base[len('lora_unet_'):])
        if gg: deltas[gg] = delta
    return deltas

# Composite tokens that must NOT be split on '_' (diffusers names use '_'
# where the GGUF/LDM naming keeps them as single dotted components).
_COMPOSITE = ['transformer_blocks', 'proj_in', 'proj_out', 'attn1', 'attn2',
              'to_out', 'to_q', 'to_k', 'to_v', 'conv_in', 'conv_norm',
              'conv_out', 'time_embed', 'label_emb', 'skip_connection',
              'time_emb_proj', 'conv_shortcut', 'conv_norm2']

def split_tokens(base):
    """Split a diffusers path into tokens, keeping composite names whole.
    e.g. 'down_blocks_0_attentions_0_proj_in' ->
         ['down', 'blocks', '0', 'attentions', '0', 'proj_in']"""
    out = []
    rest = base
    while rest:
        rest = rest.lstrip('_')
        if not rest: break
        found = None
        for c in _COMPOSITE:
            if rest == c or rest.startswith(c + '_'):
                out.append(c); rest = rest[len(c):]; found = True; break
        if found: continue
        # single segment up to next '_'
        if '_' in rest:
            tok, rest = rest.split('_', 1); out.append(tok)
        else:
            out.append(rest); rest = ''
    return out

# diffusers resnet sub-name -> LDM/GGUF sub-name (SD1.5 ResBlock)
_RESNET_SUB = {
    'conv1': 'in_layers.2',
    'conv2': 'out_layers.3',
    'conv_shortcut': 'skip_connection',
    'time_emb_proj': 'emb_layers.1',
    'conv_norm': 'in_layers.0',
    'conv_norm2': 'out_layers.0',
}

def map_resnet_tail(tokens):
    """tokens = the sub-name tokens of a resnet (e.g. ['conv1']) plus any
    following .weight-free parts. Map the first to the LDM ResBlock name."""
    if not tokens: return ''
    first = tokens[0]
    if first in _RESNET_SUB:
        mapped = _RESNET_SUB[first]
        return mapped + ('.' + '.'.join(tokens[1:]) if len(tokens) > 1 else '')
    return '.'.join(tokens)

def map_base_to_gguf(base):
    """base like 'down_blocks_0_attentions_0_proj_in' (diffusers path,
    underscore-delimited with composite tokens). Map to LDM/GGUF name.
    Standard SD1.5 UNet diffusers->LDM mapping:
      down_blocks.b.resnets.r   -> input_blocks.{1+3b+r}.0
      down_blocks.b.attentions.a-> input_blocks.{1+3b+a}.1
      down_blocks.b.downsamplers-> input_blocks.{3+3b}.0.op
      up_blocks.u.resnets.r     -> output_blocks.{3(3-u)+r}.0
      up_blocks.u.attentions.a  -> output_blocks.{3(3-u)+a}.1
      up_blocks.u.upsamplers    -> output_blocks.{2+3u}.1.conv
      mid_block.resnets.0/1/2   -> middle_block.0/2 (resnets 1->2)
      mid_block.attentions      -> middle_block.1
    """
    p = split_tokens(base)
    if len(p) >= 2 and p[0] == 'down' and p[1] == 'blocks':
        b = int(p[2]); rest = p[3:]
        if rest[0] == 'attentions':
            a = int(rest[1]); ib = 1 + 3*b + a
            return f"model.diffusion_model.input_blocks.{ib}.1." + '.'.join(rest[2:])
        elif rest[0] == 'resnets':
            r = int(rest[1]); ib = 1 + 3*b + r
            return f"model.diffusion_model.input_blocks.{ib}.0." + map_resnet_tail(rest[2:])
        elif rest[0] == 'downsamplers':
            return f"model.diffusion_model.input_blocks.{3 + 3*b}.0.op"
    if len(p) >= 2 and p[0] == 'mid' and p[1] == 'block':
        rest = p[2:]
        if rest[0] == 'attentions':
            # diffusers mid_block.attentions -> middle_block.1 ; the next
            # token is the (single) attention index — drop it, keep the rest
            idx = int(rest[1]) if len(rest) > 1 else 0
            tail = '.'.join(rest[2:]) if len(rest) > 2 else ''
            return "model.diffusion_model.middle_block.1." + tail
        elif rest[0] == 'resnets':
            r = int(rest[1]); sub = 0 if r == 0 else 2
            return f"model.diffusion_model.middle_block.{sub}." + map_resnet_tail(rest[2:])
    if len(p) >= 2 and p[0] == 'up' and p[1] == 'blocks':
        b = int(p[2]); rest = p[3:]
        # SD1.5 output_blocks 0,1 have NO attention. diffusers up_blocks.u
        # (u=3 deepest, u=0 shallowest) map to output_blocks:
        #   u=3 -> 9,10,11   u=2 -> 6,7,8   u=1 -> 3,4,5   u=0 -> 0,1,2
        obase = [0, 3, 6, 9][b]
        if rest[0] == 'attentions':
            a = int(rest[1]); ob = obase + a
            return f"model.diffusion_model.output_blocks.{ob}.1." + '.'.join(rest[2:])
        elif rest[0] == 'resnets':
            r = int(rest[1]); ob = obase + r
            return f"model.diffusion_model.output_blocks.{ob}.0." + map_resnet_tail(rest[2:])
        elif rest[0] == 'upsamplers':
            # upsamplers: u=0->output_blocks.2.1.conv, u=1->5.2.conv,
            # u=2->8.2.conv (u=3 has no upsampler). Sub-index varies: .1 for
            # block 2, .2 for blocks 5/8.
            ob = 2 + 3*b
            sub = 1 if b == 0 else 2
            return f"model.diffusion_model.output_blocks.{ob}.{sub}.conv"
    return None

# ---------- write a valid GGUF ----------
def write_gguf(path, tensors, nt, nk, kv_bytes, alignment, data_payload):
    """Stream the merged GGUF to disk incrementally (never buffer the whole
    output in RAM — a 774MB model + header + per-tensor data OOMs a 5.8GB
    box). Returns 0 ok."""
    f = open(path, 'wb')
    f.write(b'GGUF')
    f.write(struct.pack('<I', 3))
    f.write(struct.pack('<Q', nt))
    f.write(struct.pack('<Q', nk))
    f.write(kv_bytes)
    # tensor info: compute offsets first (need data lengths)
    off = 0
    infos = []
    for t in tensors:
        blen = len(t['data'])
        infos.append((t, off))
        off += blen
    # build info section
    info = bytearray()
    for t, off in infos:
        info += w_str(t['name'])
        info += struct.pack('<I', len(t['dims']))
        for d in t['dims']: info += struct.pack('<q', d)
        info += struct.pack('<i', t['type'])
        info += struct.pack('<Q', off)
    while (f.tell() + len(info)) % alignment != 0:
        info += b'\x00'
    f.write(bytes(info))
    # stream tensor data, freeing each after write
    for t in tensors:
        f.write(t['data'])
        del t['data']
    f.close()
    return 0

def lora_triples(lora):
    """Yield (gguf_name, delta) for each complete LoRA triple, computing
    and freeing each delta one at a time to bound memory (the 3x3 conv
    deltas are large; holding all 278 at once OOMs a 5.8GB box)."""
    from collections import defaultdict
    ups = defaultdict(dict)
    for k, v in lora.items():
        if not k.startswith('lora_unet_'): continue
        if '.lora_up.weight' in k: ups[k.replace('.lora_up.weight','')]['up'] = v
        elif '.lora_down.weight' in k: ups[k.replace('.lora_down.weight','')]['down'] = v
        elif '.alpha' in k: ups[k.replace('.alpha','')]['alpha'] = float(v)
    for base, d in ups.items():
        if 'up' not in d or 'down' not in d: continue
        up, down = d['up'], d['down']
        alpha = d.get('alpha', up.shape[0] if up.ndim >= 2 else 64)
        rank = down.shape[0]
        upm = up.reshape(up.shape[0], up.shape[1]) if up.ndim >= 3 else up
        down_nd = down.ndim
        dow = ['r', 'i'] + [chr(ord('a') + j) for j in range(down_nd - 2)]
        out_lbl = ['o', 'i'] + [chr(ord('a') + j) for j in range(down_nd - 2)]
        delta = np.einsum('or,' + ''.join(dow) + '->' + ''.join(out_lbl),
                          upm, down) * (alpha / rank)
        gg = map_base_to_gguf(base[len('lora_unet_'):])
        if gg: yield gg, delta
        del delta

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    base_gguf, lora_path, out_gguf = sys.argv[1], sys.argv[2], sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    tensors, kv_bytes, nt, nk, header_end, alignment = read_gguf(base_gguf)
    blob = open(base_gguf, 'rb').read()
    data_base = header_end
    pad = (alignment - (header_end % alignment)) % alignment
    data_base += pad
    lora = read_safetensors(lora_path)
    name2idx = {t['name']: i for i, t in enumerate(tensors)}
    napplied = 0; nskipped = 0
    for gg, delta in lora_triples(lora):
        ggw = gg + '.weight'          # LoRA targets the .weight tensor
        if ggw not in name2idx:
            nskipped += 1; continue
        t = tensors[name2idx[ggw]]
        if t['type'] not in DEQ:
            print(f"[merge] skip {gg}: type {GGML_TYPES.get(t['type'],t['type'])} unhandled"); nskipped += 1; continue
        n = int(np.prod(t['dims']))
        raw = blob[data_base + t['offset']: data_base + t['offset'] + n*2]
        W = DEQ[t['type']](raw, n).reshape(t['dims'])
        W2 = W + scale * delta.reshape(t['dims'])
        if t['type'] == 1:  # F16 stays F16 (no data-size change)
            t['data'] = W2.astype(np.float16).tobytes()
        else:
            t['data'] = W2.astype(np.float16).tobytes()  # requant to F16
            t['type'] = 1
        napplied += 1
        del W, W2, delta
    # copy untouched tensor data as memoryview refs into the single blob
    for i, t in enumerate(tensors):
        if 'data' not in t:
            nbytes = nbytes_for(t)
            t['data'] = memoryview(blob)[data_base + t['offset']: data_base + t['offset'] + nbytes]
        if i % 200 == 0:
            print(f"[merge] prepared {i}/{len(tensors)} tensors", flush=True)
    print(f"[merge] applied {napplied}, untouched {len(tensors)-napplied}, unmapped {nskipped}", flush=True)
    write_gguf(out_gguf, tensors, nt, nk, kv_bytes, alignment, None)
    print(f"[merge] wrote {out_gguf}")

def nbytes_for(t):
    # element count * bytes (F16=2, Q4_0=18/32, Q8_0=34/32 ...)
    n = int(np.prod(t['dims']))
    typ = t['type']
    if typ == 1: return n * 2
    if typ == 2: return (n // 32) * 18
    if typ == 8: return (n // 32) * 34
    raise ValueError(f"untouched type {typ} byte-size unknown")

if __name__ == '__main__':
    main()
