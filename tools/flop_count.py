"""flop_count.py - what one token actually costs in MACs, and what that implies for prefill at long context.

Why this exists: the prefill wall turned out to be arithmetic, not memory or batching, so the load-bearing number is
GFLOP/token. Deriving it by hand is error-prone (an earlier version of this script matched `ffn_gate_up_exps.scale`
instead of the expert tensor and under-counted MoE by 25x), so read the shapes out of the GGUF and print the table.

Shapes that matter, from blk.N in Inkling-2-Mini-UD-IQ2_XXS.gguf:
    attn_q [2304,2304]  attn_k [2304,1024]  attn_v [2304,1024]  attn_output [3072,2304]  attn_gate [2304,3072]
    ffn_gate_up_exps [2304,2304,128]  -> per expert 2304x2304 (gate and up fused)
    ffn_down_exps    [1152,2304,128]  -> per expert 1152x2304
    (the loop_*.weight tensors are rank-64 slices, ~0.3M MAC each, negligible)

Usage: python flop_count.py [path-to-gguf]
"""
import struct
import sys

# defaults; overridden from the file's own metadata when it parses
D = 2304
D_FFE = 1152
N_USED = 8
N_LAYER = 48

ATTN = (('attn_q', 2304, 2304), ('attn_k', 2304, 1024), ('attn_v', 2304, 1024),
        ('attn_output', 3072, 2304), ('attn_gate', 2304, 3072))
GATE_UP = (2304, 2304)     # per expert, gate+up fused
DOWN = (1152, 2304)        # per expert


def read_meta(path):
    """Pull d_model / d_ff_expert / expert_used / block_count out of the GGUF, return them or the defaults."""
    try:
        f = open(path, 'rb')
    except OSError:
        return {}
    magic, ver, nt, nkv = struct.unpack('<IIQQ', f.read(24))
    T = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4),
         6: ('f', 4), 7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}

    def rs():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def rv(t):
        if t == 8:
            return rs()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            return [rv(et) for _ in range(n)]
        fmt, sz = T[t]
        return struct.unpack('<' + fmt, f.read(sz))[0]

    out = {}
    for _ in range(nkv):
        k = rs(); t = struct.unpack('<I', f.read(4))[0]; v = rv(t)
        if k in ('inkling2.embedding_length', 'inkling2.expert_feed_forward_length',
                 'inkling2.expert_used_count', 'inkling2.block_count',
                 'inkling2.attention.key_length', 'inkling2.attention.value_length'):
            out[k] = v
    f.close()
    return out


def main():
    global N_LAYER, N_USED
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf'
    m = read_meta(path)
    N_LAYER = m.get('inkling2.block_count', N_LAYER)
    N_USED = m.get('inkling2.expert_used_count', N_USED)

    attn = sum(r * c for _, r, c in ATTN)
    per_expert = GATE_UP[0] * GATE_UP[1] + DOWN[0] * DOWN[1]
    moe = N_USED * per_expert
    layer = attn + moe
    mac = layer * N_LAYER
    flop = 2.0 * mac

    print(f'layers={N_LAYER} experts_used={N_USED}')
    print(f'attention projections/layer  {attn / 1e6:7.2f} M MAC')
    print(f'{N_USED} experts/layer           {moe / 1e6:7.2f} M MAC  '
          f'(per expert gate_up {GATE_UP[0]*GATE_UP[1]/1e6:.2f} + down {DOWN[0]*DOWN[1]/1e6:.2f})')
    print(f'per layer                    {layer / 1e6:7.2f} M MAC')
    print(f'per token                    {mac / 1e9:7.2f} GMAC = {flop / 1e9:.2f} GFLOP')
    print()
    # measured: 0.195 s per prefilled position, 0.22 s per decoded token
    print(f'at 0.195 s/position (measured prefill)  -> {flop / 0.195 / 1e9:6.1f} GFLOPS sustained')
    print(f'at 0.22  s/token    (measured decode)   -> {flop / 0.22 / 1e9:6.1f} GFLOPS  (decode is memory-bound;'
          f' it moves 1.5 GB/token, so the same number is mostly idle cores)')
    print()
    print('prefill hours at a sustained 40 GFLOPS (this box measured 43.3, so 40 is not optimistic):')
    for n in (8192, 32768, 131072, 524288, 1048576):
        print(f'  {n:>9,} tokens: {flop * n / 40e9 / 3600:6.1f} h')


if __name__ == '__main__':
    main()
