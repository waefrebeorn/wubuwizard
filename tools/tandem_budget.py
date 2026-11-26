"""Tandem placement budget: what actually fits in 5920 MiB of VRAM, from the GGUF's own tensor bytes.

Strata/Avalon/maya all split the same way: the small always-read tiers (attention projections, routers, shared
experts, output head, KV) live in VRAM, and the big expert population is shared between a VRAM expert cache and the
CPU. This measures the tiers for OUR file instead of assuming, so the split is arithmetic.
"""
import struct
import sys

P = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf'
TT = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 4, 9: None, 10: 8, 11: 8, 12: 8}
NELEM = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 10: 1, 11: 1, 12: 1}

# ggml type -> bytes per block, block elems (per ggml-common.h static_asserts / this checkout's enum)
BLK = {0: (1, 1), 1: (1, 1), 2: (34, 32), 3: (18, 32), 6: (210, 256), 7: (144, 256), 8: (34, 32),
       9: (176, 256), 10: (82, 256), 11: (110, 256), 12: (34, 32), 13: (50, 256), 14: (34, 32),
       16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256),
       22: (82, 256), 23: (136, 256), 29: (56, 256), 30: (34, 32), 39: (18, 32), 40: (20, 32),
       41: (18, 32), 42: (34, 32)}


def main():
    f = open(P, 'rb')
    magic, ver, nt, nkv = struct.unpack('<IIQQ', f.read(24))

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
        fmt = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[t]
        return struct.unpack('<' + fmt, f.read(TT[t]))[0]

    for _ in range(nkv):
        k = rs(); t = struct.unpack('<I', f.read(4))[0]; rv(t)

    tiers = {'attention': 0, 'router+shared': 0, 'experts': 0, 'head+embed': 0, 'norms/scales/small': 0}
    by_type = {}
    for _ in range(nt):
        nm = rs()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack('<I', f.read(4))[0]
        struct.unpack('<Q', f.read(8))[0]
        elems = 1
        for x in dims:
            elems *= x
        bb, be = BLK.get(ty, (1, 1))
        nbytes = (elems // be) * bb
        by_type[ty] = by_type.get(ty, 0) + nbytes
        if 'exps' in nm:
            tiers['experts'] += nbytes
        elif nm.startswith('blk.') and ('attn_' in nm and 'res_query' not in nm):
            tiers['attention'] += nbytes
        elif nm.startswith('blk.') and ('gate_inp' in nm or 'shexp' in nm or 'gate_up_scale' in nm or 'down_scale' in nm):
            tiers['router+shared'] += nbytes
        elif nm.startswith('token_embd') or nm.startswith('output'):
            tiers['head+embed'] += nbytes
        else:
            tiers['norms/scales/small'] += nbytes

    total = sum(tiers.values())
    print(f'{P.split("/")[-1]}   total {total/1048576:.0f} MiB')
    print()
    for k, v in sorted(tiers.items(), key=lambda kv: -kv[1]):
        print(f'  {k:22s} {v/1048576:8.0f} MiB  {100.0*v/total:5.1f}%')

    VRAM = 5920.0
    dense = tiers['attention'] + tiers['router+shared'] + tiers['head+embed'] + tiers['norms/scales/small']
    print(f'\nnon-expert (always needed) = {dense/1048576:.0f} MiB')
    print(f'VRAM free                  = {VRAM:.0f} MiB')
    print(f'left for expert cache + KV = {(VRAM - dense/1048576):.0f} MiB')
    kv_full = 8 * 1024 * 1024 * 2 * 4 / 1048576  # 8 full visits, 1024 positions, K+V, f32
    print(f'  KV at 1k ctx (8 full)    = {kv_full:.0f} MiB ; at 16k = {kv_full*16:.0f} MiB')
    ex = tiers['experts'] / 1048576
    print(f'expert population          = {ex:.0f} MiB, so a cache of that size holds '
          f'{100.0*(VRAM-dense/1048576)/ex:.0f}% of it')
    print()
    print('per-token read, measured elsewhere in this repo: dense 696 MB, experts 369 MB, head 469 MB')
    print('PCIe 4.0 x8 on this laptop: ~10-13 GB/s real, so a 1 MB expert slab costs ~80-100 us each way')


main()
