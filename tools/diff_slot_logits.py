#!/usr/bin/env python3
"""diff_slot_logits.py - compare our dflash forward's per-slot logits against the reference's.

The instrumented build dumps the drafter's own output as logits_seq_<seq>_<slot> [200058]. This runs our forward on the
same context rows and the same taps, then reports cosine and top-5 per slot, which localises the error to a stage
rather than to "the drafter".
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dflash_ref import Drafter, forward, N_EMBD  # noqa: E402


def topk(v, k=5):
    idx = np.argsort(-v)[:k]
    return [(int(i), float(v[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='C:/Users/eman5/models/inkling-2-drafter/f32')
    ap.add_argument('--gguf', default='C:/Users/eman5/models/inkling-2-drafter/dspark-Inkling-2-Mini-Q8_0.gguf')
    ap.add_argument('--unembed', required=True)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--taps', required=True, help='comma list, in reference order')
    ap.add_argument('--ref', required=True, help='dump dir with logits_seq_0_*')
    ap.add_argument('--n', type=int, default=259, help='context rows to use (reference chunk)')
    a = ap.parse_args()

    un = np.memmap(a.unembed, dtype='<f4', mode='r').reshape(-1, N_EMBD)[:200058]
    dr = Drafter(a.dir, un, gguf=a.gguf, verbose=False)

    taps = [np.fromfile(p, dtype='<f4').reshape(-1, N_EMBD)[:a.n] for p in a.taps.split(',')]
    taps = np.stack(taps)
    emb = np.fromfile(a.emb, dtype='<f4')[:N_EMBD]
    ctx_pos = np.arange(a.n)
    blk_pos = (a.n - 1) + np.arange(8)
    h, base, f_rnn = forward(dr, emb, taps, ctx_pos, blk_pos)
    print(f"our base {base.shape} rms {np.sqrt((base.astype(np.float64)**2).mean()):.4f}")

    for slot in range(8):
        p = os.path.join(a.ref, f'logits_seq_0_{slot}.bin')
        if not os.path.exists(p):
            print(f"slot {slot}: no reference dump")
            continue
        ref = np.fromfile(p, dtype='<f4')
        ours = base[slot]
        c = float(ours.astype(np.float64) @ ref.astype(np.float64) /
                  (np.linalg.norm(ours.astype(np.float64)) * np.linalg.norm(ref.astype(np.float64))))
        print(f"\nslot {slot}: cos {c:+.6f}   ref rms {np.sqrt((ref.astype(np.float64)**2).mean()):.4f}  "
              f"ours rms {np.sqrt((ours.astype(np.float64)**2).mean()):.4f}")
        print(f"   ref  top5 {topk(ref)}")
        print(f"   ours top5 {topk(ours)}")


if __name__ == '__main__':
    main()
