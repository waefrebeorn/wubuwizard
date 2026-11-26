#!/usr/bin/env python3
"""dspark_ref.py - reference implementation of DSpark stage 1 (context fusion).

Source of truth: strata engine/src/inkling2/cuda/ink_dspark.cuh, k_dspark_mix_taps (:52-86).

    for each of the 4 fuse rows:  w = softmax(fuse_w[row][0..3])
    v[d] = sum_j w[j] * tap_j[d]
    out  = RMSNorm(v; norm_w)          inv = rsqrt(mean(v^2) + eps)

Taps are the post-layer hidden states at visits 5, 23, 41 and 47 (see docs/research/dspark-port-plan.md;
visit 48 in `target_layer_ids` is the final state after the loop, which is visit 47's output plus the
final norm in our engine).

Usage:
  python dspark_ref.py --weights DIR --taps DIR           # print the 4 fused contexts' statistics
  python dspark_ref.py --weights DIR --taps DIR --dump OUT.npy
"""
import argparse, glob, os, sys
import numpy as np

H = 2304
TAP_VISITS = (5, 23, 41, 47)


def load(weights, name):
    p = os.path.join(weights, name + '.f32')
    if not os.path.exists(p):
        sys.exit(f"missing tensor: {p}")
    return np.fromfile(p, dtype='<f4')


def load_taps(taps_dir, visits=TAP_VISITS):
    """Load one hidden state per visit. The visit dumper writes V%03d_il%02d_p%d.bin, and the pass
    suffix varies per visit (V023 is p1, V041 is p2), so match on the visit number only."""
    out = []
    for v in visits:
        hits = sorted(glob.glob(os.path.join(taps_dir, f'V{v:03d}_*.bin')))
        if not hits:
            sys.exit(f"no dump for visit {v} in {taps_dir}")
        a = np.fromfile(hits[0], dtype='<f4')
        if a.size % H:
            sys.exit(f"visit {v}: {a.size} floats is not a multiple of {H}")
        out.append(a.reshape(-1, H))
    return out


def fuse_taps(taps, fuse, norm, eps=1e-6):
    """taps: list of [M, H]. Returns [4, M, H] - one fused context per fuse_weight row."""
    M = min(t.shape[0] for t in taps)
    taps = np.stack([t[:M] for t in taps])            # [4, M, H]
    outs = np.empty((4, M, H), dtype=np.float32)
    weights = []
    pre_rms = []
    for r in range(4):
        fw = fuse[r * 4:(r + 1) * 4]
        e = np.exp(fw - fw.max())
        w = e / e.sum()
        weights.append(w)
        v = np.tensordot(w.astype(np.float32), taps, axes=(0, 0))     # [M, H]
        ms = (v * v).mean(-1, keepdims=True)
        pre_rms.append(float(np.sqrt(ms.mean())))                     # true RMS before normalising
        outs[r] = v / np.sqrt(ms + eps) * norm
    return outs, weights, pre_rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--taps', required=True)
    ap.add_argument('--visits', default=','.join(map(str, TAP_VISITS)))
    ap.add_argument('--dump')
    a = ap.parse_args()

    visits = tuple(int(x) for x in a.visits.split(','))
    fuse = load(a.weights, 'dspark.fuse_weight')
    norm = load(a.weights, 'dspark.norm.weight')
    taps = load_taps(a.taps, visits)

    for v, t in zip(visits, taps):
        print(f"visit {v:02d}: {t.shape[0]} pos x {t.shape[1]}")
    outs, weights, pre_rms = fuse_taps(taps, fuse, norm)

    rms_norm = float(np.sqrt((norm ** 2).mean()))
    print(f"\nrms(norm_w) = {rms_norm:.4f}   <- rms_post should land near this")
    print(f"{'row':>3}  {'softmax(fuse_w[row])':<42} {'rms_pre':>8} {'rms_post':>9}")
    for r in range(4):
        post = float(np.sqrt((outs[r] ** 2).mean()))
        ws = '[' + ', '.join(f'{x:.3f}' for x in weights[r]) + ']'
        print(f"{r:>3}  {ws:<42} {pre_rms[r]:8.4f} {post:9.4f}")

    if a.dump:
        np.save(a.dump, outs)
        print(f"\nwrote {outs.shape} {outs.dtype} -> {a.dump}")


if __name__ == '__main__':
    main()
