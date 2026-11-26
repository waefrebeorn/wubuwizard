#!/usr/bin/env python3
"""match_taps.py - identify which trunk states the reference's four DSpark taps actually are.

The instrumented build dumps upstream's `inp_target_features` as [n_target=4, n_embd=2304, n_tokens], i.e. the exact
features the drafter consumes. Our engine dumps per-layer/per-pass residuals (`L%02d_p%d.bin`, [n_tokens, 2304]).
Correlating each reference tap against every candidate answers "which states are the taps" from the reference itself,
instead of inferring it from visit arithmetic.
"""
import argparse, glob, os, re, sys

import numpy as np


def load_features(path):
    a = np.fromfile(path, dtype='<f4')
    # ggml order: ne0=4 (tap) fastest, then 2304 (embd), then n_tokens
    n = a.size
    n_tok = n // (4 * 2304)
    x = a.reshape(n_tok, 2304, 4)
    return np.ascontiguousarray(np.moveaxis(x, 2, 0))   # [4, n_tok, 2304]


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--dirs', required=True, help='comma-separated directories of L%02d_p%d.bin')
    ap.add_argument('--n', type=int, default=0, help='positions to use (0 = min available)')
    a = ap.parse_args()

    F = load_features(a.feats)
    n_tap, nt, n_embd = F.shape
    print(f"reference taps: {n_tap} x {nt} positions x {n_embd}")
    print(f"reference rms per tap: {[round(float(np.sqrt((F[j].astype(np.float64)**2).mean())), 5) for j in range(n_tap)]}")

    cands = []
    for d in a.dirs.split(','):
        for p in sorted(glob.glob(os.path.join(d, 'L*_p*.bin'))):
            m = re.match(r'L(\d+)_p(\d+)\.bin$', os.path.basename(p))
            if m:
                cands.append((int(m.group(1)), int(m.group(2)), p))
    print(f"{len(cands)} candidate dumps")

    n = a.n or nt
    rows = []
    for layer, ps, path in cands:
        v = np.fromfile(path, dtype='<f4')
        if v.size % n_embd:
            continue
        v = v.reshape(-1, n_embd)
        if v.shape[0] < n:
            continue
        for j in range(n_tap):
            rows.append((cos(F[j, :n], v[:n]), j, layer, ps, path))
    rows.sort(reverse=True)

    for j in range(n_tap):
        print(f"\n-- tap {j} best matches --")
        shown = 0
        for c, jj, layer, ps, path in rows:
            if jj != j:
                continue
            print(f"  cos {c:+.6f}  L{layer:02d}_p{ps}  {os.path.basename(path)}")
            shown += 1
            if shown >= 4:
                break


if __name__ == '__main__':
    main()
