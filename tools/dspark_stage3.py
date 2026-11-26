#!/usr/bin/env python3
"""dspark_stage3.py - DSpark Markov head (stage 3) over the stage-2 block output.

Source: ink_dspark.cuh
  k_dspark_markov_prep (:573)    joint[5120] = [ state[256] | markov_w1[prev_tok][:256] | hidden[2304] | f_rnn_ctx[2304] ]
  k_dspark_markov_gru_upd (:604) proj = joint @ rnn_proj^T + rnn_proj.bias   -> [768]
                                 g = sigmoid(proj[0:256]); cand = tanh(proj[256:512])
                                 state = cand + g*(state - cand); lat_out = tanh(proj[512:768])
  k_dspark_add_bias_and_h (:620) corr = lat_out @ hidden_out^T + hidden_out.bias (+ hs optional)
  k_dspark_score_cand_eval(:630) score_i = dot(corr, unembed[cand_i])

rnn_proj is [768, 5120] Q8_ROW, hidden_out is [2304, 256] Q8_ROW, markov_w1 is [201024, 256] Q8_ROW (52 MB).
The unembed rows are not in this package - they live in the trunk (model output matrix) - so this tool stops at
`corr` and reports its statistics. Scoring needs those rows, which is the next step.

Usage: python dspark_stage3.py --weights DIR --taps DIR [--steps 7]
"""
import argparse, glob, os, sys
import numpy as np

H, R, STEPS = 2304, 256, 7
EPS = 1e-6
TAP_VISITS = (5, 23, 41, 47)


def load(w, name):
    p = os.path.join(w, name + '.f32')
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return np.fromfile(p, dtype='<f4').astype(np.float32)


def mat(w, name, rows, cols):
    a = load(w, name)
    if a.size != rows * cols:
        sys.exit(f"{name}: {a.size} != {rows}x{cols}")
    return a.reshape(rows, cols)


def rmsnorm(x, wt, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * wt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--taps', required=True)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--block', type=int, default=8)
    a = ap.parse_args()

    taps = []
    for v in TAP_VISITS:
        hits = sorted(glob.glob(os.path.join(a.taps, f'V{v:03d}_*.bin')))
        if not hits:
            sys.exit(f"no dump for visit {v}")
        taps.append(np.fromfile(hits[0], dtype='<f4').reshape(-1, H))
    M = min(t.shape[0] for t in taps)
    taps = np.stack([t[:M] for t in taps])

    w1 = mat(a.weights, 'dspark.markov_head.markov_w1.weight', 201024, R)
    rp = mat(a.weights, 'dspark.markov_head.rnn_proj.weight', 768, 5120)
    rpb = load(a.weights, 'dspark.markov_head.rnn_proj.bias')
    ho = mat(a.weights, 'dspark.markov_head.hidden_out.weight', H, R)
    hob = load(a.weights, 'dspark.markov_head.hidden_out.bias')
    print(f"markov_w1 {w1.shape}  rnn_proj {rp.shape}  hidden_out {ho.shape}")

    # Block output from stage 2 is not reproduced here; use the final tap as the 'hidden' feeding the head,
    # one row per slot. This exercises the head's maths, not the block half of the pipeline.
    hidden = np.repeat(taps[3][:1], a.block, axis=0)                 # [B, H]
    frnn = taps[3][:1]                                               # [1, H] context row
    state = np.zeros((a.block, R), dtype=np.float32)
    prev = np.zeros(a.block, dtype=np.int64)                          # anchor id placeholder (0)

    print(f"\n{'step':>4} {'state_rms':>10} {'state_absmax':>13} {'corr_rms':>10} {'corr_absmax':>12}")
    for s in range(a.steps):
        joint = np.concatenate([state, w1[prev], hidden, np.repeat(frnn, a.block, axis=0)], axis=1)  # [B,5120]
        if joint.shape[1] != 5120:
            sys.exit(f"joint {joint.shape} != [B,5120]")
        proj = joint @ rp.T + rpb                                        # [B,768]
        g = sigmoid(proj[:, :R])
        cand = np.tanh(proj[:, R:2 * R])
        state = cand + g * (state - cand)
        lat = np.tanh(proj[:, 2 * R:3 * R])
        corr = lat @ ho.T + hob                                          # [B, H]
        print(f"{s:>4} {np.sqrt((state**2).mean()):10.4f} {np.abs(state).max():13.4f} "
              f"{np.sqrt((corr**2).mean()):10.4f} {np.abs(corr).max():12.4f}")

    print(f"\nfinal state bounded in (-1,1): {np.abs(state).max() < 1.0}")
    print(f"corr finite: {np.isfinite(corr).all()}")


if __name__ == '__main__':
    main()
