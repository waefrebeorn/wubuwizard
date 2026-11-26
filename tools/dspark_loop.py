#!/usr/bin/env python3
"""dspark_loop.py - 7-step DSpark drafting loop, measured against the engine's own continuation.

Each step: corr_i = Markov(state_i, markov_w1[prev_i], block_out[i], f_rnn) and
final_i = base(block_out[i]) + corr_i, where base and the unembed dot are computed by the engine
(IN2_BASE_VEC / IN2_ADD_VEC) so the arithmetic is the engine's own kernel, not a numpy reimplementation.

prev_0 is the anchor token; prev_{i+1} is the token drafted at step i - the loop feeds itself, which is the
whole point: an error at step 0 propagates.

Reports drafted ids against the known continuation and the accepted-prefix length.
"""
import glob, os, re, subprocess, sys
import numpy as np

sys.path.insert(0, r'C:/Users/eman5/WuBu/repos/wubuwizard/tools')
import dspark_stage2 as s2

W = r'C:/Users/eman5/models/inkling-2-strata/dspark-f32'
TAPD = r'C:/Users/eman5/AppData/Local/Temp/vtap'
EXE = r'C:/Users/eman5/WuBu/repos/wubuwizard/gen_fast.exe'
TMP = r'C:/Users/eman5/AppData/Local/Temp'
H, R, B, STEPS = 2304, 256, 8, 7
ANCHOR = 149879
ORACLE = [149876, 220, 36838, 182189, 149933, 220, 50843, 11]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def engine_top1(base_vec, corr_vec):
    np.ascontiguousarray(base_vec, dtype='<f4').tofile(os.path.join(TMP, 'hv.bin'))
    np.ascontiguousarray(corr_vec, dtype='<f4').tofile(os.path.join(TMP, 'cv.bin'))
    env = dict(os.environ, IN2_BASE_VEC=os.path.join(TMP, 'hv.bin'), IN2_ADD_VEC=os.path.join(TMP, 'cv.bin'))
    r = subprocess.run([EXE, '1', '371', '151451', '149879'], capture_output=True, text=True, env=env)
    m = re.search(r'ADD5\s+(-?\d+):([-0-9.]+)', r.stderr)
    if not m:
        sys.exit(f"no ADD5 line; stderr tail:\n{r.stderr[-400:]}")
    return int(m.group(1)), float(m.group(2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--taps', default=TAPD)
    ap.add_argument('--oracle', default=','.join(map(str, ORACLE)))
    ap.add_argument('--anchor', type=int, default=ANCHOR)
    ap.add_argument('--embd', default=os.path.join(TMP, 'our_embd.bin'))
    a = ap.parse_args()
    tapd = a.taps
    oracle = [int(x) for x in a.oracle.split(',')]
    anchor = a.anchor
    taps = []
    for v in (5, 23, 41, 47):
        p = sorted(glob.glob(os.path.join(tapd, f'V{v:03d}_*.bin')))
        taps.append(np.fromfile(p[0], dtype='<f4').reshape(-1, H))
    M = min(t.shape[0] for t in taps)
    taps = np.stack([t[:M] for t in taps])
    fuse = s2.load(W, 'dspark.fuse_weight')
    mask = s2.load(W, 'dspark.mask_embed')
    # Block init exactly as emit_dspark_backbone does it: slot 0 is the anchor token embedded and passed
    # through token_embd_norm, slots 1..B-1 are mask_embed. The anchor embedding comes from the engine's own
    # post-embedding dump (IN2_DUMP_EMBD with the anchor as the only input token).
    anchor_emb = np.fromfile(a.embd, dtype='<f4')
    assert anchor_emb.size == H, anchor_emb.size
    x0 = np.repeat(mask[None, :], B, axis=0)
    x0[0] = anchor_emb
    # Context is [c0, anchor_pos): the anchor's own position is NOT in the context - it is block slot 0.
    # ink_cpu_backend.cpp:1103-1105 c_len = anchor_pos - c0, and the block supplies positions anchor_pos+0..B-1.
    anchor_pos = M - 1
    taps_ctx = taps[:, :anchor_pos]                 # positions 0..anchor_pos-1
    h = s2.forward(x0, taps_ctx, fuse, W, np.arange(anchor_pos, anchor_pos + B), np.arange(anchor_pos), B)
    # Final norm before the head: dspark.norm is unused elsewhere in this pipeline and RMSNorm with its weight
    # (rms 3.09, matching the trunk taps' own 2.9-7.7) is what maps the block stream into the hidden distribution
    # the unembed matrix expects. IN2_DSPARK_POSTNORM=0 disables it for an A/B.
    if os.environ.get('IN2_DSPARK_POSTNORM', '1') != '0':
        h = s2.rmsnorm(h, s2.load(W, 'dspark.norm.weight'))
    w1 = s2.mat(W, 'dspark.markov_head.markov_w1.weight', 201024, R)
    rp = s2.mat(W, 'dspark.markov_head.rnn_proj.weight', 768, 5120)
    rpb = s2.load(W, 'dspark.markov_head.rnn_proj.bias')
    ho = s2.mat(W, 'dspark.markov_head.hidden_out.weight', H, R)
    hob = s2.load(W, 'dspark.markov_head.hidden_out.bias')
    fw = fuse[12:16]
    e = np.exp(fw - fw.max())
    frnn = s2.rmsnorm(np.tensordot((e / e.sum()).astype(np.float32), taps_ctx, axes=(0, 0)),
                      s2.load(W, 'dspark.markov_head.dflare_norm.weight'))

    state = np.zeros((B, R), dtype=np.float32)
    prev = anchor
    drafted = []
    print(f"{'step':>4} {'prev':>7} {'draft':>7} {'oracle':>7} {'score':>10}  ok")
    for i in range(STEPS):
        joint = np.concatenate([state[i], w1[prev], h[i + 1], frnn[-1]], axis=0)[None, :]
        proj = joint @ rp.T + rpb
        g = sig(proj[:, :R]); cand = np.tanh(proj[:, R:2 * R])
        state[i] = cand + g * (state[i] - cand)
        lat = np.tanh(proj[:, 2 * R:3 * R])
        corr = (lat @ ho.T + hob)[0]
        tok, sc = engine_top1(h[i + 1], corr)
        drafted.append(tok)
        ok = 'Y' if tok == oracle[i] else 'n'
        print(f"{i:>4} {prev:>7} {tok:>7} {oracle[i]:>7} {sc:10.3f}  {ok}")
        prev = tok                                     # feedback: the loop consumes its own drafts

    acc = 0
    for x, b in zip(drafted, oracle):
        if x == b:
            acc += 1
        else:
            break
    print(f"\ndrafted   {drafted}")
    print(f"oracle    {oracle}")
    print(f"accepted prefix: {acc}/{STEPS}")


if __name__ == '__main__':
    main()
