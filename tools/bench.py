#!/usr/bin/env python3
"""Median-of-N timing for the inkling2 generator.

Why this exists: the model file is 8.4 GB and memory-mapped, so the first run after a rebuild faults pages in
from disk and every later run reads from the page cache. Three consecutive runs of the same binary on identical
work were measured at 2.0 / 1.5 / 1.2 s per step - a 1.7x spread that is pure cache state, not code. Any
optimisation claim smaller than that spread is indistinguishable from warming.

So: warm once, then take the median of N runs and report the min/max spread alongside it. If the delta you are
testing is inside the spread, the harness says so instead of letting you believe it.

Usage:
    python tools/bench.py [-n 5] [-g 3] [-p PREFIX] [--env KEY=VAL ...] -- <tokens...>

Example:
    python tools/bench.py -n 5 -g 3 371 151451 149879
    python tools/bench.py -n 5 -g 3 --env IN2_KV=1 371 151451 149879
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen_fast.exe")
STEP = re.compile(r"^gen\[(\d+)\].*?([0-9.]+)s")


def run(env_extra, n_gen, tokens):
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run([EXE, str(n_gen), *tokens], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit("run failed rc=%d: %s" % (p.returncode, p.stderr[-400:]))
    out = {}
    for line in p.stdout.splitlines():
        m = STEP.match(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="timed runs (default 5)")
    ap.add_argument("-g", type=int, default=3, help="tokens to generate (default 3)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL for the child")
    ap.add_argument("tokens", nargs="+")
    a = ap.parse_args()

    env_extra = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v

    print("warming (one full run, pages faulted in from disk)...")
    run(env_extra, a.g, a.tokens)

    samples = {i: [] for i in range(a.g)}
    for i in range(a.n):
        r = run(env_extra, a.g, a.tokens)
        for k, v in r.items():
            samples[k].append(v)
        print("  run %d: %s" % (i + 1, " ".join("%.2fs" % r.get(k, float("nan")) for k in range(a.g))))

    print("\n%-8s %8s %8s %8s %8s" % ("step", "median", "min", "max", "spread"))
    for k in range(a.g):
        s = samples[k]
        if not s:
            continue
        med, lo, hi = statistics.median(s), min(s), max(s)
        print("%-8d %7.2fs %7.2fs %7.2fs  %5.0f%%" % (k, med, lo, hi, 100.0 * (hi - lo) / med if med else 0))

    print("\nOnly trust a delta larger than the spread above. Steady state is what matters here;")
    print("step 0 includes the prefill and is not comparable to the later steps.")


if __name__ == "__main__":
    main()
