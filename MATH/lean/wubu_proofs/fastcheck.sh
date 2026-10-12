#!/bin/bash
# fastcheck.sh — check a single Lean file, OLEAN-CACHED + PARALLEL.
#
# MEASURED (2026-10-12, this repo, mathlib cache warm):
#   old `lake env lean <file>` : ~293s per check (re-elaborates the WHOLE
#                                  Mathlib graph single-threaded every time)
#   `lake build <module>`      : ~13s  per check (reuses cached oleans,
#                                  parallel across `nproc` cores)
#   => 22x faster iteration loop.
#
# Usage: ./fastcheck.sh WubuProofs/MobiusAddND.lean
#        ./fastcheck.sh WubuProofs.lean        (whole library)
#
# Converts the .lean path to the module name (WubuProofs/MobiusAddND.lean
# -> WubuProofs.MobiusAddND) and runs `lake build` on just that target,
# which reuses cached oleans for everything else and only recompiles the
# changed file + its dependents.
set -u
cd "$(dirname "$0")"

FILE="${1:?usage: fastcheck.sh <path-to-lean-file>}"
# strip .lean, convert / to .  => module name
MOD="${FILE%.lean}"
MOD="${MOD//\//.}"

# `lake build <module>` reuses cached oleans and parallelizes (reads jobs
# from lake config). Measured ~9-13s vs ~293s for `lake env lean`. No
# -j/--jobs flag — this lake version rejects both.
timeout 300 lake build "$MOD" 2>&1 \
  | grep -vE "^trace|^info:|^Building|^✔|^\[|Replayed|ring_nf|^[[:space:]]*(The .ring.|Note that|works primarily|consider using|rejects|/[[:space:]]*$|\[apply\]|\[focus\]|case|⊢|goals?)|Welcome to Lean|Documentation|Management|Support|System|Usage of|Memory|Swap|Canonical|https://ubuntu|^[[:space:]]*\*|^$"
echo "FASTCHECK_EXIT:${PIPESTATUS[0]}"
