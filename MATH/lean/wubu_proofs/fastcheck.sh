#!/bin/bash
# fastcheck.sh — check a single Lean file without lake's full job graph.
# Usage: ./fastcheck.sh WubuProofs/MobiusAddND.lean
# ~4-6 min per run instead of ~7 min via `lake build <target>`, and no
# trace/job noise — just errors, or silence on success.
set -u
cd "$(dirname "$0")"
FILE="${1:?usage: fastcheck.sh <path-to-lean-file>}"
timeout 540 lake env lean "$FILE" 2>&1 | grep -vE "^trace|^\[|Welcome to|Documentation|Management|Support|System|Usage of|Memory|Swap|Canonical|https://ubuntu|^\s*\*|^$"
echo "FASTCHECK_EXIT:$?"
