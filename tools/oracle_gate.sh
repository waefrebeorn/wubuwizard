#!/usr/bin/env bash
# oracle_gate.sh - the parity gate this project keeps needing, as one command.
#
# Every change in this repo is gated on the 12-token oracle being token-for-token identical. Doing that by hand has
# repeatedly gone wrong in this session in the same way: a failed build leaves gen_fast.exe missing, both runs return an
# EMPTY ids line, and an equality test compares "" to "" and reports success. This script cannot make that mistake:
# it requires the reference to be non-empty, requires the expected token count, and compares the whole line.
#
#   usage: bash tools/oracle_gate.sh [--rebuild]
#   exit:  0 = identical to the golden oracle, 1 = mismatch, 2 = harness problem (missing/empty/too few tokens)
#
# Golden oracle (12 tokens from prompt 371 151451 149879):
#   371 151451 149879 149876 220 36838 182189 149933 220 50843 11 149984 149933 151534 149919
set -u
cd "$(dirname "$0")/.." || exit 2

GOLDEN="IDS: 371 151451 149879 149876 220 36838 182189 149933 220 50843 11 149984 149933 151534 149919"
WANT_TOKENS=12   # n_gen; the IDS line holds 3 prompt + 12 generated = 15 tokens

if [ "${1:-}" = "--rebuild" ] || [ ! -x ./gen_fast.exe ]; then
    rm -f gen_fast.exe
    if ! bash tools/build_gen_fast.sh > /tmp/oracle_build.log 2>&1; then
        echo "GATE FAIL(2): build failed - see /tmp/oracle_build.log"
        grep -E " error" /tmp/oracle_build.log | head -5
        exit 2
    fi
fi

ids=$(./gen_fast.exe $WANT_TOKENS 371 151451 149879 2>/dev/null | grep '^IDS:')
rc=$?

# harness checks first: an empty or short line is NOT a pass, it is a broken run.
if [ -z "$ids" ]; then
    echo "GATE FAIL(2): oracle produced no IDS line (build broken, binary missing, or crash)"
    exit 2
fi
n=$(printf '%s' "$ids" | wc -w)
# IDS: + 12 tokens = 13 words
if [ "$n" -ne 16 ]; then
    echo "GATE FAIL(2): expected 16 words (IDS: + 3 prompt + 12 generated), got $n -> '$ids'"
    exit 2
fi

if [ "$ids" = "$GOLDEN" ]; then
    echo "GATE PASS: 15/15 oracle tokens identical (3 prompt + 12 generated)"
    exit 0
fi

echo "GATE FAIL(1): oracle mismatch"
echo "  want: $GOLDEN"
echo "  got : $ids"
exit 1
