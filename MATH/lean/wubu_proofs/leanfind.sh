#!/bin/bash
# leanfind.sh — instant mathlib lemma-name lookup. No build cycle.
#
# The other half of the speedup: instead of guessing a lemma name and
# burning a ~4s (or ~293s pre-fastcheck) build cycle to discover it
# doesn't exist, grep the mathlib source directly. This is the
# "grep before guess" rule from the lean4-mathlib-proofs skill,
# mechanized.
#
# Usage:
#   ./leanfind.sh tanh              # all tanh lemmas
#   ./leanfind.sh "sq_lt_sq"        # a specific lemma
#   ./leanfind.sh "inner_smul"      # inner-product smul lemmas
#   ./leanfind.sh "theorem.*iterate"  # regex
#
# Prints: <file>:<line>: <signature>  for every match, so you can see
# the exact statement + which module to import.
set -u
cd "$(dirname "$0")"
ML=".lake/packages/mathlib/Mathlib"
QUERY="${1:?usage: leanfind.sh <lemma-name-or-regex>}"

# grep theorems/lemmas/defs matching the query, show the signature line
grep -rnE "^(theorem|lemma|def|abbrev|instance).*\b${QUERY}\b" "$ML" 2>/dev/null \
  | sed 's#'"$ML"'/##' \
  | head -25

# also catch namespace-qualified ones like Real.tanh_lt_one
grep -rnE "^(theorem|lemma|def).*[.]${QUERY}\b" "$ML" 2>/dev/null \
  | sed 's#'"$ML"'/##' \
  | head -10
