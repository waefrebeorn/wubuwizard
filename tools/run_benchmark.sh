#!/bin/bash
# run_benchmark.sh — bytropix benchmark automation (Cell 173)
# Usage: ./tools/run_benchmark.sh [model.gguf]
# Env: MODEL, OMP_NUM_THREADS, N_RUNS, N_TOKENS

set -euo pipefail

MODEL="${1:-${MODEL:-$HOME/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
N_RUNS="${N_RUNS:-3}"
cd "$(dirname "$0")/.."

echo "=========================================="
echo " bytropix Benchmark Suite"
echo "=========================================="
echo "Model:     $MODEL"
echo "Threads:   $OMP_NUM_THREADS"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "Commit:    $(git log --oneline -1 2>/dev/null || echo N/A)"

echo -n "Build: "
make -s gen_text_cpu 2>&1 | grep -vE 'warning:|is up to date' || true
echo "OK"

[ -f "$MODEL" ] || { echo "ERROR: Model not found"; exit 1; }
S=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
echo "Size:     $(numfmt --to=iec "$S" 2>/dev/null || echo "$S bytes")"

echo -n "Warmup:  "
OMP_NUM_THREADS=$OMP_NUM_THREADS MODEL="$MODEL" CHAT="" \
    ./gen_text_cpu "Hello" 1 5 &>/dev/null || true
echo "OK"

echo ""
echo "--- Prefill ---"
for spec in "Short:Hello world" "Medium:What is the theory of relativity"; do
    label="${spec%%:*}"
    prompt="${spec#*:}"
    out=$(OMP_NUM_THREADS=$OMP_NUM_THREADS MODEL="$MODEL" ./gen_text_cpu "$prompt" 1 0 2>&1)
    ts=$(echo "$out" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    tc=$(echo "$out" | sed -n 's/Prefill: \([0-9]*\) tok.*/\1/p')
    tm=$(echo "$out" | sed -n 's/.* in \([0-9.]*\)s.*/\1/p')
    printf "  %-8s %4d tok %7.2fs %6.1f tok/s\n" "$label" "${tc:-0}" "${tm:-0}" "${ts:-0}"
done

echo ""
echo "--- Decode (50 tok x 3 runs) ---"
acc=""
for i in 1 2 3; do
    out=$(OMP_NUM_THREADS=$OMP_NUM_THREADS MODEL="$MODEL" \
        ./gen_text_cpu "Hello world" 50 5 2>&1)
    ts=$(echo "$out" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    tc=$(echo "$out" | sed -n 's/Decode: *\([0-9]*\) tok.*/\1/p')
    tm=$(echo "$out" | sed -n 's/.* in \([0-9.]*\)s.*/\1/p')
    printf "  Run %d:   %4d tok %7.2fs %6.1f tok/s\n" "$i" "${tc:-0}" "${tm:-0}" "${ts:-0}"
    acc="$acc ${ts:-0}"
done

echo ""
echo "--- Layer Profile (PROFILE=1) ---"
OMP_NUM_THREADS=$OMP_NUM_THREADS MODEL="$MODEL" PROFILE=1 \
    ./gen_text_cpu "Hello" 1 3 2>&1 | grep -E 'ssm:|moe:|out_proj:|attn:' | head -8

avg=$(echo "$acc" | tr ' ' '\n' | awk '{s+=$1; n++} END {printf "%.1f", n>0 ? s/n : 0}')
echo ""
echo "=========================================="
echo " Decode avg: $avg tok/s (3 runs)"
echo "=========================================="
