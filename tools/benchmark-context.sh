#!/bin/bash
# benchmark-context.sh — Measure decode tok/s at different context lengths
#
# Usage:
#   MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf bash tools/benchmark-context.sh
#
# Tests dense and sparse (USE_SPARSE_ATTN=1, SPARSE_MIN=512) at each length.
# Outputs a table with tok/s and ms/tok.
#
# Environment:
#   MODEL       — GGUF model path
#   OMP_THREADS — OpenMP threads (default: 4)
#
# NOTE: Each test spawns a new gen_text_cpu process (model reload ~80s).
# Total runtime: ~13 minutes for 4 lengths × 2 modes. Run overnight.

set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf}"
OMP_THREADS="${OMP_THREADS:-4}"
GEN_TOKENS=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BYTROPIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="${BYTROPIX_DIR}/gen_text_cpu"

CONTEXT_LENGTHS=(50 100 200 500)

echo "=== Context Growth Benchmark ==="
echo "Model:  $(basename "$MODEL")"
echo "OMP:    $OMP_THREADS"
echo "Binary: $BINARY"
echo ""

[ -x "$BINARY" ] || { echo "ERROR: $BINARY not found or not executable"; exit 1; }

bench_one() {
    local N="$1"
    local sparse="$2"  # "" or "USE_SPARSE_ATTN=1 SPARSE_MIN=512"
    
    # Build N-token prompt
    local prompt
    prompt="$(python3 -c "import sys; sys.stdout.write(' '.join(['hello'] * $N))")"
    
    local extra_env=""
    local label="dense"
    if [ -n "$sparse" ]; then
        extra_env="$sparse"
        label="sparse"
    fi
    
    local out
    out="$(MODEL="$MODEL" OMP_NUM_THREADS="$OMP_THREADS" $extra_env \
           timeout 600 "$BINARY" "$prompt" $GEN_TOKENS 1 2>/dev/null)" || {
        echo "$label: ERR"
        return
    }
    
    local rate
    rate="$(echo "$out" | grep "^Decode:" | awk '{print $6}' | tr -d '()')"
    
    if [ -n "$rate" ]; then
        echo "$label: $rate"
    else
        echo "$label: ERR"
    fi
}

# Warmup: run once to load model, discard results
echo "Warming up (loading model)..."
MODEL="$MODEL" OMP_NUM_THREADS="$OMP_THREADS" timeout 600 "$BINARY" "hello" 1 1 >/dev/null 2>&1 || true
echo ""

printf "%-8s | %-16s | %-16s | %-16s | %-16s\n" "KV" "Dense(tok/s)" "Sparse(tok/s)" "Dense(ms/tok)" "Sparse(ms/tok)"
printf "%-8s-+-%-16s-+-%-16s-+-%-16s-+-%-16s\n" "--------" "----------------" "----------------" "----------------" "----------------"

for N in "${CONTEXT_LENGTHS[@]}"; do
    d_result=$(bench_one "$N" "")
    s_result=$(bench_one "$N" "USE_SPARSE_ATTN=1 SPARSE_MIN=512")
    
    d_rate=$(echo "$d_result" | grep "^dense:" | cut -d' ' -f2 || echo "ERR")
    s_rate=$(echo "$s_result" | grep "^sparse:" | cut -d' ' -f2 || echo "ERR")
    
    d_ms="?"
    s_ms="?"
    if [ "$d_rate" != "ERR" ] && [ "$d_rate" != "TIMEOUT" ]; then
        d_ms=$(python3 -c "print(f'{1000/$d_rate:.1f}')" 2>/dev/null || echo "?")
    fi
    if [ "$s_rate" != "ERR" ] && [ "$s_rate" != "TIMEOUT" ]; then
        s_ms=$(python3 -c "print(f'{1000/$s_rate:.1f}')" 2>/dev/null || echo "?")
    fi
    
    printf "%-8s | %-16s | %-16s | %-16s | %-16s\n" "$N" "$d_rate" "$s_rate" "$d_ms" "$s_ms"
done

echo ""
echo "Done."
