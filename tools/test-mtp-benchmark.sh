#!/bin/bash
# test-mtp-benchmark.sh — MTP (Multi-Token Prediction) CPU benchmark
#
# MTP requires loading TWO model instances: main model + MTP head.
# Total memory: ~22 GB for 11B IQ2_M model (7.9GB × 2 + overhead).
# Current hardware: 11 GB RAM — cannot load.
#
# This script checks if MTP can run and documents the requirements.
# It gracefully skips if memory is insufficient.

MODEL="${MODEL:-$HOME/bytropix/models/qwen3.6-iq2m.gguf}"
MTP_BIN="${MTP_BIN:-./gen_text_mtp_cpu}"
AVAIL_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAIL_MEM_GB=$((AVAIL_MEM_KB / 1024 / 1024))
NEED_MEM_GB=22

echo "=== MTP CPU Benchmark ==="
echo "Available RAM:  ${AVAIL_MEM_GB} GB"
echo "Required RAM:   ${NEED_MEM_GB}+ GB (dual model load)"
echo ""

if [ "$AVAIL_MEM_GB" -lt "$NEED_MEM_GB" ]; then
    echo "INSUFFICIENT MEMORY: Need ${NEED_MEM_GB}GB, have ${AVAIL_MEM_GB}GB"
    echo ""
    echo "To run MTP benchmark:"
    echo "  1. Upgrade to machine with 32GB+ RAM"
    echo "  2. Build: make gen_text_mtp_cpu"
    echo "  3. Run: MODEL=$MODEL ./gen_text_mtp_cpu -n 50 \"The future of AI\""
    echo "  4. Compare tok/s vs gen_text_cpu"
    echo ""
    echo "Expected MTP speedup on CPU: ~1.3-1.5x (parallel token prediction)"
    echo "MTP head: +1 extra token predicted per decode step"
    echo ""
    echo "SKIP: insufficient RAM"
    exit 0
fi

# Attempt MTP run if binary exists and memory is sufficient
if [ ! -f "$MTP_BIN" ]; then
    echo "MTP binary not found. Build with: make gen_text_mtp_cpu"
    exit 1
fi

echo "Running MTP benchmark..."
$MTP_BIN -n 50 "The future of AI is" 2>&1 | head -20
echo ""
echo "MTP benchmark complete."
