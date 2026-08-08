#!/usr/bin/env bash
# start-bytropix-server.sh — Start the bytropix local inference server
# Usage: ./tools/start-bytropix-server.sh [port]
# Default port: 8001

set -euo pipefail

PORT="${1:-8001}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BYTROPIX_DIR="$(dirname "$SCRIPT_DIR")"

# Default model path
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf}"

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "Set MODEL env var or create symlink at $MODEL"
    exit 1
fi

# Check binary exists
BIN="${BYTROPIX_DIR}/gen_text_cpu"
if [ ! -f "$BIN" ]; then
    echo "[BUILD] gen_text_cpu not found — building..."
    cd "$BYTROPIX_DIR" && make gen_text_cpu -j$(nproc) 2>&1 | tail -3
fi

# Start server
echo "[START] bytropix inference server on port $PORT"
echo "[MODEL] $MODEL"
echo "[BIN]   $BIN"
echo "[THREADS] ${OMP_NUM_THREADS:-4}"
echo ""

cd "$BYTROPIX_DIR"
exec python3 tools/serve_local.py --port "$PORT" --model "$MODEL" --bin "$BIN"
