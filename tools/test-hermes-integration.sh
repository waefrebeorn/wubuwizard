#!/usr/bin/env bash
# test-hermes-integration.sh — Verify bytropix works as a Hermes provider
# Tests: server up → model list → chat completion → Hermes config → cleanup

set -euo pipefail

PORT="${1:-8003}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BYTROPIX_DIR="$(dirname "$SCRIPT_DIR")"
MODEL="${MODEL:-$HOME/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf}"
BIN="${BYTROPIX_DIR}/gen_text_cpu"

PASS=0
FAIL=0

pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== bytropix → Hermes Integration Test ===\n"

# 1. Check binary exists
if [ -f "$BIN" ]; then
    pass "Inference binary exists: $BIN"
else
    fail "Binary not found: $BIN — run 'make gen_text_cpu' first"
    echo "\nResults: $PASS pass, $FAIL fail"
    exit 1
fi

# 2. Check model exists
if [ -f "$MODEL" ]; then
    pass "Model exists: $MODEL"
else
    fail "Model not found: $MODEL"
    echo "\nResults: $PASS pass, $FAIL fail"
    exit 1
fi

# 3. Start server
echo "\n--- Starting server on port $PORT ---"
cd "$BYTROPIX_DIR"
MODEL="$MODEL" python3 tools/serve_local.py --port "$PORT" &
SERVER_PID=$!
sleep 2

if kill -0 "$SERVER_PID" 2>/dev/null; then
    pass "Server running (PID: $SERVER_PID)"
else
    fail "Server failed to start"
    echo "\nResults: $PASS pass, $FAIL fail"
    exit 1
fi

# 4. Health endpoint
HEALTH=$(curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "")
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    pass "/health — status=ok"
else
    fail "/health failed"
fi

# 5. Models endpoint
MODELS=$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo "")
if echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['data'])>0" 2>/dev/null; then
    pass "/v1/models — returns model list"
else
    fail "/v1/models failed"
fi

# 6. Chat completion (basic, shorter timeout)
CHAT=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}' 2>/dev/null || echo "")
if echo "$CHAT" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
    pass "/v1/chat/completions — returns choices"
    echo "  Output: $(echo "$CHAT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:60])" 2>/dev/null)"
else
    fail "/v1/chat/completions failed"
fi

# 7. Completions endpoint
COMP=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","max_tokens":8}' 2>/dev/null || echo "")
if echo "$COMP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
    pass "/v1/completions — returns choices"
else
    fail "/v1/completions failed"
fi

# 8. OpenAI API format check (Hermes-compatible)
# Hermes expects: /v1/chat/completions with "choices[0].message.content"
if echo "$CHAT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
assert d['object']=='chat.completion'
assert 'content' in d['choices'][0]['message']
assert d['choices'][0]['finish_reason']=='stop'
" 2>/dev/null; then
    pass "OpenAI-format response (Hermes-compatible)"
else
    fail "Response format not Hermes-compatible"
fi

# 9. 404 handling
ERR=$(curl -sf "http://127.0.0.1:$PORT/nonexistent" 2>/dev/null || echo "404")
if echo "$ERR" | python3 -c "import sys; assert '404' in str(sys.stdin.read())" 2>/dev/null; then
    pass "404 handling works"
else
    fail "404 handling failed"
fi

echo "\n=== Results: $PASS pass, $FAIL fail ==="

# Kill server
cleanup

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
