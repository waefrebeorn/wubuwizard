#!/usr/bin/env python3
"""Test persistent gen_text_cpu with multi-turn conversation."""
import subprocess
import struct
import sys
import time
import os

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf"))
BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen_text_cpu")

def send_prompt(proc, text, max_tokens=10, top_k=1):
    """Send a prompt via binary protocol, receive result."""
    data = text.encode("utf-8")
    # Send: <4-byte len> <text> <4-byte max_tokens> <4-byte top_k>
    proc.stdin.write(struct.pack("<I", len(data)))
    proc.stdin.write(data)
    proc.stdin.write(struct.pack("<II", max_tokens, top_k))
    proc.stdin.flush()
    
    # Receive: <4-byte len> <result> <4-byte tokens_generated>
    len_bytes = proc.stdout.read(4)
    if not len_bytes or len(len_bytes) < 4:
        return "", 0
    result_len = struct.unpack("<I", len_bytes)[0]
    result = proc.stdout.read(result_len).decode("utf-8", errors="replace")
    gen_bytes = proc.stdout.read(4)
    gen = struct.unpack("<I", gen_bytes)[0] if len(gen_bytes) == 4 else 0
    return result, gen

def reset(proc):
    """Send reset signal to clear KV cache."""
    proc.stdin.write(struct.pack("<I", 0))
    proc.stdin.flush()
    time.sleep(0.5)

# Start persistent process
print(f"Starting {BINARY} --persist...")
sys.stdout.flush()

proc = subprocess.Popen(
    [BINARY, "--persist"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ, "MODEL": MODEL, "OMP_NUM_THREADS": "4"},
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# Wait for model to load by polling for "[persist] ready" on stderr
import select
print("Waiting for model load...")
sys.stdout.flush()
ready_line = b""
while b"[persist] ready" not in ready_line:
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        print(f"Process died before ready! stderr: {stderr[:1000]}")
        sys.exit(1)
    # Read stderr non-blocking
    r, _, _ = select.select([proc.stderr], [], [], 1.0)
    if r:
        chunk = proc.stderr.read1(4096)
        ready_line += chunk
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()

print("Model loaded!")
sys.stdout.flush()

# Turn 1: simple prompt
t0 = time.time()
result1, gen1 = send_prompt(proc, "hello", 10, 1)
t1 = time.time()
print(f"Turn 1: {result1!r} ({gen1} tokens in {t1-t0:.1f}s)")
sys.stdout.flush()

if gen1 == 0:
    # Process likely died — get stderr
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    print(f"Process stderr (checking for errors):")
    for line in stderr.split("\n"):
        print(f"  {line}")
    sys.exit(1)

# Turn 2: follow-up (appends to KV cache)
t0 = time.time()
result2, gen2 = send_prompt(proc, "tell me more", 10, 1)
t1 = time.time()
print(f"Turn 2: {result2!r} ({gen2} tokens in {t1-t0:.1f}s)")
sys.stdout.flush()

# Turn 3: another follow-up
t0 = time.time()
result3, gen3 = send_prompt(proc, "what else?", 10, 1)
t1 = time.time()
print(f"Turn 3: {result3!r} ({gen3} tokens in {t1-t0:.1f}s)")
sys.stdout.flush()

# Get stderr for diagnostics
stderr = proc.stderr.read().decode() if proc.stderr else ""
# Filter persist messages
for line in stderr.split("\n"):
    if "[persist]" in line:
        print(f"  {line.strip()}")

proc.stdin.close()
proc.wait(timeout=5)

print("\nDone.")
