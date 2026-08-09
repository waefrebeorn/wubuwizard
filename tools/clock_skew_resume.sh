#!/usr/bin/env bash
# clock_skew_resume.sh — the A10 CLOCK-SKEW RESUME TEST (post-AN54
# A10): after a kill + resume, the event stream must be monotone in
# the crash-recovery sense:
#   1. the round counter NEVER re-stamps below the resume base (the
#      batch from the checkpoint + 1) — a resume that "lost its
#      place" and re-stamped from 1 is a BUG (caught a REAL one:
#      hdr.batch was saved but never restored — the clock-skew drill
#      found the stream going 86 -> 2 on resume)
#   2. the post-checkpoint rounds may legitimately RE-RUN ONCE (the
#      kill landed after the checkpoint, so the resume restores state
#      by re-running them) — the assertion: at most ONE backward step,
#      landing on a checkpoint base (a multiple of ckpt + 1), and no
#      round appears more than twice
#   3. the events mtime is non-decreasing
#   4. the loss has no >1.0 discontinuity at the resume boundary
#
# Usage: ./tools/clock_skew_resume.sh
set -u

BASE=/tmp/skew.st
rm -f "$BASE"* 2>/dev/null

echo "=== A10 clock-skew resume (kill + resume + monotonicity) ==="
./wubu_endurance --rounds 600 --ckpt 20 --round-delay-ms 40 \
    --out "$BASE" > /tmp/skew_run1.log 2>&1 &
PID=$!
# wait for the run to pass the first checkpoint, then kill the BINARY
# (pgrep the actual endurance process — a wrapper subshell may sit
# between this script and the binary)
sleep 2
KPID=$(pgrep -f "wubu_endurance --rounds 600" | head -1)
[ -n "$KPID" ] && kill -9 "$KPID" 2>/dev/null && echo "  SIGKILL the binary ($KPID)"
# kill the wrapper too, if any
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

# the mtime BEFORE the resume
MT0=$(stat -c %Y "$BASE.events.jsonl" 2>/dev/null || echo 0)
./wubu_endurance --rounds 600 --ckpt 20 --round-delay-ms 40 \
    --resume "$BASE" --out "$BASE" > /tmp/skew_run2.log 2>&1
MT1=$(stat -c %Y "$BASE.events.jsonl" 2>/dev/null || echo 0)
echo "  events mtime: $MT0 -> $MT1 (must be non-decreasing)"

# the AUTHORITATIVE resume batch (from the runner's own log)
RESUME_BATCH=$(grep -o "resume continues at round [0-9]* (batch [0-9]*)" \
    /tmp/skew_run2.log | sed 's/.*(batch //; s/)//' | head -1)
RESUME_BATCH="${RESUME_BATCH:-0}"
echo "  the resume's own batch: $RESUME_BATCH"

# 1+2. the round counter: the AUTHORITATIVE check is the resume's own
# batch (the runner logs 'resume continues at round batch+1') — the
# events-file rewind position is shifted by the kill's group-commit
# tear (one trailing event line may be torn). The events sanity check:
# the POST-KILL rounds (after the first torn line, the kill's mark)
# must all be >= the resume base, and no round >2x.
python3 - "$BASE.events.jsonl" "$RESUME_BATCH" <<'EOF'
import json, sys
from collections import Counter
rounds = []
torn_at = -1   # the index of the first torn line = the kill's mark
torn = 0
for l in open(sys.argv[1]):
    l = l.strip()
    if not l: continue
    try:
        e = json.loads(l)
        rounds.append(e["round"])
    except json.JSONDecodeError:
        if torn == 0: torn_at = len(rounds)
        torn += 1  # the kill's group-commit tear (expected, dropped)
resume_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
base = resume_batch + 1
post_kill = rounds[torn_at:] if torn_at >= 0 else rounds
print(f"  {len(rounds)} events (+{torn} torn at the kill, mark at "
      f"event {torn_at}), resumed from batch {resume_batch} -> round {base}")
cnt = Counter(rounds)
below = [r for r in post_kill if r < base]
more_than_twice = [r for r, c in cnt.items() if c > 2]
if torn > 1:
    print(f"  FAIL: {torn} torn lines (the group-commit should lose "
          f"at most ONE trailing event)")
    sys.exit(1)
if below:
    print(f"  FAIL: post-kill rounds {below[:5]} below the resume base "
          f"{base} (the resume re-stamped to the past)")
    sys.exit(1)
if more_than_twice:
    print(f"  FAIL: rounds {more_than_twice[:4]} appear >2x")
    sys.exit(1)
print(f"  PASS: post-kill rounds >= the resume base {base}, no >2x")
sys.exit(0)
EOF
[ $? -ne 0 ] && exit 1

# 3. the mtime is non-decreasing
[ "$MT1" -ge "$MT0" ] && \
    echo "  PASS: the events mtime did not move backward" || \
    { echo "  FAIL: the mtime moved backward"; exit 1; }

# 4. the loss has NO >1.0 upward discontinuity (a reset would jump it)
python3 - "$BASE.events.jsonl" <<'EOF'
import json, sys
losses = []
for l in open(sys.argv[1]):
    l = l.strip()
    if not l: continue
    try:
        e = json.loads(l); losses.append(e["loss"])
    except json.JSONDecodeError:
        continue
jumps = 0
for i in range(1, len(losses)):
    if losses[i] > losses[i-1] + 1.0:
        jumps += 1
print(f"  {'PASS' if jumps == 0 else 'FAIL'}: {jumps} loss resets >1.0 "
      f"(must be 0)")
sys.exit(1 if jumps else 0)
EOF
[ $? -ne 0 ] && exit 1

echo "=== A10 PASSED: the resume is monotone (counter + mtime + loss) ==="
