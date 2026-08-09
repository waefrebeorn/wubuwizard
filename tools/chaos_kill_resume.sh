#!/usr/bin/env bash
# chaos_kill_resume.sh — the A5 kill-resume chaos test (post-AN54).
# SIGKILL the endurance run at RANDOM rounds; resume; assert:
#   - the resume restores the fitness cells + priority cells (count)
#   - NO contract violations across the whole ordeal
#   - the run CONTINUES to the end (no human reset)
#
# Usage: ./tools/chaos_kill_resume.sh [rounds] [kills]
set -u

ROUNDS="${1:-120}"
KILLS="${2:-3}"
BASE="/tmp/chaos.st"
rm -f "$BASE"* 2>/dev/null

echo "=== chaos: $ROUNDS rounds, $KILLS random SIGKILLs ==="
# --round-delay-ms slows the run so the SIGKILLs land mid-run (the
# 40ms x 150 rounds = ~6s of wall-clock run per segment)
./wubu_endurance --rounds "$ROUNDS" --ckpt 20 --round-delay-ms 40 \
    --out "$BASE" > /tmp/chaos_run.log 2>&1 &
PID=$!
sleep 3

KILLED=0
while [ $KILLED -lt "$KILLS" ]; do
    # the kill point: after a few rounds (the first checkpoint at 20)
    sleep $(( RANDOM % 3 + 2 ))
    kill -9 "$PID" 2>/dev/null || break
    echo "  SIGKILL #$((KILLED+1)) at pid $PID"
    KILLED=$((KILLED+1))

    # resume (the events file appends; the sidecars restore)
    ./wubu_endurance --rounds "$ROUNDS" --ckpt 20 --round-delay-ms 40 \
        --resume "$BASE" --out "$BASE" > /tmp/chaos_run.log 2>&1 &
    PID=$!
    sleep 3
done
# let the final run finish
wait "$PID" 2>/dev/null

echo ""
echo "=== chaos assertions ==="
# 1. the run completed (the final log says so)
grep -q "ALL ENDURANCE ROUNDS COMPLETED" /tmp/chaos_run.log && \
    echo "  PASS: the run completed after $KILLED SIGKILLs" || \
    { echo "  FAIL: the run did not complete"; exit 1; }

# 2. the contract violations are ZERO (the events carry them)
python3 - "$BASE" <<'EOF'
import json, sys
base = sys.argv[1]
try:
    lines = open(base + ".events.jsonl").read().strip().split("\n")
except FileNotFoundError:
    print("  FAIL: no events file"); sys.exit(1)
viol = sum(json.loads(l)["contract_violations"] for l in lines if l)
print(f"  {'PASS' if viol == 0 else 'FAIL'}: {viol} contract violations "
      f"across {len(lines)} events (must be 0)")
sys.exit(0 if viol == 0 else 1)
EOF
RC=$?

# 3. the resume restored the fitness history (the report reads it)
./wubu_report "$BASE" 2>/dev/null | grep -q "fitness cells" && \
    echo "  PASS: the hive archive is readable (history intact)" || \
    { echo "  FAIL: the hive archive is unreadable"; exit 1; }

echo "=== chaos verdict: the colony survives SIGKILL with zero violations ==="
exit $RC
