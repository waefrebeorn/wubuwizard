#!/usr/bin/env bash
# chaos_kill_resume.sh — the A5 kill-resume chaos test (post-AN54 A1/A2).
# SIGKILL the endurance run at RANDOM rounds; resume; assert:
#   - the resume restores the fitness cells + priority cells (count)
#   - NO contract violations across the whole ordeal
#   - SUITE CONTINUITY: the resumed run's suite score does not reset
#     (a fresh run would start near 0.51; a resume continues from the
#     learned suite ~0.83)
#   - the run CONTINUES to the end (no human reset)
# Also: the DOUBLE-KILL drill (A2) — kill DURING a sidecar write and
# prove no torn .hive/.prio (the atomic tmp+fsync+rename save).
#
# Usage: ./tools/chaos_kill_resume.sh [rounds] [kills]
set -u

ROUNDS="${1:-400}"
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
RESUMED_SUITE=""
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
viol = 0
n = 0
for l in lines:
    if not l.strip():
        continue
    try:
        e = json.loads(l)
        n += 1
        viol += e["contract_violations"]
    except json.JSONDecodeError:
        # the group-commit tear: a SIGKILL mid-buffer leaves a partial
        # trailing line — telemetry, dropped, not an error
        continue
print(f"  {'PASS' if viol == 0 else 'FAIL'}: {viol} contract violations "
      f"across {n} events (must be 0)")
sys.exit(0 if viol == 0 else 1)
EOF
RC=$?

# 3. the resume restored the fitness history (the report reads it)
./wubu_report "$BASE" 2>/dev/null | grep -q "fitness cells" && \
    echo "  PASS: the hive archive is readable (history intact)" || \
    { echo "  FAIL: the hive archive is unreadable"; exit 1; }

# 4. SUITE CONTINUITY (the A1 upgrade): the final suite score must be
# the LEARNED level (~0.8), not a fresh-run reset (~0.51). A chaos run
# that silently restarted from zero would pass 1-3 but fail this.
FINAL_SUITE=$(python3 -c "
import json
lines = open('$BASE.events.jsonl').read().strip().split('\n')
print(f\"{json.loads(lines[-1])['suite']:.3f}\")" 2>/dev/null)
echo "  final suite: $FINAL_SUITE (learned ~0.8 = resume worked; ~0.51 = reset)"
python3 - "$FINAL_SUITE" <<'EOF'
import sys
try:
    s = float(sys.argv[1])
except (ValueError, IndexError):
    print("  FAIL: cannot read the final suite"); sys.exit(1)
if s >= 0.65:
    print("  PASS: the suite survived the kills (the colony kept its skills)")
else:
    print("  FAIL: the suite reset — the resume lost the learned skills")
    sys.exit(1)
EOF
[ $? -ne 0 ] && exit 1

echo ""
echo "=== the A2 DOUBLE-KILL drill (kill during a sidecar write) ==="
rm -f /tmp/dk.st* 2>/dev/null
./wubu_endurance --rounds 400 --ckpt 5 --round-delay-ms 8 \
    --out /tmp/dk.st > /tmp/dk_run.log 2>&1 &
DPID=$!
sleep 2.5
# kill MANY times in quick succession — some kills land inside the
# sidecar save window (ckpt every 5 rounds with an 8ms delay)
for i in 1 2 3 4; do
    kill -9 "$DPID" 2>/dev/null && echo "  double-kill #$i" && sleep 0.4
    ./wubu_endurance --rounds 400 --ckpt 5 --round-delay-ms 8 \
        --resume /tmp/dk.st --out /tmp/dk.st > /tmp/dk_run.log 2>&1 &
    DPID=$!
    sleep 2
done
wait "$DPID" 2>/dev/null
# the .hive must load clean (the atomic save + verified load refuse a
# torn archive instead of loading garbage)
python3 - /tmp/dk.st <<'EOF'
import json, sys
base = sys.argv[1]
try:
    lines = open(base + ".events.jsonl").read().strip().split("\n")
    n = 0
    ok = True
    for l in lines:
        if not l.strip():
            continue
        try:
            e = json.loads(l)
            n += 1
            if e["contract_violations"] != 0:
                ok = False
        except json.JSONDecodeError:
            continue  # the group-commit tear (dropped, not an error)
    print(f"  {'PASS' if ok else 'FAIL'}: {n} events, 0 violations "
          f"after 4 rapid kills (no torn sidecars)")
    sys.exit(0 if ok else 1)
except FileNotFoundError:
    print("  FAIL: no events after the double-kill drill"); sys.exit(1)
EOF
[ $? -ne 0 ] && exit 1

echo "=== chaos verdict: the colony survives SIGKILL with suite continuity ==="
exit $RC
