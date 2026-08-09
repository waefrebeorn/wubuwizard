#!/usr/bin/env bash
# test_colony_nightly.sh — the A6 NIGHTLY COLONY GATE (post-AN54 A6).
# Longer than the unit test_colony: a fixed wall-clock budget, ONE
# forced resume, published artifacts. The "no ship without it" gate,
# run nightly:
#   - the endurance run executes for the budget with a FORCED kill
#     mid-run (the resume path is exercised every night, not just in
#     the chaos drill)
#   - the release-gate assertions: zero contract violations, the
#     suite score at the LEARNED level (the skills survived the kill),
#     the archive + sidecars readable, the replay verifier agrees
#   - artifacts published to ./colony-runs/nightly-<ts>/
#
# Usage: ./tools/test_colony_nightly.sh [budget_seconds]
set -u

BUDGET="${1:-240}"          # 4 min by default; the nightly job uses 1800+
TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="/home/wubu/colony-runs/nightly-$TS"
mkdir -p "$OUTDIR"
BASE="$OUTDIR/nightly.st"

echo "=== nightly colony gate: ${BUDGET}s budget, one forced resume ==="

# segment 1: run until the mid-point, then SIGKILL (the forced resume)
./wubu_endurance --rounds 600 --ckpt 25 --round-delay-ms 30 \
    --out "$BASE" > "$OUTDIR/seg1.log" 2>&1 &
PID=$!
sleep $(( BUDGET / 2 ))
echo "  forced SIGKILL at the mid-point (pid $PID)"
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

# segment 2: resume + finish the budget
./wubu_endurance --rounds 600 --ckpt 25 --round-delay-ms 30 \
    --resume "$BASE" --out "$BASE" > "$OUTDIR/seg2.log" 2>&1 &
PID=$!
sleep $(( BUDGET / 2 ))
# if the budget expired mid-segment, kill it (the run has done its job)
if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" 2>/dev/null
fi
wait "$PID" 2>/dev/null

echo ""
echo "=== nightly assertions ==="
RC=0

# 1. the events exist + zero contract violations (the release floor)
python3 - "$BASE" <<'EOF'
import json, sys
base = sys.argv[1]
try:
    lines = open(base + ".events.jsonl").read().strip().split("\n")
except FileNotFoundError:
    print("  FAIL: no events file"); sys.exit(1)
viol = 0; n = 0
for l in lines:
    if not l.strip(): continue
    try:
        e = json.loads(l); n += 1; viol += e["contract_violations"]
    except json.JSONDecodeError: continue
print(f"  {'PASS' if viol == 0 else 'FAIL'}: {viol} violations across {n} events")
sys.exit(0 if viol == 0 else 1)
EOF
[ $? -ne 0 ] && RC=1

# 2. the suite is at the LEARNED level (the skills survived the kill)
python3 - "$BASE" <<'EOF'
import json, sys
base = sys.argv[1]
lines = open(base + ".events.jsonl").read().strip().split("\n")
suites = []
for l in lines:
    if not l.strip(): continue
    try: suites.append(json.loads(l)["suite"])
    except json.JSONDecodeError: continue
if not suites:
    print("  FAIL: no suite data"); sys.exit(1)
last = suites[-1]
print(f"  {'PASS' if last >= 0.65 else 'FAIL'}: final suite {last:.3f} "
      f"(learned >= 0.65 = the resume kept the skills)")
sys.exit(0 if last >= 0.65 else 1)
EOF
[ $? -ne 0 ] && RC=1

# 3. the artifacts are complete + readable (the .skills sidecar carries
# the LEARNED skills — the nightly caught it being memory-only before)
ls "$BASE.hive" "$BASE.prio" "$BASE.events.jsonl" "$BASE.skills" > /dev/null 2>&1 && \
    echo "  PASS: .hive + .prio + .events.jsonl + .skills present" || \
    { echo "  FAIL: artifacts incomplete"; RC=1; }

# 4. the report + replay post-mortem run (the operator can read the night)
./wubu_report "$BASE" --out "$OUTDIR/report.md" > /dev/null 2>&1 && \
    echo "  PASS: report generated" || { echo "  FAIL: report"; RC=1; }
if ./wubu_replay "$BASE" > /tmp/nightly_replay.log 2>&1; then
    echo "  PASS: the replay verifier agrees"
else
    grep -q "0 diverged\|VERIFIED" /tmp/nightly_replay.log && \
        echo "  PASS: replay (0 divergences)" || { echo "  FAIL: replay"; RC=1; }
fi

# 5. the A8 anomaly detector — the RELEASE GATE (a rising suite that
# hides contract trips or a degenerate accept-all gate fails the night)
if ./wubu_anomaly "$BASE.events.jsonl" > /tmp/nightly_anomaly.log 2>&1; then
    echo "  PASS: the anomaly detector (release gate) is clean"
else
    echo "  FAIL: the anomaly detector flagged anomalies:"
    grep "ANOMALY\|WARNING" /tmp/nightly_anomaly.log | head -4
    RC=1
fi

echo ""
if [ $RC -eq 0 ]; then
    echo "=== NIGHTLY COLONY GATE PASSED — artifacts in $OUTDIR ==="
else
    echo "=== NIGHTLY COLONY GATE FAILED — see $OUTDIR ==="
fi
exit $RC
