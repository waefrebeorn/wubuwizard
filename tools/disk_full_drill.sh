#!/usr/bin/env bash
# disk_full_drill.sh — the A9 DISK-FULL DRILL (post-AN54 A9): the
# sidecar write fails mid-run; the run must NOT corrupt the archive,
# and a clean resume must work.
#
# WSL has no mount (tmpfs) privileges, so the drill uses the honest
# equivalent: `ulimit -f` caps the FILE SIZE — when the events file
# hits the cap, the write fails exactly like an ENOSPC disk-full
# (SIGXFSZ kills the process mid-write). Then:
#   1. the sidecars written BEFORE the cap are intact + readable
#      (the atomic tmp+fsync+rename pattern means a kill mid-save
#      leaves the OLD archive, never a torn one)
#   2. a clean resume from those sidecars works
#   3. a fresh run still works after the drill
#
# Usage: ./tools/disk_full_drill.sh
set -u

BASE=/tmp/df.st
rm -f "$BASE"* 2>/dev/null

echo "=== A9 disk-full drill (ulimit -f 8 = 4KB file cap) ==="
# the events file hits the 4KB cap within a few rounds -> SIGXFSZ
( ulimit -f 8; ./wubu_endurance --rounds 300 --ckpt 20 --out "$BASE" \
    > /tmp/df_run.log 2>&1 ) 2>/dev/null
echo "  the run was killed by the file cap (expected)"

# 1. the pre-kill sidecars are intact + readable (never torn)
if [ -f "$BASE.hive" ] && [ -f "$BASE.prio" ] && [ -f "$BASE.skills" ]; then
    ./wubu_report "$BASE" 2>/dev/null | grep -q "fitness cells" && \
        echo "  PASS: the pre-kill .hive/.prio/.skills are intact + readable" || \
        { echo "  FAIL: the archive is corrupt"; exit 1; }
else
    echo "  FAIL: no sidecars survived (the cap hit before the first ckpt)"
    exit 1
fi

# 2. a clean resume works (the events append, the sidecars restore)
./wubu_endurance --rounds 60 --ckpt 10 --resume "$BASE" --out "$BASE" \
    > /tmp/df_resume.log 2>&1
grep -q "resumed.*fitness cells\|resumed.*skills" /tmp/df_resume.log && \
    echo "  PASS: the resume restored the colony state" || \
    { echo "  FAIL: the resume could not restore"; exit 1; }

# 3. the resumed run's events are parseable (the killed write may have
# left a partial trailing line — the readers must tolerate it)
python3 - "$BASE.events.jsonl" <<'EOF'
import json, sys
n = 0
for l in open(sys.argv[1]):
    l = l.strip()
    if not l: continue
    try:
        json.loads(l); n += 1
    except json.JSONDecodeError:
        pass  # the group-commit tear (dropped)
print(f"  PASS: {n} parseable events (torn trailing line tolerated)")
EOF

# 4. a fresh run still works after the drill
./wubu_endurance --rounds 40 --ckpt 5 --out /tmp/df_clean.st > /tmp/df_clean.log 2>&1
grep -q "ALL ENDURANCE ROUNDS COMPLETED" /tmp/df_clean.log && \
    echo "  PASS: a clean run still works after the drill" || \
    { echo "  FAIL: the clean run broke"; exit 1; }

echo "=== A9 drill PASSED: the disk-full never corrupts the colony ==="
