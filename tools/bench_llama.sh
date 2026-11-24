#!/usr/bin/env bash
# tools/bench_llama.sh -- reproducible llama-bench workflow for wubuwizard.
# Usage:
#   bash tools/bench_llama.sh <model.gguf> [tag] [llama-bench.exe]
# Env:
#   REPS=5                 repetitions per cell (default 5)
#   DEVICE=Vulkan1         explicit device
#   THREADS=4              CPU threads
#   PROFILE_KEYS=chat,coding,agent,saturation
#   VARIANTS=baseline,loopopt
#   OUTROOT=<repo>/data/bench
# Output: a timestamped manifest.json, raw/*.json+stderr, results.csv, summary.md.
set -euo pipefail

MODEL="${1:?model gguf path required}"
TAG="${2:-run}"
BIN="${3:-$HOME/models/llama/unsloth-b12651-vulkan/llama-bench.exe}"
REPS="${REPS:-5}"
DEVICE="${DEVICE:-Vulkan1}"
THREADS="${THREADS:-4}"
PROFILE_KEYS="${PROFILE_KEYS:-chat,coding,agent,saturation}"
VARIANTS="${VARIANTS:-baseline,loopopt}"
PY="${PY:-C:/Users/eman5/AppData/Local/Programs/Python/Python311/python.exe}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTROOT="${OUTROOT:-$REPO/data/bench}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUTROOT/${STAMP}-${TAG}"
mkdir -p "$OUT/raw"

git -C "$REPO" rev-parse HEAD > "$OUT/git-rev.txt" 2>/dev/null || echo "no-git" > "$OUT/git-rev.txt"
{
  echo '{'
  echo "  \"utc\": \"$STAMP\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"bench\": \"$BIN\","
  echo "  \"git_rev\": \"$(cat "$OUT/git-rev.txt")\","
  echo "  \"repetitions\": $REPS,"
  echo "  \"device\": \"$DEVICE\","
  echo "  \"threads\": $THREADS,"
  echo "  \"profiles\": \"$PROFILE_KEYS\","
  echo "  \"variants\": \"$VARIANTS\""
  echo '}'
} > "$OUT/manifest.json"

# Model path must be native C:/... for Windows binaries under MSYS bash.
case "$MODEL" in
  /c/*) MODEL_NATIVE="C:${MODEL#/c}" ;;
  *)    MODEL_NATIVE="$MODEL" ;;
esac

profile_args() {
  case "$1" in
    chat)        echo "-p 128 -n 32" ;;
    coding)      echo "-p 512 -n 64" ;;
    agent)       echo "-p 1024 -n 32" ;;
    saturation)  echo "-pg 512,64 -pg 1024,64 -pg 2048,64" ;;
    *) return 1 ;;
  esac
}

echo "variant,profile,prompt,gen,rep_count,avg_ts,stddev_ts,cv,device,threads,git_rev,notes" > "$OUT/results.csv"

IFS=',' read -ra VLIST <<< "$VARIANTS"
IFS=',' read -ra PLIST <<< "$PROFILE_KEYS"
for variant in "${VLIST[@]}"; do
  case "$variant" in
    baseline) VCMD=(-ngl 99 -ncmoe 12) ;;
    loopopt)  VCMD=(-ngl 99 -ncmoe 0 -ot 'blk\.(0|1|2|3|4|5|18|19|20|21|22|23)\.ffn_.*_exps=CPU') ;;
    *) echo "unknown variant $variant" >&2; exit 2 ;;
  esac
  for profile in "${PLIST[@]}"; do
    pargs="$(profile_args "$profile")" || { echo "unknown profile $profile" >&2; exit 2; }
    # shellcheck disable=SC2206
    PARGS=($pargs)
    cmd=("$BIN" -m "$MODEL_NATIVE" -o json -r "$REPS" -dev "$DEVICE" -t "$THREADS" "${VCMD[@]}" "${PARGS[@]}")
    echo "+ ${cmd[*]}" | tee -a "$OUT/commands.log"
    "${cmd[@]}" > "$OUT/raw/${variant}-${profile}.json" 2> "$OUT/raw/${variant}-${profile}.stderr" || {
      echo "FAILED ${variant}/${profile}; see raw stderr" >&2
      tail -20 "$OUT/raw/${variant}-${profile}.stderr" >&2
      exit 3
    }
  done
done

case "$OUT" in
  /c/*) OUT_PY="C:${OUT#/c}" ;;
  *)    OUT_PY="$OUT" ;;
esac
"$PY" - "$OUT_PY" <<'PY'
import csv, json, math, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = []
for jf in sorted((out / "raw").glob("*.json")):
    stem = jf.stem
    variant, profile = stem.split("-", 1)
    try:
        raw = jf.read_text(encoding="utf-8", errors="replace")
        start = raw.find("[")
        end = raw.rfind("]")
        data = json.loads(raw[start:end+1] if start >= 0 and end > start else raw)
    except Exception as e:
        print(f"skip {jf.name}: {e}")
        continue
    if isinstance(data, dict):
        data = [data]
    for rec in data:
        if not isinstance(rec, dict) or "avg_ts" not in rec:
            continue
        avg = float(rec.get("avg_ts", 0.0))
        sd = float(rec.get("stddev_ts", 0.0))
        rows.append(dict(variant=variant, profile=profile,
                         prompt=rec.get("n_prompt", ""), gen=rec.get("n_gen", ""),
                         rep_count=len(rec.get("samples_ts", [])) or "",
                         avg_ts=f"{avg:.4f}", stddev_ts=f"{sd:.4f}",
                         cv=f"{(sd/avg if avg else 0.0):.5f}",
                         device=rec.get("gpu_info", ""), threads=rec.get("n_threads", ""),
                         git_rev=(out / "git-rev.txt").read_text(encoding="utf-8").strip(),
                         notes=rec.get("model_type", "")))
with (out / "results.csv").open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["variant","profile","prompt","gen","rep_count","avg_ts","stddev_ts","cv","device","threads","git_rev","notes"])
    writer.writerows(rows)
lines = ["# Benchmark summary", "", "| variant | profile | pp t/s | tg t/s | CV |", "|---|---|---:|---:|---:|"]
for r in rows:
    pp = float(r["avg_ts"]) if int(r["gen"] or 0) == 0 else 0.0
    tg = float(r["avg_ts"]) if int(r["gen"] or 0) > 0 else 0.0
    lines.append(f"| {r['variant']} | {r['profile']} | {pp:.2f} | {tg:.2f} | {r['cv']} |")
lines += ["", "Decision rule: docs/BENCHMARK-WORKFLOW.md. Keep raw JSON for audit.", ""]
(out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(f"parsed {len(rows)} benchmark rows")
PY
echo "BENCH_DIR=$OUT"
