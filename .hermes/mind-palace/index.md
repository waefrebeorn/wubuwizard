# bytropix Mind Palace — May 28, 2026

## Active Branch
- `cpu-optimize-may26` — all parity fixes, compilation flags fix, persistent KV
- `master` — stable (push-guarded)

## Core Documents
| Document | Purpose | Status |
|----------|---------|--------|
| `state.md` | Current performance state, benchmarks, compilation flags fix | ✅ updated May 28 |
| `plan.md` | Next steps, priority | ✅ updated May 28 |
| `walkway.md` | Step-by-step path | ✅ updated May 28 |
| `goal-mantra.md` | Goal and operating loop | ✅ updated May 28 |
| `goal-paste-agent.md` | Session start paste | ✅ updated May 28 |
| `bytropix-300-gap-battleship.md` | 300-gap fresh analysis | ✅ updated May 28 |
| `bytropix-accomplishments-vaulted.md` | All completed work archived | ✅ |
| `prestige_prompt.md` | Session resume prompt | ⬜ stale (May 21) |
| `project.md` | Project overview | ⬜ stale (May 21) |
| `entry.md` | Build commands | ⬜ stale (May 16) |
| `testing.md` | Testing protocol | ⬜ stale (May 21) |
| `fresh_start_prompt.md` | Boot prompt | ⬜ stale (May 17) |
| `overnight-map.md` | Autonomous session navigation | ⬜ stale (May 21) |

## Status
- **Context growth penalty**: ✅ ELIMINATED (persistent KV: 7.9×, per-turn constant ~31s)
- **Compilation flags**: ✅ `-ffast-math` → `-fno-fast-math` (IEEE 754 compliance)
- **Cos-sim vs llama.cpp**: 0.976 (improved from 0.974 with compilation fix)
- **All 3 cos-sim regression tests**: ✅ pass at 0.975 threshold
- **Cos-sim between fast/no-fast builds**: 0.99975580, top-5 argmax identical

## Key Paths
- Models: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (10.7 GB)
- Embeddings: `~/bytropix/data/qwen36_embeddings_c.bin.raw` (2 GB)
- Tokenizer: `~/bytropix/data/vocab.bin` + `~/bytropix/data/merges.bin`
- Reference: `~/llama.cpp/build/bin/llama-cli` (BLAS-linked)

## Build
```bash
cd ~/bytropix && make gen_text_cpu
OMP_NUM_THREADS=4 MODEL=~/models/... ./gen_text_cpu "prompt" 100 40
```

## Vault & Memory Direction
- Discovery → write `vault/[topic].md` + `memory target:memory content:"vault vault/[topic].md — one-line what it covers"`
- Palace insight → `memory target:memory content:"mind palace mind-palace/[path] — one-line update"`
- Preference → `memory target:user content:"wubu prefers [preference]"`
