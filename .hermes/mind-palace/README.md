# bytropix Mind Palace — May 28, 2026

> **Inference FIXED.** Context growth penalty ELIMINATED. Compilation flags IEEE 754 compliant.
> Cos-sim 0.976 vs llama.cpp (IQ2_M floor — hardware ceiling reached).

## Quick Navigation

| File | Purpose | Status |
|------|---------|--------|
| `state.md` | Current performance state, benchmarks, known issues | ✅ updated May 28 |
| `plan.md` | Priority queue — all gaps closed | ✅ updated May 28 |
| `walkway.md` | Step-by-step path — all steps done | ✅ updated May 28 |
| `goal-mantra.md` | Goal and operating loop | ✅ updated May 28 |
| `goal-paste-agent.md` | Session start paste | ✅ NEW vague version May 28 |
| `index.md` | Full index | ✅ updated May 28 |
| `bytropix-300-gap-battleship.md` | 300-gap fresh analysis | ✅ updated May 28 |
| `workflow-parity.md` | Cos-sim parity debug workflow | ✅ accurate |
| `bytropix-accomplishments-vaulted.md` | All completed work archived | ✅ |
| `testing.md` | Test protocol | ⬜ stale — needs refresh |
| `fresh_start_prompt.md` | Boot prompt | ⬜ stale — needs refresh |
| `project.md` | Mission, phases | ⬜ stale — May 21 |
| `entry.md` | Build commands | ⬜ stale — May 16 |

## Key Facts (May 28)

- **Context growth penalty: ELIMINATED** by persistent KV (7.9×, per-turn constant ~31s)
- **Compilation: IEEE 754** — `-ffast-math` removed (was causing FP drift in 30 SSM layers)
- **Cos-sim vs llama.cpp: 0.976** — IQ2_M quantization floor. Up from 0.974.
- **Cos-sim regression: 3/3 pass at 0.975 threshold**
- **Between-builds cos-sim: 0.99975580** — top-5 argmax identical
- **All 6 test suite tests pass**
- **All actionable code gaps closed.** Remaining: hardware-gated (GPU, Q3_K+ model); MTP now possible with 14GB RAM

## Key Paths

- Models: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (10.7 GB)
- Embeddings: `~/bytropix/data/qwen36_embeddings_c.bin.raw` (2 GB)
- Reference: `~/llama.cpp/build/bin/llama-cli`

## Build

```bash
cd ~/bytropix && make gen_text_cpu
OMP_NUM_THREADS=4 MODEL=~/models/... ./gen_text_cpu "prompt" 100 40
```

## Structure

```
.hermes/
├── mind-palace/              ← YOU ARE HERE
│   ├── README.md              ← This file (v8 — May 28)
│   ├── index.md               ← Full index (updated May 28)
│   ├── state.md               ← State dashboard (updated May 28)
│   ├── plan.md                ← Priority queue (updated May 28)
│   ├── walkway.md             ← Step path (updated May 28)
│   ├── goal-mantra.md         ← Goal loop (updated May 28)
│   ├── goal-paste-agent.md    ← Session paste (NEW May 28)
│   ├── bytropix-300-gap-battleship.md  ← Gap analysis (updated May 28)
│   ├── workflow-parity.md     ← Parity workflow
│   ├── testing.md             ← Test protocol (stale)
│   ├── fresh_start_prompt.md  ← Boot prompt (stale)
│   ├── project.md             ← Project overview (stale)
│   ├── entry.md               ← Build commands (stale)
│   └── plans/                 ← Devil's advocate audits (stale)
└── vault/
    ├── README.md              ← Vault index
    ├── context-growth-penalty.md  ← Full analysis + compilation fix
    ├── LEGACY.md              ← Legacy index
    └── bins/                  ← Archived old mind palace versions
```

## Vault & Memory Direction
Discovery → write `vault/[topic].md` + `memory target:memory`
Palace insight → `memory target:memory`
Preference → `memory target:user`
