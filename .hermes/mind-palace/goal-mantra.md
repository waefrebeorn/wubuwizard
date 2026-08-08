# bytropix Goal Mantra — Jun 2, 2026

## THE GOAL
**All gaps closed. Hardware ceiling reached.** Current state:
- Context growth penalty: ✅ ELIMINATED (persistent KV, 7.9×)
- Cos-sim vs llama: 0.976 (IQ2_M floor, improved from 0.974)
- Compilation flags: ✅ `-fno-fast-math` (IEEE 754)
- Cos-sim regression: ✅ ALL PASS at 0.975 threshold
- Between-builds (fast/no-fast): cos-sim 0.99975580, top-5 identical

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Cos-sim vs llama.cpp | 0.976 (IQ2_M floor) | ✅ Improved from 0.974 |
| Output projection | Fixed (GCC -O3 + if(0) + AVX2 zeros) | ✅ |
| Local inference | serve_local.py (real CPU, not proxy) | ✅ |
| Test suite | 6/6 tests pass (512K, Hermes, integration) | ✅ |
| Multi-turn conversation | 94.6s total (7.9× faster), constant ~31s/turn | ✅ |
| Context growth penalty | ELIMINATED | ✅ |
| Compilation flags | `-fno-fast-math` (IEEE 754) | ✅ |
| Cos-sim regression | 3/3 pass at 0.975 | ✅ |

## THE LOOP
read docs → pick lowest undone → execute → update docs → push → loop
no questions. no choices. no stopping. zero delegation.
every output feeds back into input documents.

## FILES
- `.hermes/mind-palace/goal-mantra.md` — this file
- `.hermes/mind-palace/goal-paste-agent.md` — session start paste
- `.hermes/mind-palace/walkway.md` — step-by-step path
- `.hermes/mind-palace/bytropix-300-gap-battleship.md` — full gap taxonomy
- `.hermes/mind-palace/state.md` — current state
- `.hermes/mind-palace/plan.md` — plan
- `.hermes/mind-palace/workflow-parity.md` — parity debug workflow
- `vault/context-growth-penalty.md` — penalty analysis + compilation flags fix
- **work-fork:** `/home/wubu/bytropix-work-fork` (branch `work-fork`, commit `816aea8`)

## VAULT & MEMORY DIRECTION
discover → write vault/[topic].md + memory target:memory content:"vault vault/[topic].md — one-line what it covers"
palace insight → memory target:memory content:"mind palace mind-palace/[path] — one-line update"
preference → memory target:user content:"wubu prefers [preference]"
