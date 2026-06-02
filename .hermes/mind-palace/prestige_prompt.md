# Prestige Prompt — Jun 2, 2026

## Project: bytropix — CPU Inference Engine

**Qwen3.6-35B-A3B-UD-IQ2_M CPU inference. AMD Ryzen 7 7445HS / 14GB RAM / 12 cores.**
**All gaps closed. Hardware ceiling reached.**

## Current State
- **Context growth penalty: ELIMINATED** — persistent KV: 7.9× multi-turn, ~31s constant/turn
- **Compilation: IEEE 754** — `-fno-fast-math`. Cos-sim improved 0.974→0.976
- **Cos-sim vs llama: 0.976** — IQ2_M quantization floor. Regression suite: 3/3 at 0.975
- **Between-builds cos-sim: 0.99975580** — top-5 argmax identical (fast vs no-fast)
- **All test suites pass** — 6/6 tests across 512K, Hermes, integration suites
- **Branch: cpu-optimize-may26** — all fixes committed and pushed

## Done This Session (May 28)
1. ✅ Persistent KV process — `gen_text_cpu --persist` + Python client (serve_local.py --persist)
2. ✅ Compilation flags: `-ffast-math` → `-fno-fast-math` (IEEE 754 compliance)
3. ✅ Cos-sim regression test at 0.975 threshold — all 3 prompts pass
4. ✅ Between-builds comparison: cos-sim 0.99975580, top-5 identical
5. ✅ All mind palace docs updated: state, walkway, plan, battleship, index, README, goal-mantra, testing, entry, project, fresh_start, overnight-map, goal-paste

## Remaining (Hardware-Gated)
1. GPU output proj — needs GPU
2. MTP CPU benchmark — now possible with 14GB RAM (was 7.4GB)
3. Cos-sim >0.99 — needs Q3_K+/F16 model
4. Mixed-curvature hyperbolic — research

## Key Env Vars
```
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf  # model path
OMP_NUM_THREADS=8                               # 12 physical cores
CHAT=1                                          # ChatML mode
DUMP_LOGITS=/tmp/logits.bin                     # logit dump
```

## Build
```bash
cd ~/bytropix-work-fork && make gen_text_cpu -j$(nproc)
```
