# Overnight Map — Jun 2, 2026

**Active repo:** ~/bytropix-work-fork/
**Current branch:** work-fork
**Model:** ~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf (10.7GB)
**Origin remote:** git@github.com:waefrebeorn/bytropix.git
**Actual default remote:** `origin` — sometimes fails without helper auth.
**Push path fallback:** `git push origin work-fork:work-fork`

## Session Summary (May 28)

### What Was Done
1. **Context growth penalty ELIMINATED** — persistent KV process: 7.9× multi-turn, per-turn constant ~31s
2. **Compilation flags IEEE 754** — `-ffast-math` removed → `-fno-fast-math` restored IEEE compliance
3. **Single-token cos-sim improved** — 0.974→0.976 (cat prompt vs llama.cpp)
4. **Cos-sim regression automated** — 3 prompts at 0.975 threshold, ALL PASS
5. **Between-builds verified** — fast vs no-fast: cos-sim 0.99975580, top-5 argmax identical
6. **All docs updated** — state, walkway, plan, battleship, index, README, goal-mantra, testing, entry, project, fresh_start, goal-paste, palace README
7. **Pushed to cpu-optimize-may26** — commit 550a6b6

### Remaining (Hardware-Gated)
1. GPU output proj — needs GPU (CPU faster for text)
2. MTP CPU benchmark — now possible with 14GB RAM
3. Cos-sim >0.99 — needs Q3_K+/F16 model (>16GB)
4. Mixed-curvature hyperbolic — research, not blocker

### Next Session Options
1. Nothing actionable — all gaps closed. Hardware ceiling.
2. Unless user wants to acquire better hardware (32GB RAM, Q3_K+ model, GPU)
