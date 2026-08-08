# System State — 2026-05-25 18:48

## Heartbeat Status
- pm-money-loop: 🟢 33s — core money loop healthy
- room-feed: 🟢 33s — C room data feed healthy
- cron-health-watchdog: 🟡 154s — 5min cycle, within tolerance
- monkey-scraper: 🟡 639s — 15min cycle, normal range
- polygon-wallet: 🟡 638s — 15min cycle, normal range
- kraken-collector: 🟡 506s
- btc-wallet: 🔴 1786s — hourly cycle, 30min since last, ok
- money-loop-daily: 🔴 24471s (~7h ago) — fixed, next fire midnight
- data-collector/cleaner/live-news: stale ~23min — orphaned heartbeats, no cron jobs found

## Fixed Today
- money-loop-daily: was error (paid model). Pinned to nous:deepseek/deepseek-v4-flash:free
- timeline-collector: was error (script path has full cmd). Created ~/.hermes/scripts/timeline_collector.sh wrapper
- RustChain miner: started then killed (user runs Windows miner)
- Papers README: corrected HASHMIND → actual paths (bytropix/ENCODERS/hash-mind/ and bytropix/THEORY/math_viz/)
- Downloaded 8 missing arxiv papers → 30 total

## Papers Vault
- 30 papers total (22 original + 8 new downloaded today)
- Hyperbolic, sparse attention, DeepSeek, MoE, LLM papers
- 4 WuBu papers (DCT, DFT, GAAD ST1/ST2)
- All path references corrected from old HASHMIND/ → actual locations

## Timeline DB
- 2.5GB, 7.6M+ rows, 15yr span (2011-2026)
- Updates every 15-20min via monkey-scraper + timeline-collector

## PM Pipeline
- 5-min money loop running (pm_money_loop.py via wrapper)
- Traderoom: 3W/9L (25% WR) — recent resolved trades 3/3 correct (improving)
- Top ecosystem agents: 76.6% WR, $4.8K PnL (Gen#2750)
- All signals bearish (BTC 0%, ETH -2.35, SOL -6.5, XRP -6.43)
- ETH showing UP signal at 52.3% confidence

## RustChain Miner
- Wallet: RTC17c0d21f04f6f65c1a85c0aeb5d4a305d57531096
- User runs Windows miner (not WSL)
- Windows miner failing with TLS connection error to rustchain.org

## bytropix Inference Engine
- Pure C engine for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)
- GPU SSM decode ~5.9 tok/s on RTX 5050 (8GB VRAM)
- **Blocker:** GPU SSM divergence from CPU (cos-sim -0.66) — anti-correlated
- Vision encoder: 384 LoC 3D ViT ported + mmproj projection
- 8 commits not pushed to remote
- Plan: Phase 29 = fix GPU divergence, Phase 30 = fix CPU build + push

## WuBu Mind Codebase (ENCODERS/hash-mind/)
- WuBuMindJAX v1-v7 — JAX nested hyperbolic implementations
- WuBuNest_Trainer.py (200KB) + WuBuNest_Inference.py — main training pipeline
- wubu_nesting_impl.py / wubu_nesting_example.py — core implementation
- SimpleHash V1-V3 — hash-based attention approximations
- C port at ENCODERS/hash-mind/c/ — simplehash.c, wubu_math.c/.h + Makefile
- GAAD-WuBu-ST1/ST2 papers + WuBuNestingv0.1 paper/PDF

## Optimizers (bytropix/OPTIMIZERS/)
- Q-controller: JAX 45 lines, 10-state × 5-action Q-table, ε-greedy exploration
  - qcontroller.py (3.8KB) at ~/bytropix/OPTIMIZERS/q-controller/
  - qlearnerexample.py (9KB) — standalone Q-learning example
  - pidexample.py (7.6KB) — PID controller
  - toroidexample.py (8.4KB) — toroidal gradient decomposition
- PID Lambda controller for second-order loss balancing
- Source code exists at these paths (not at old HASHMIND/ reference)

## Math Viz (Lean Proofs)
- 15/18 theorems proven in Lean 4
- 3 unproven: curvature scaling (matrix exp), Φ_inv_right, volume-preserving
- Holographic optimizer edge case: domain Z × (-π, π] not Z × [-π, π]
- Python scripts all pass numerically (error < 1e-15)

## CPU Inference Mission — Status
**Goal:** Optimize bytropix CPU → match llama.cpp CPU → 512K context in llama engine for agentic flow

### Done
- ✅ Installed libopenblas-dev (0.3.32)
- ✅ Cloned llama.cpp (ggml-org/llama.cpp, 91MB)
- ✅ Fix bytropix CPU build — `make gen_text_cpu` compiles and links clean
- ✅ Added `-lopenblas` to bytropix LDFLAGS for BLAS integration

### In Progress
- 🟡 Downloading non-MTP GGUF (IQ2_XXS): ~2.3GB / ~9-10GB
- 🟡 Downloading MTP GGUF (IQ2_XXS): ~2.0GB / ~9-10GB
- 🟡 Building llama.cpp CPU-only with OpenBLAS (cmake --build -j4)

### Next
- Benchmark: llama.cpp CPU tok/s at 4K → 128K
- Benchmark: bytropix `gen_text_cpu` same model
- Profile gaps → optimize SSM + GQA + dequant bottlenecks
- Wire BLAS into GQA prefill path
- Scale to 512K context