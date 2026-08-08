# Research Papers — WuBuText AI

> **Location:** `/home/wubu/bytropix/THEORY/papers/`
> **Total:** 30 papers (all arxiv abstracts as markdown)
> **Last updated:** 2026-05-25

---

## DeepSeek Core Papers (10)

| # | arXiv | Title | Key For WuBu |
|---|-------|-------|-------------|
| 1 | [2401.02954](https://arxiv.org/abs/2401.02954) | DeepSeek LLM: Scaling Open-Source LLMs | Baseline scaling laws |
| 2 | [2401.06066](https://arxiv.org/abs/2401.06066) | DeepSeekMath: Pushing Limits of Math Reasoning | Math pretrain data |
| 3 | [2402.03300](https://arxiv.org/abs/2402.03300) | DeepSeek-Coder: LLM Meets Programming | Code training data |
| 4 | [2404.06585](https://arxiv.org/abs/2404.06585) | DeepSeek-VL: Vision-Language Understanding | Vision encoder pattern |
| 5 | [2405.04434](https://arxiv.org/abs/2405.04434) | **DeepSeek-V2: MoE Language Model** | MLA attention origin |
| 6 | [2409.04783](https://arxiv.org/abs/2409.04783) | DeepSeek-Prover-V1.5 | Formal proof integration |
| 7 | [2412.19437](https://arxiv.org/abs/2412.19437) | **DeepSeek-V3 Technical Report** | MLA + MoE + aux-loss-free |
| 8 | [2501.12948](https://arxiv.org/abs/2501.12948) | **DeepSeek-R1: Reasoning via RL** | GRPO training method |
| 9 | [2502.02529](https://arxiv.org/abs/2502.02529) | DeepSeek-Prover-V2 | Lean/Coq proof gen |
| 10 | [2512.02556](https://arxiv.org/abs/2512.02556) | **DeepSeek-V3.2: Pushing Frontier** | ⭐ DSA Sparse Attention + RL scaling |

## Sparse Attention Papers (6)

| # | arXiv | Title | Key For WuBu |
|---|-------|-------|-------------|
| 11 | [2502.14864](https://arxiv.org/abs/2502.14864) | Delta Attention | Mask optimization |
| 12 | [2502.17529](https://arxiv.org/abs/2502.17529) | Sparse Linear Algebra with CUDA | SpMV kernel patterns |
| 13 | [2502.04461](https://arxiv.org/abs/2502.04461) | Sparse Attention for Knowledge Distillation | Compression patterns |
| 14 | [2503.10488](https://arxiv.org/abs/2503.10488) | **Native Sparse Attention (NSA)** | Block-sparse masks |
| 15 | [2503.09542](https://arxiv.org/abs/2503.09542) | **Gated Sparse Attention** | Linear attention int. |
| 16 | [2505.04888](https://arxiv.org/abs/2505.04888) | MISA: Missing-Interaction-Aware Sparse Attn | Sparse GPU kernels |

## Hyperbolic Geometry Papers (4)

| # | arXiv | Title | Key For WuBu |
|---|-------|-------|-------------|
| 17 | [1705.08039](https://arxiv.org/abs/1705.08039) | Poincaré Embeddings for Hierarchical Rep. | Foundational Poincaré |
| 18 | [1802.03367](https://arxiv.org/abs/1802.03367) | Hyperbolic Neural Networks (Ganea) | Gyrovector formalism |
| 19 | [2205.04641](https://arxiv.org/abs/2205.04641) | Fully Hyperbolic Neural Networks | End-to-end hyperbolic |
| 20 | [2311.11394](https://arxiv.org/abs/2311.11394) | Möbius Transformers (Hyperbolic Attention) | ⭐ Hyperbolic transformer |

## Efficient ML Papers (2)

| # | arXiv | Title | Key For WuBu |
|---|-------|-------|-------------|
| 21 | [2010.11929](https://arxiv.org/abs/2010.11929) | ViT: An Image is Worth 16x16 Words | Vision encoder |
| 22 | [2503.16160](https://arxiv.org/abs/2503.16160) | Long CoT meets Lean Formal Proofs | Lean + RL for reasoning |

## Key GitHub Repos to Study

| Repo | Stars | Notes |
|------|-------|-------|
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 12.6K | CUDA kernels for Multi-head Latent Attention |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 7.2K | FP8 GEMM with fine-grained scaling |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | 9.6K | Expert-parallel communication library |
| [3FS](https://github.com/deepseek-ai/3FS) | 9.9K | Distributed filesystem for training |
| [DualPipe](https://github.com/deepseek-ai/DualPipe) | 3.0K | Pipeline parallelism |

## WuBu Research Papers (4 — from HASHMIND research repo)

| # | Title | Size | Key Content |
|---|-------|------|-------------|
| 23 | **GAAD-WuBu-ST1** | 27KB | Golden Aspect Adaptive Decomposition: Recursive Golden Subdivision + Phi-Spiral for video understanding |
| 24 | **GAAD-WuBu-ST2** | 43KB | Integrated GAAD-WuBu-ST: φ-infused nested hyperbolic with spatio-temporal video framework |
| 25 | **DFT-WuBu** | 31KB | DFT-enhanced WuBu: spectral decomposition + hyperbolic geometry |
| 26 | **DCT-WuBu** | 20KB | DCT-enhanced WuBu: discrete cosine transform integration |

These live at `~/bytropix/ENCODERS/hash-mind/` — the WuBu Mind JAX implementations (WuBuMindJAX*.py, WuBuNest_Trainer.py 200KB, wubu_nesting_impl.py), plus C port at `~/bytropix/ENCODERS/hash-mind/c/`.

## Runnable Math Proofs (`THEORY/math_viz/`)

| Script | What It Proves | Run It |
|--------|---------------|--------|
| `01_nested_hyperbolic_spaces.py` | Nested Poincaré disks with φ-scaled curvatures | `pip install matplotlib numpy && python3 THEORY/math_viz/01_nested_hyperbolic_spaces.py` |
| `02_golden_ratio_decomposition.py` | GAAD: recursive φ-subdivision + phi-spiral | Same |
| `03_poincare_clock.py` | Holographic optimizer: soul/echo decomposition exact (error < 1e-15) | Same |
| `04_lie_group_nesting.py` | WuBu as principal G-bundle with SO(n) connection | Same |
| `05_fiber_bundle_proof.py` | Fiber bundle structure proof | Same |
| `06_symplectic_optimizer.py` | Symplectic integration of RSGD in T(H^n) | Same |
| `07_lean_certificate.py` | Lean formal certificate generator | Same |

All live at `~/bytropix/THEORY/math_viz/` — Python + Lean dual-proven theorems.

## Systems Reference: Tailslayer (5 files — hedged reads / speculative execution)

| File | Size | Description |
|------|------|-------------|
| `tailslayer-notes.md` | 3.2KB | Relevance analysis: hedged reads → speculative decoding pattern match |
| `tailslayer-README.md` | 4.0KB | Original README — DRAM refresh tail latency via channel replication |
| `tailslayer-hedged-reader.hpp` | 7.8KB | Core C++ hedged reader implementation (template-based, N-way) |
| `tailslayer-trefi-probe.c` | 11KB | DRAM refresh timing probe (TSC calibration, jitter detection) |
| `tailslayer-example.cpp` | 2.4KB | Usage example |

**Connection to WuBuText:** The hedged read pattern (issue N identical reads across independent channels, take first response) is the **exact structural analog** of speculative decoding in LLMs — draft model proposes N candidates, target model verifies in parallel, accept longest valid prefix. The `N`-way replica management maps to MoE expert parallelism. The precise timing techniques inform CUDA profiling.

Origin: `~/HASHMIND/tailslayer/` — clone of [github.com/LaurieWired/tailslayer](https://github.com/LaurieWired/tailslayer)

## Papers Still to Download

✅ All previously missing papers downloaded 2026-05-25.

## PDFs Converted to Markdown

| File | Pages | Chars | Source |
|------|-------|-------|--------|
| `THEORY/WuBu_Nesting.md` (converted) | 10 | 27K | `THEORY/WuBu_Nesting.pdf` |
| `ENCODERS/hash-mind/WuBuNestingv0.1.md` (converted) | 13 | 36K | `ENCODERS/hash-mind/WuBuNestingv0.1.pdf` |
| `THEORY/03-wubu-nesting-paper.md` (human-written) | — | 57K | Original markdown paper |
