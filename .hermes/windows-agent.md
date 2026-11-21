# bytropix — Windows Agent Walkway

## Purpose
Windows-side agent context for bytropix development. Tracks Windows-specific concerns alongside the main Linux agent's work.

## Key Decision
The Linux agent (previous Hermes) maintains the main mind palace at `.hermes/`. This file is the Windows agent's coordination point. Do NOT overwrite existing mind palace files.

## Project Identity
- **Name:** bytropix — WuBu Text AI Multi-Modal Inference Engine
- **Repo:** https://github.com/waefrebeorn/bytropix
- **Local Path:** `/c/Users/eman5/bytropix/`
- **Main Mind Palace:** `.hermes/mind-palace/` (maintained by Linux agent)
- **Linux Agent's Goal Mantra:** Phase 28r — CPU path stable at 8.9 tok/s, chunked SSM fixed

## What the Linux Agent Tracks
- GPU SSM/GQA kernels, CUDA code
- Vision pipeline (GPU ViT + mmproj)
- MTP spec decode
- Layer-by-layer cos-sim debugging
- Phase roadmap (P0-P3)

## What the Windows Agent Tracks
- Windows build compatibility
- Cross-compile from Windows (MinGW-w64 or MSVC)
- Git sync between Windows fork and Linux origin
- Windows-specific testing

## Current State (as of June 2, 2026)
| Component | Phase | Status |
|-----------|-------|--------|
| CPU inference | 28r | ✅ 8.9/17.8 tok/s |
| GPU hybrid text | 28r | ⚠️ Net-negative vs CPU |
| MTP spec decode | 28r | ✅ 8.5 tok/s, 4% acceptance |
| Vision pipeline | 28r | ✅ 15.7s total |
| Chunked SSM CS=1 | 28r | ✅ Exact |
| Chunked SSM CS>1 | 28r | ⚠️ FP accumulation |
| GPU MoE | 28r | ❌ Fundamental code-path diff |

## Key Files
| Path | Purpose |
|------|---------|
| `.hermes/goal-mantra.md` | Latest phase state |
| `.hermes/STATUS.md` | Ground truth metrics |
| `.hermes/mind-palace/vault/` | Historical snapshots |
| `src/cuda_kernels.cu` | GPU kernels |
| `src/wubu_model.c` | Core model |
| `Makefile` | Build system |

## Coordination Protocol
1. Read `.hermes/goal-mantra.md` before every session
2. Read `STATUS.md` for ground truth
3. Check `.hermes/mind-palace/vault/bins/` for recent snapshots
4. Never overwrite Linux agent's files
5. Write Windows-specific notes to `.hermes/windows-note-YYYYMMDD.md`
