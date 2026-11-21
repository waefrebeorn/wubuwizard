# Hardware & Model Tiers

One codebase. Tiers are configurations and gates, not forks. A module that cannot run on
the lowest tier it claims is broken on that tier — mark it honestly.

## Tier ladder

| Tier | Hardware | Model quant | What it must do |
|------|----------|-------------|-----------------|
| T0 | CPU only, 16 GB RAM, SSD | `UD-IQ2_XXS` (8.49 GB) | Load, decode, agentic step. No GPU required |
| T1 | 6 GB VRAM, 16 GB RAM, SSD | `UD-IQ2_XXS` (8.49 GB) + SSD offload | **Everyman target.** Hot weights in VRAM, rest streamed |
| T2 | 12–16 GB VRAM, 32 GB RAM | `UD-Q4_K_XL` (17.63 GB) | Full 4-bit in VRAM, longer context |
| T3 | 24 GB+ VRAM, 64 GB RAM | `UD-Q8_K_XL` (~24 GB) | Lossless quant, wide context |
| T4 | Multi-GPU / DGX class | Native NVFP4 + MXFP8 (~20 GB) | Training math, variable-loop research |

Local machine: RTX 4050 Laptop 6 GB + 16 GB RAM + SSD = **T1**.

## Quant reference (Inkling 2 Mini, vendor/Unsloth reported)

| Quant | Size | Top-token agreement vs Q8 | Tier |
|-------|------|---------------------------|------|
| UD-IQ1_S | 7.76 GB | 89.4% | T0/T1 floor |
| UD-IQ2_XXS | 8.49 GB | 90.6% | **T1 primary** |
| UD-Q2_K_XL | 9.14 GB | — | T1 optional |
| UD-IQ3_XXS | 11.7 GB | — | T1/T2 boundary |
| UD-Q4_K_XL | 17.63 GB | 98.0% | **T2 primary** |
| UD-Q8_K_XL | ~24 GB | lossless | T3 |

Source: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
Sizes are published file sizes; verify locally after acquisition.

## Acquired weights

| Model | Path | Status |
|-------|------|--------|
| Inkling 2 Mini UD-IQ2_XXS | `models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf` | Acquired, verified GGUF v3, 8,486,236,352 B (T1) |
| Inkling 2 Mini UD-Q4_K_XL | `models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_XL.gguf` | Acquired, verified GGUF v3, 17,629,759,456 B (T2) |
| Gemma 4 12B UD-IQ2_M | `models/gemma-12b/` | Present |

## Tier gates

| Gate | Meaning |
|------|---------|
| `make test_tier0` | CPU decode works with no GPU present |
| `make test_budget_6gb` | Allocator refuses >6 GB VRAM on T1; streams instead |
| `make test_ssd_stream` | Sequential SSD weight streaming, no random I/O |
| `make test_tier1_e2e` | Load → decode → agentic step on T1 hardware |
| `make test_tier2_e2e` | Q4_K_XL full VRAM path |
| `make test_backend_cpu/cuda/hip` | Same kernel output across backends |

## Backend ladder

1. **CPU** — always correct, reference for differential tests.
2. **CUDA** — current accelerated backend, RTX 4050 6 GB.
3. **HIP/ROCm** — add behind the same dispatch; no CUDA intrinsics in model kernels.

Backend choice is runtime, family specialization is compile-time. Never bake a vendor into
a model header.

## Rule

Higher tiers get more capability, never a different architecture. If a feature only works
on T3, the T1 path must degrade explicitly (stream, smaller quant, or documented refusal) —
never silently break.
