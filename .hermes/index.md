# bytropix Mind Palace Index (June 14, 2026)

## Walkway Files (Read Order)
1. `prestige_prompt.md` — Full resume: mission, architecture, priority queue
2. `goal-mantra.md` — Single pasteable block: STATE, BUILD, NEXT, VAULT
3. `state.md` — Live status: model detection, dynamic dims, blocker list
4. `plan.md` — Roadmap: multi-model adapter, GPU kernels, benchmarks

## Architecture Deep-Dives (Mind Palace)

| File | Model | Content |
|------|-------|---------|
| `paradigm-shift-gemma4.md` | Gemma 4 12B | Full architecture map, tensor names, forward pass, ISWA pattern |
| `diffusiongemma-integration.md` | DiffusionGemma-26B | Integration notes, blockers, dynamic dims |
| `tier4-validation/13-benchmarks/gemma4-baseline.md` | Gemma 4 12B | Benchmark plan, comparison targets |

## Supporting Files
| File | Purpose |
|------|---------|
| `./hermes/STATUS.md` | True state: works/broken/priorities (Qwen era, outdated) |
| `./hermes/README.md` | Vault index |
| `./hermes/unsloth-qwen3.6-quant-formula.md` | Qwen3.6 per-tensor quantization map |

## Multi-Model Architecture (June 14)

**Three architectures, one codebase:**
- **DiffusionGemma-26B**: 30 GQA + MoE (128 experts, top-8), d_model=2816, heterogeneous head_dim
- **Gemma 4 12B**: 48 ISWA (dense), d_model=3840, dual RoPE, QAT quantized
- **Qwen3.6-35B**: 30 SSM + 10 GQA + MoE (256 experts), D_MODEL=2048

**Adapter pattern**: `wubu_model.c` detects naming convention → extracts dims from GGUF → configures per-layer → allocates dynamic KV cache

## Key Paths
- Source: `/home/wubu/bytropix/`
- Models: `/home/wubu/models/` (DiffusionGemma, Gemma4, Qwen3.6)
- llama.cpp ref: `/home/wubu/llama.cpp/`
- Benchmark: `./bench_512k_full <model.gguf> 4096 1 0`
