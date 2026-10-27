# Gemma 4 12B Baseline Benchmark

**Date:** 2026-06-10
**llama.cpp:** commit 039e20a2d, build b9588
**Model:** `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (Unsloth QAT Q4_K_XL)
**Hardware:** RTX 5050 8GB (sm_120) | i7-14650HX | 64GB DDR5

---

## Results

### Prompt Processing (512 tokens)

| Metric | Value |
|--------|-------|
| **Average** | **1,352.56 tok/s** |
| Samples | 1,397.66 / 1,216.88 / 1,409.67 / 1,394.16 / 1,344.44 |
| Stddev | 79.84 tok/s |

### Text Generation (256 tokens)

| Metric | Value |
|--------|-------|
| **Average** | **42.93 tok/s** |
| Samples | 41.78 / 41.89 / 47.22 / 41.93 / 41.82 |
| Stddev | 2.40 tok/s |

### Model Info

| Field | Value |
|-------|-------|
| Parameters | **11.91 B** |
| File size | **6.24 GiB** (6,700,531,904 bytes) |
| Type | Q4_0 (QAT variant) |
| KV cache | FP16, no flash attention |
| Layers on GPU | 99 (all) |

---

## Comparison: Gemma 4 12B vs Qwen3.6-35B-A3B

| Metric | Gemma 4 12B | Qwen3.6-35B (bytropix, GPU) |
|--------|-------------|------------------------------|
| Prompt | **1,352 tok/s** | ~500 tok/s (estimated) |
| Generation | **42.9 tok/s** | ~5.9 tok/s (GPU SSM path) |
| Active params | **11.91B** | ~3.5B (8/256 experts) |
| VRAM | **~6.3 GB** | ~5.1 GB |
| Quality | QAT q4_0 (trained quant) | Mixed IQ2-Q6 |
