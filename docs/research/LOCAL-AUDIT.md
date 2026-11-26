# Local audit — what the tree actually contains, before the external research lands

Written while the five research fronts run. This is the half no external source can supply: what THIS tree does.

## Corrections to my own framing

- **"pure scalar C, no SIMD" is false.** `immintrin`/`__m256`/`__m128` appear in ten files including
  `src/wubu_inkling2_fwd.c`, `src/wubu_gemm.c`, `src/wubu_fast_attn.c`, `src/quantized_matmul.c`,
  `src/wubu_bf16_gemv.c`, `src/wubu_dequant_fp4.c`. The engine is SIMD, not scalar.
- **"7.9 GB of weights re-read per token" is not supported.** The model is MoE (128 experts, 8 active per layer), so
  per-token weight traffic is the active set, not the file. The 7.9 GB / 0.11 s = 72 GB/s coincidence matches the DDR5
  ceiling but the bytes are not actually touched — that arithmetic proves nothing and I am not treating it as a finding.

## The real local finding: batched kernels exist and the engine never calls them

    src/wubu_gemm.c          batched matmul, AVX2
    src/wubu_fast_attn.c     flash-style attention with q8 KV cache

Both are exercised only by `tools/test_fast_attn*.c`, `tools/test_q8k_pqv.c`, `tools/test_splitk.c`. A grep for callers
outside their own translation units returns test harnesses and nothing else. `src/wubu_inkling2_fwd.c` contains no
reference to either.

Meanwhile the forward's hot loops are **token-innermost matvecs**: `for (int t = 0; t < n_tok; ++t)` appears at lines
883, 1097, 1171, 1191, 1261, 1300, 1326, 1345, 1393 calling `in2_matvec_quant` per token. So a prefill of N tokens issues
N single-token matvecs per weight matrix instead of one GEMM over N rows. The comment at line 618 states the principle
("applies it to all nt tokens, which is the whole reason batching matters: per-position calls would re-read the
weights") — and the code beside it does the per-position thing.

## Measured, for the record

    f32 prefill    N=60  0.117 s/token      N=120  0.110 s/token     flat in context
    TQ-6 prefill   N=60  0.479 s/token      N=120  0.827 s/token     excess = 0.00597*N s/token

Flat-in-context at f32 is the important shape: attention cost does not grow measurably over 60→120 tokens, so at these
lengths the per-token cost is dominated by the weight-matmul path, not by attention. That is consistent with N
single-token matvecs per layer and with `wubu_gemm` being unused.

## Questions the external reports must answer

1. What does a real GEMM-over-N-rows buy over N gemv calls for a 2-bit/3-bit quantised weight matrix — is the win the
   amortised weight read, the tensor-core path, or both, and what factor?
2. Does llama.cpp-class prefill actually batch the MoE expert matmuls across tokens, and what does its prefill tok/s
   look like for a 12B-class quantised model at 4k / 32k / 128k on a 6 GB laptop GPU?
3. Is flash-style attention (already sitting unused in `wubu_fast_attn.c`) the thing that stops the O(N^2) KV decode
   during prefill, or does it need the KV to be unquantised to help?
4. Which of the SOTA long-context mechanisms need retraining, and is any of them reachable from a frozen checkpoint.

## The batched quantised matmul exists, supports IQ2_XXS, and four of five models use it

`src/quantized_matmul.c:511`:

    // Batched quantized matmul: N input vectors through the same weight.
    // x: [N, D_MODEL] F32 input      W: quantized weight [D_MODEL, n_cols]
    // Key: weight data read ONCE from RAM, shared across all N tokens.
    // Supports: F32, IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K, Q4_K
    void quantized_matmul_batched(const float *x, const void *W, int weight_type,
                                  int64_t n_rows, int64_t n_cols,
                                  int64_t col_stride_bytes, int N, float *y);

It handles the exact weight types this model uses, it was written for exactly this purpose, and the callers are:

    src/lfm2_forward.c          src/lfm2_math.c
    src/wubu_gemma4_model.c     src/wubu_ssm.c
    src/wubu_poincare_gqa.c

Every model in the repo prefill-batches through it. **`src/wubu_inkling2_fwd.c` is the one that does not** - it calls
`in2_matvec_quant` per token at nine sites. Note also that `src/wubu_gemm.c` is f32-only and therefore useless here; the
quantised path is `quantized_matmul.c`, not `wubu_gemm.c`.

So the 8-hour prefill figure is not a property of the hardware, the model, or the maths. It is one model file not using a
kernel the other four use, with a doc comment that names the exact cost being paid ("weight data read ONCE ... shared
across all N tokens" implies the per-token path reads it N times).

## What wiring it must answer, in this order

1. **Speed.** Prefill N=120 f32 currently 13.2 s. A batched path should move it toward N x gemv / min(N, bandwidth), but
   the honest expectation is set by the measured gemv cost (0.41 ms per 2.6 MB expert slab = 6.3 GB/s effective, i.e. the
   IQ2 decode is ALU-bound, not bandwidth-bound) - so the win depends on how much of that 0.41 ms is weight *decode* that a
   GEMM amortises across N tokens. Measure it, do not assume the theoretical factor.
2. **Parity.** Changing accumulation order changes rounding. The 12-token oracle is exact today and llama.cpp itself
   batches prefill, so parity may survive - but it must be checked, and the prefill batch size must not become a variable
   that silently moves the reference tokens.
3. **Scope.** The wiring targets the dense projections (wq/wk/wv/wo, gate, up, down, lm_head) where the weight slab is
   re-read per token. Attention is a separate question and `wubu_fast_attn.c` is the mechanism for it.
