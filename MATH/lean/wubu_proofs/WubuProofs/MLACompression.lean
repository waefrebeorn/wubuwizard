import Mathlib
open scoped Matrix.Norms.Frobenius
open Real Matrix

set_option maxHeartbeats 0

/-
MLACompression.lean — DeepSeek-style Multi-head Latent Attention KV
compression, machine-verified.

Two results:

1. **Factor identity** (algebra): the compression error factors through the
   weight difference:
       H·W_K − H·W_DKV·U_K = H·(W_K − W_DKV·U_K)

2. **Frobenius error bound** (analysis): the compression error on any hidden
   state is bounded by the weight difference scaled by ‖H‖_F:
       ‖H·W_K − H·W_DKV·U_K‖_F ≤ ‖H‖_F · ‖W_K − W_DKV·U_K‖_F

   via mathlib's `frobenius_norm_mul` (valid for rectangular matrices, not
   just square — the earlier sketch note claiming mathlib restricts to
   square matrices was wrong; `Mathlib.Analysis.Matrix.Normed` provides
   `frobenius_norm_mul : ‖A * B‖ ≤ ‖A‖ * ‖B‖` for
   `A : Matrix l m α`, `B : Matrix m n α`).

The optimality claim (that the truncated SVD *minimizes* this error among
all rank-d_c factorizations — Eckart–Young–Mirsky) is NOT proven here; it
would require an SVD library (FLYWHEEL has it but pins Lean 4.26 while we
are on 4.29). The one-sided bound above is what WuBu's argument needs and
is fully machine-checked.
-/

/-- MLA compression factor identity: the error factors through the
    weight difference. -/
theorem mla_compression_factor {T D d_c K : ℕ}
    (H_states : Matrix (Fin T) (Fin D) ℝ)
    (W_DKV : Matrix (Fin D) (Fin d_c) ℝ)
    (U_K : Matrix (Fin d_c) (Fin K) ℝ)
    (W_K_full : Matrix (Fin D) (Fin K) ℝ) :
    H_states * W_K_full - H_states * W_DKV * U_K
      = H_states * (W_K_full - W_DKV * U_K) := by
  calc
    H_states * W_K_full - H_states * W_DKV * U_K
        = H_states * W_K_full - H_states * (W_DKV * U_K) := by
          rw [Matrix.mul_assoc]
    _ = H_states * (W_K_full - W_DKV * U_K) := by rw [Matrix.mul_sub]

/-- MLA compression Frobenius error bound: the compression error on hidden
    states is bounded by ‖H‖_F times ‖W_K − W_DKV·U_K‖_F. -/
theorem mla_compression_error_bound {T D d_c K : ℕ}
    (H_states : Matrix (Fin T) (Fin D) ℝ)
    (W_DKV : Matrix (Fin D) (Fin d_c) ℝ)
    (U_K : Matrix (Fin d_c) (Fin K) ℝ)
    (W_K_full : Matrix (Fin D) (Fin K) ℝ) :
    ‖H_states * W_K_full - H_states * W_DKV * U_K‖
      ≤ ‖H_states‖ * ‖W_K_full - W_DKV * U_K‖ := by
  rw [mla_compression_factor]
  exact frobenius_norm_mul H_states (W_K_full - W_DKV * U_K)
