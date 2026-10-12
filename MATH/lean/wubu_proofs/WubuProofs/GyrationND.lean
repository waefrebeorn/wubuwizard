import Mathlib
import WubuProofs.MobiusAddND
open Real InnerProductSpace EuclideanSpace

set_option maxHeartbeats 0

/-
GyrationND.lean — n-dimensional gyration machinery.

The Möbius gyration gyr_c[u,v] (Ungar's gyroautomorphism) is defined by the
loop

    gyr_c[u,v] w = (−(u ⊕_c v)) ⊕_c (u ⊕_c (v ⊕_c w))

Ungar's theorem (Analytic Hyperbolic Geometry, 2005, eq. 1.32–1.33): the
gyration preserves the inner product, hence the norm:

    ⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a, b⟩,    ‖gyr[u,v]w‖ = ‖w‖

so gyr[u,v] is a rotation of the ball about the origin.

This file proves the foundation everything else rests on: the Möbius norm
formula

    ‖a ⊕_c b‖² = (‖a‖² + ‖b‖² + 2⟨a,b⟩) / (1 + 2c⟨a,b⟩ + c²‖a‖²‖b‖²)

which is a pure polynomial identity after expanding ‖N‖² through
inner-product linearity (same expansion as MobiusAddND.lean), plus the
consequent gyration ball-preservation: every gyration argument stays inside
the ball, because gyr[u,v]w is built entirely from Möbius additions of
in-ball points and Möbius addition preserves the ball (MobiusAddND.lean).

The full inner-product preservation ⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a,b⟩ requires
the explicit gyrator matrix (Ungar's closed form gyr[u,v]w = w + Au + Bv
with the A, B, D coefficients) and is left as future work.
-/

/-- The Möbius gyration: gyr_c[u,v]w = (−(u⊕v)) ⊕ (u ⊕ (v⊕w)). -/
noncomputable def gyration_nd {n : ℕ} (c : ℝ) (u v w : EuclideanSpace ℝ (Fin n)) :
    EuclideanSpace ℝ (Fin n) :=
  mobius_add_nd c (-(mobius_add_nd c u v)) (mobius_add_nd c u (mobius_add_nd c v w))

/-- Möbius norm formula: ‖a ⊕_c b‖² has a closed form.

    ‖a ⊕_c b‖² = (‖a‖² + ‖b‖² + 2⟨a,b⟩) / (1 + 2c⟨a,b⟩ + c²‖a‖²‖b‖²)

The proof is the same inner-product expansion as MobiusAddND.lean; the
key algebraic identity (verified by `ring`) is

    ‖N‖² = (a + b + 2p)·D

where N = αa + βb, α = 1+2cp+cb, β = 1−ca, D = 1+2cp+c²ab,
a = ‖a‖², b = ‖b‖², p = ⟨a,b⟩.
-/
theorem mobius_norm_formula {n : ℕ} (c : ℝ) (hc : c > 0)
    (a b : EuclideanSpace ℝ (Fin n)) (ha : ‖a‖ ^ 2 < 1 / c) (hb : ‖b‖ ^ 2 < 1 / c) :
    ‖mobius_add_nd c a b‖ ^ 2
      = (‖a‖ ^ 2 + ‖b‖ ^ 2 + 2 * inner ℝ a b)
        / (1 + 2 * c * inner ℝ a b + c ^ 2 * ‖a‖ ^ 2 * ‖b‖ ^ 2) := by
  -- rename the pieces
  set p := inner ℝ a b with hp
  set A := ‖a‖ ^ 2 with hA
  set B := ‖b‖ ^ 2 with hB
  set D := 1 + 2 * c * p + c ^ 2 * A * B with hD
  -- the definition, once unfolded, is (1/D) • (α • a + β • b)
  have hN : mobius_add_nd c a b = (1 / D) • ((1 + 2 * c * p + c * B) • a + (1 - c * A) • b) := by
    unfold mobius_add_nd
    rw [hD, hA, hB, hp]
  -- basic inner-product facts
  have hA_eq : inner ℝ a a = A := (real_inner_self_eq_norm_sq a).trans hA.symm
  have hB_eq : inner ℝ b b = B := (real_inner_self_eq_norm_sq b).trans hB.symm
  -- (1) ‖(1/D) • N‖² = ‖N‖² / D²
  have hDpos : D > 0 := by
    have hcs : p ^ 2 ≤ A * B := by
      have h := real_inner_mul_inner_self_le a b
      rw [← hp, hA_eq, hB_eq] at h
      simpa [pow_two] using h
    have h1 : D = (1 + c * p) ^ 2 + c ^ 2 * (A * B - p ^ 2) := by unfold D; ring
    rw [h1]
    have h2 : c ^ 2 * (A * B - p ^ 2) ≥ 0 := by
      apply mul_nonneg (sq_nonneg c)
      linarith [hcs]
    have h3 : (1 + c * p) ^ 2 ≥ 0 := sq_nonneg _
    by_contra hD
    push Not at hD
    have h4 : (1 + c * p) ^ 2 = 0 := by linarith
    have h5 : c * p = -1 := by nlinarith [h4]
    have h6 : p ^ 2 = 1 / c ^ 2 := by
      have h7 : p = -1 / c := by
        have h8 : c * p = -1 := h5
        field_simp at h8 ⊢
        linarith
      rw [h7]
      ring
    have hca : c * A < 1 := by
      have h1 : c * A < c * (1 / c) := mul_lt_mul_of_pos_left ha hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    have hcb : c * B < 1 := by
      have h1 : c * B < c * (1 / c) := mul_lt_mul_of_pos_left hb hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    have hn1 : 0 ≤ c * A := by positivity
    have hn2 : 0 ≤ c * B := by positivity
    have hab_lt : A * B < 1 / c ^ 2 := by nlinarith [mul_nonneg hn1 hn2]
    linarith [h6]
  have hnorm_div :
      ‖(1 / D) • ((1 + 2 * c * p + c * B) • a + (1 - c * A) • b)‖ ^ 2
        = ‖(1 + 2 * c * p + c * B) • a + (1 - c * A) • b‖ ^ 2 / D ^ 2 := by
    rw [norm_smul, norm_div, norm_one, Real.norm_of_nonneg (le_of_lt hDpos)]
    ring
  -- (2) expand ‖N‖² through inner-product linearity
  have hN_expand :
      ‖(1 + 2 * c * p + c * B) • a + (1 - c * A) • b‖ ^ 2
        = (1 + 2 * c * p + c * B) ^ 2 * A
          + 2 * (1 + 2 * c * p + c * B) * (1 - c * A) * p + (1 - c * A) ^ 2 * B := by
    rw [norm_add_sq_real]
    rw [norm_smul, norm_smul, hA, hB]
    rw [inner_smul_left a ((1 - c * A) • b) (1 + 2 * c * p + c * B),
        inner_smul_right a b (1 - c * A)]
    have h1 : (starRingEnd ℝ) (1 + 2 * c * p + c * B) = 1 + 2 * c * p + c * B := rfl
    rw [h1]
    rw [Real.norm_eq_abs, Real.norm_eq_abs]
    rw [mul_pow, mul_pow, sq_abs, sq_abs, ← hp]
    ring
  -- (3) the algebraic identity: ‖N‖² = (A + B + 2p)·D
  have halg :
      (1 + 2 * c * p + c * B) ^ 2 * A
          + 2 * (1 + 2 * c * p + c * B) * (1 - c * A) * p + (1 - c * A) ^ 2 * B
        = (A + B + 2 * p) * D := by
    unfold D
    ring
  -- finish
  rw [hN, hnorm_div, hN_expand, halg]
  field_simp [hDpos.ne']

/-- Gyration preserves the ball: every argument of the gyration loop stays
    inside the ball, so gyr_c[u,v] maps the ball to itself.

    Proof: gyr[u,v]w = (−(u⊕v)) ⊕ (u ⊕ (v⊕w)). Each ⊕ argument is a Möbius
    sum of in-ball points, and Möbius addition preserves the ball
    (mobius_add_nd_preserves_ball), so each intermediate result is in-ball,
    and the outer addition preserves the ball again.
-/
theorem gyration_nd_preserves_ball {n : ℕ} (c : ℝ) (hc : c > 0)
    (u v w : EuclideanSpace ℝ (Fin n))
    (hu : ‖u‖ ^ 2 < 1 / c) (hv : ‖v‖ ^ 2 < 1 / c) (hw : ‖w‖ ^ 2 < 1 / c) :
    ‖gyration_nd c u v w‖ ^ 2 < 1 / c := by
  unfold gyration_nd
  apply mobius_add_nd_preserves_ball c hc
  · simpa using mobius_add_nd_preserves_ball c hc u v hu hv
  · apply mobius_add_nd_preserves_ball c hc u _
    · exact hu
    · exact mobius_add_nd_preserves_ball c hc v w hv hw

/-- The Möbius gyration in explicit closed form (Ungar, Analytic Hyperbolic
    Geometry 2005, Def 3.31; also implemented directly in hyperbolic DL
    code such as TAHyper-SDM24):

    gyr_cf[u,v]w = w + (2/d)·(a·u + b·v)

    a = −c²⟨u,w⟩‖v‖² + c⟨v,w⟩ + 2c²⟨u,v⟩⟨v,w⟩
    b = −c²⟨v,w⟩‖u‖² − c⟨u,w⟩
    d = 1 + 2c⟨u,v⟩ + c²‖u‖²‖v‖²

This closed form equals the loop definition `gyration_nd` by direct
algebraic expansion (rational-function identity); that equivalence is left
as future work. Everything below is proven for the closed form, which is
the form actually used in implementations.
-/
noncomputable def gyr_cf {n : ℕ} (c : ℝ) (u v w : EuclideanSpace ℝ (Fin n)) :
    EuclideanSpace ℝ (Fin n) :=
  w + (2 / (1 + 2 * c * inner ℝ u v + c ^ 2 * ‖u‖ ^ 2 * ‖v‖ ^ 2)) •
      ((-c ^ 2 * inner ℝ u w * ‖v‖ ^ 2 + c * inner ℝ v w
          + 2 * c ^ 2 * inner ℝ u v * inner ℝ v w) • u
       + (-c ^ 2 * inner ℝ v w * ‖u‖ ^ 2 - c * inner ℝ u w) • v)

/-- The load-bearing scalar identity behind gyration norm-preservation:

    d·(a·p + b·q) + 2·(a²A + 2abr + b²B) = 0

where p = ⟨u,w⟩, q = ⟨v,w⟩, r = ⟨u,v⟩, A = ‖u‖², B = ‖v‖², and a, b, d are
the gyration coefficients. This is exactly what makes the two correction
terms in ‖w + (2/d)·(a·u + b·v)‖² cancel, leaving ‖w‖².
-/
theorem gyr_scalar_identity (c p q A B r a b d : ℝ)
    (ha : a = -c ^ 2 * p * B + c * q + 2 * c ^ 2 * r * q)
    (hb : b = -c ^ 2 * q * A - c * p)
    (hd : d = 1 + 2 * c * r + c ^ 2 * A * B) :
    d * (a * p + b * q) + (a ^ 2 * A + 2 * a * b * r + b ^ 2 * B) = 0 := by
  rw [ha, hb, hd]
  ring

/-- Gyration preserves the norm — the key theorem behind "gyration is a
    rotation" (Ungar 2005, eq. 1.33). For the closed-form gyration with
    d ≠ 0:

        ‖gyr_cf[u,v]w‖² = ‖w‖²

    Proof: write gyr = w + (2/d)·(a·u + b·v). Expanding
    ‖w + (2/d)·N‖² = ‖w‖² + 2·(2/d)·⟨w,N⟩ + (2/d)²·‖N‖² via
    `norm_add_sq_real`, the cross term is (2/d)(ap+bq) and the norm term is
    (4/d²)(a²A + 2abr + b²B); their sum is
    (2/d²)·[d(ap+bq) + 2(a²A + 2abr + b²B)] = 0 by
    `gyr_scalar_identity`. Hence the norm is unchanged.
-/
theorem gyr_cf_norm_preserving {n : ℕ} (c : ℝ)
    (u v w : EuclideanSpace ℝ (Fin n))
    (hd : 1 + 2 * c * inner ℝ u v + c ^ 2 * ‖u‖ ^ 2 * ‖v‖ ^ 2 ≠ 0) :
    ‖gyr_cf c u v w‖ ^ 2 = ‖w‖ ^ 2 := by
  -- name the pieces
  set p := inner ℝ u w with hp
  set q := inner ℝ v w with hq
  set r := inner ℝ u v with hr
  set A := ‖u‖ ^ 2 with hA
  set B := ‖v‖ ^ 2 with hB
  set W := ‖w‖ ^ 2 with hW
  set d := 1 + 2 * c * r + c ^ 2 * A * B with hd_def
  set a := -c ^ 2 * p * B + c * q + 2 * c ^ 2 * r * q with ha_def
  set b := -c ^ 2 * q * A - c * p with hb_def
  -- the definition unfolds to w + (2/d)•(a•u + b•v)
  have hgyr : gyr_cf c u v w = w + (2 / d) • (a • u + b • v) := by
    unfold gyr_cf
    rw [hd_def, ha_def, hb_def]
  -- inner products needed for the expansion
  have hwp : inner ℝ w u = p := by rw [real_inner_comm u w]
  have hwq : inner ℝ w v = q := by rw [real_inner_comm v w]
  have hBv : inner ℝ v v = B := (real_inner_self_eq_norm_sq v).trans hB.symm
  -- cross term: ⟨w, (2/d)•(a•u + b•v)⟩ = (2/d)·(a·p + b·q)
  have hcross : inner ℝ w ((2 / d) • (a • u + b • v)) = (2 / d) * (a * p + b * q) := by
    rw [inner_smul_right w (a • u + b • v) (2 / d)]
    rw [inner_add_right, inner_smul_right, inner_smul_right, hwp, hwq]
  -- norm term: ‖(2/d)•(a•u + b•v)‖² = (2/d)²·(a²A + 2ab·r + b²B)
  have hnnorm : ‖(2 / d) • (a • u + b • v)‖ ^ 2
      = (2 / d) ^ 2 * (a ^ 2 * A + 2 * a * b * r + b ^ 2 * B) := by
    rw [norm_smul, mul_pow, norm_add_sq_real]
    rw [norm_smul, norm_smul, hA, hB]
    rw [inner_smul_left u (b • v) a, inner_smul_right u v b]
    have hs : (starRingEnd ℝ) a = a := rfl
    rw [hs]
    simp only [Real.norm_eq_abs, sq_abs, mul_pow, hr]
    ring
  -- the scalar cancellation identity (instantiate gyr_scalar_identity)
  have hcancel : d * (a * p + b * q) + (a ^ 2 * A + 2 * a * b * r + b ^ 2 * B) = 0 := by
    exact gyr_scalar_identity c p q A B r a b d ha_def hb_def hd_def
  -- finish
  rw [hgyr, norm_add_sq_real, hcross, hnnorm]
  field_simp [hd]
  nlinarith [hcancel, mul_self_nonneg d]

/-- The 4-coefficient scalar identity behind inner-product preservation.

    With w1, w2 carrying independent coefficient sets (a1,b1) and (a2,b2):

        d·[(a2·p1 + b2·q1) + (a1·p2 + b1·q2)]
          + 2·(a1·a2·A + (a1·b2 + a2·b1)·r + b1·b2·B) = 0

    where p1 = ⟨u,w1⟩, q1 = ⟨v,w1⟩, p2 = ⟨u,w2⟩, q2 = ⟨v,w2⟩,
    r = ⟨u,v⟩, A = ‖u‖², B = ‖v‖². This is what makes the four
    correction terms in ⟨gyr w1, gyr w2⟩ cancel.
-/
theorem gyr_inner_scalar_identity (c p1 q1 p2 q2 A B r a1 b1 a2 b2 d : ℝ)
    (ha1 : a1 = -c ^ 2 * p1 * B + c * q1 + 2 * c ^ 2 * r * q1)
    (hb1 : b1 = -c ^ 2 * q1 * A - c * p1)
    (ha2 : a2 = -c ^ 2 * p2 * B + c * q2 + 2 * c ^ 2 * r * q2)
    (hb2 : b2 = -c ^ 2 * q2 * A - c * p2)
    (hd : d = 1 + 2 * c * r + c ^ 2 * A * B) :
    d * ((a2 * p1 + b2 * q1) + (a1 * p2 + b1 * q2))
      + 2 * (a1 * a2 * A + (a1 * b2 + a2 * b1) * r + b1 * b2 * B) = 0 := by
  rw [ha1, hb1, ha2, hb2, hd]
  ring

/-- Gyration preserves the inner product — the full form of "gyration is a
    rotation" (Ungar 2005, eq. 1.32). For the closed-form gyration with
    d ≠ 0:

        ⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a, b⟩

    Proof: write gyr w = w + (2/d)·(a_w·u + b_w·v). Expanding
    ⟨gyr a, gyr b⟩ through inner-product bilinearity gives ⟨a,b⟩ plus four
    correction terms (two cross, one norm-of-sum), whose sum is
    (2/d²)·[d·(cross) + 2·(normterm)] = 0 by `gyr_inner_scalar_identity`.
    Hence the inner product is unchanged.

    The full two-vector expansion (with independent coefficient sets for a
    and b) is the heaviest step; the load-bearing scalar identity
    `gyr_inner_scalar_identity` is proven above. The theorem here proves
    the a = b diagonal case directly from norm preservation, and states
    the general case as the reduction target.
-/
theorem gyr_cf_inner_preserving_diag {n : ℕ} (c : ℝ)
    (u v w : EuclideanSpace ℝ (Fin n))
    (hd : 1 + 2 * c * inner ℝ u v + c ^ 2 * ‖u‖ ^ 2 * ‖v‖ ^ 2 ≠ 0) :
    inner ℝ (gyr_cf c u v w) (gyr_cf c u v w) = inner ℝ w w := by
  simpa [real_inner_self_eq_norm_sq] using gyr_cf_norm_preserving c u v w hd

/-- Gyration preserves the inner product — the FULL form of "gyration is a
    rotation" (Ungar 2005, eq. 1.32). For the closed-form gyration with
    d ≠ 0:

        ⟨gyr[u,v]w1, gyr[u,v]w2⟩ = ⟨w1, w2⟩

    Proof (mirrors `gyr_cf_norm_preserving`): write
    gyr_cf c u v w = w + (2/d)·(a_w·u + b_w·v) with the SAME d (depends only
    on u,v) but INDEPENDENT coefficient sets (a1,b1) for w1 and (a2,b2) for
    w2. Expanding ⟨w1 + tN1, w2 + tN2⟩ (t = 2/d) through inner-product
    bilinearity:

        ⟨w1,w2⟩ + t·(⟨w1,N2⟩ + ⟨N1,w2⟩) + t²·⟨N1,N2⟩

    The two cross terms give (2/d)·[(a2·p1 + b2·q1) + (a1·p2 + b1·q2)] and
    the norm term gives (4/d²)·(a1·a2·A + (a1·b2 + a2·b1)·r + b1·b2·B).
    Their sum is (2/d²)·[d·(cross) + 2·(normterm)] = 0 by
    `gyr_inner_scalar_identity`. Hence the inner product is unchanged.
-/
theorem gyr_cf_inner_preserving {n : ℕ} (c : ℝ)
    (u v w1 w2 : EuclideanSpace ℝ (Fin n))
    (hd : 1 + 2 * c * inner ℝ u v + c ^ 2 * ‖u‖ ^ 2 * ‖v‖ ^ 2 ≠ 0) :
    inner ℝ (gyr_cf c u v w1) (gyr_cf c u v w2) = inner ℝ w1 w2 := by
  -- shared scalar pieces (depend only on u, v)
  set r := inner ℝ u v with hr
  set A := ‖u‖ ^ 2 with hA
  set B := ‖v‖ ^ 2 with hB
  set d := 1 + 2 * c * r + c ^ 2 * A * B with hd_def
  -- independent coefficient sets for w1 and w2
  set p1 := inner ℝ u w1 with hp1
  set q1 := inner ℝ v w1 with hq1
  set p2 := inner ℝ u w2 with hp2
  set q2 := inner ℝ v w2 with hq2
  set a1 := -c ^ 2 * p1 * B + c * q1 + 2 * c ^ 2 * r * q1 with ha1
  set b1 := -c ^ 2 * q1 * A - c * p1 with hb1
  set a2 := -c ^ 2 * p2 * B + c * q2 + 2 * c ^ 2 * r * q2 with ha2
  set b2 := -c ^ 2 * q2 * A - c * p2 with hb2
  -- both gyrations unfold to w + (2/d)•(a•u + b•v) with their own coeffs
  have hgyr1 : gyr_cf c u v w1 = w1 + (2 / d) • (a1 • u + b1 • v) := by
    unfold gyr_cf; rw [hd_def, ha1, hb1]
  have hgyr2 : gyr_cf c u v w2 = w2 + (2 / d) • (a2 • u + b2 • v) := by
    unfold gyr_cf; rw [hd_def, ha2, hb2]
  -- inner products with the generators (both directions)
  have hw1u : inner ℝ w1 u = p1 := by rw [real_inner_comm u w1]
  have hw1v : inner ℝ w1 v = q1 := by rw [real_inner_comm v w1]
  have huw2 : inner ℝ u w2 = p2 := rfl
  have hvw2 : inner ℝ v w2 = q2 := rfl
  have huu : inner ℝ u u = A := (real_inner_self_eq_norm_sq u).trans hA.symm
  have hvv : inner ℝ v v = B := (real_inner_self_eq_norm_sq v).trans hB.symm
  -- cross term 1: ⟨w1, (2/d)•(a2·u + b2·v)⟩ = (2/d)·(a2·p1 + b2·q1)
  have hcross1 : inner ℝ w1 ((2 / d) • (a2 • u + b2 • v)) = (2 / d) * (a2 * p1 + b2 * q1) := by
    rw [inner_smul_right w1 (a2 • u + b2 • v) (2 / d)]
    rw [inner_add_right, inner_smul_right, inner_smul_right, hw1u, hw1v]
  -- cross term 2: ⟨(2/d)•(a1·u + b1·v), w2⟩ = (2/d)·(a1·p2 + b1·q2)
  have hcross2 : inner ℝ ((2 / d) • (a1 • u + b1 • v)) w2 = (2 / d) * (a1 * p2 + b1 * q2) := by
    rw [inner_smul_left (a1 • u + b1 • v) w2 (2 / d)]
    have hs : (starRingEnd ℝ) (2 / d) = 2 / d := rfl
    have hsa1 : (starRingEnd ℝ) a1 = a1 := rfl
    have hsb1 : (starRingEnd ℝ) b1 = b1 := rfl
    rw [hs, inner_add_left, inner_smul_left, inner_smul_left, huw2, hvw2, hsa1, hsb1]
  -- norm term: ⟨(2/d)•N1, (2/d)•N2⟩ = (2/d)²·(a1·a2·A + (a1·b2+a2·b1)·r + b1·b2·B)
  have hnnorm : inner ℝ ((2 / d) • (a1 • u + b1 • v)) ((2 / d) • (a2 • u + b2 • v))
      = (2 / d) ^ 2 * (a1 * a2 * A + (a1 * b2 + a2 * b1) * r + b1 * b2 * B) := by
    rw [inner_smul_left (a1 • u + b1 • v) ((2 / d) • (a2 • u + b2 • v)) (2 / d),
        inner_smul_right (a1 • u + b1 • v) (a2 • u + b2 • v) (2 / d)]
    have hs : (starRingEnd ℝ) (2 / d) = 2 / d := rfl
    have hsa1 : (starRingEnd ℝ) a1 = a1 := rfl
    have hsb1 : (starRingEnd ℝ) b1 = b1 := rfl
    have hsa2 : (starRingEnd ℝ) a2 = a2 := rfl
    have hsb2 : (starRingEnd ℝ) b2 = b2 := rfl
    rw [hs]
    simp only [inner_add_left, inner_add_right, inner_smul_left, inner_smul_right,
      huu, hvv, hr, huw2, hvw2, hsa1, hsb1, hsa2, hsb2, real_inner_comm v u]
    ring
  -- the scalar cancellation (instantiate gyr_inner_scalar_identity)
  have hcancel :
      d * ((a2 * p1 + b2 * q1) + (a1 * p2 + b1 * q2))
        + 2 * (a1 * a2 * A + (a1 * b2 + a2 * b1) * r + b1 * b2 * B) = 0 := by
    exact gyr_inner_scalar_identity c p1 q1 p2 q2 A B r a1 b1 a2 b2 d
      ha1 hb1 ha2 hb2 hd_def
  -- finish: split ⟨w1+tN1, w2+tN2⟩ = ⟨w1,w2⟩ + ⟨w1,tN2⟩ + ⟨tN1,w2⟩ + ⟨tN1,tN2⟩
  -- via inner_add_right then inner_add_left (each fires once on the current
  -- outermost sum; the smul terms stay intact so hcross/hnnorm match).
  rw [hgyr1, hgyr2, inner_add_right, inner_add_left, inner_add_left]
  rw [hcross1, hcross2, hnnorm]
  field_simp [hd]
  nlinarith [hcancel, mul_self_nonneg d]
