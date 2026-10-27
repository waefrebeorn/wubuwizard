-- Probe: I1/I2 with LITERAL denominator (no d atom) — field_simp multiplies through, then ring
import Mathlib
open Real
set_option maxHeartbeats 0

theorem probe_I1_I2_lit (c p a b p2 bz : ℝ)
    (hD1pos : 1 - 2*c*p + c^2*a*b > 0)
    (hp2m_lit : p2 * (1 - 2*c*p + c^2*a*b) = -(1 - 2*c*p + c*b)*a + (1 - c*a)*p)
    (hbzm_lit : bz * (1 - 2*c*p + c^2*a*b) = ((1 - 2*c*p + c*b)^2*a - 2*(1 - 2*c*p + c*b)*(1 - c*a)*p + (1 - c*a)^2*b) / (1 - 2*c*p + c^2*a*b)) :
    (1 + 2*c*p2 + c*bz) * (1 - 2*c*p + c^2*a*b) = (1 - c*a) * (1 - 2*c*p + c*b) ∧
    (1 + 2*c*p2 + c^2*a*bz) * (1 - 2*c*p + c^2*a*b) = (1 - c*a)^2 := by
  constructor
  · -- I1
    rw [add_mul, add_mul, one_mul]
    nth_rewrite 1 [mul_assoc (2*c) p2 (1 - 2*c*p + c^2*a*b)]
    nth_rewrite 1 [hp2m_lit]
    nth_rewrite 1 [mul_assoc (c) bz (1 - 2*c*p + c^2*a*b)]
    nth_rewrite 1 [hbzm_lit]
    field_simp [hD1pos.ne']
    grobner
  · -- I2
    rw [add_mul, add_mul, one_mul]
    nth_rewrite 1 [mul_assoc (2*c) p2 (1 - 2*c*p + c^2*a*b)]
    nth_rewrite 1 [hp2m_lit]
    nth_rewrite 1 [mul_assoc (c^2*a) bz (1 - 2*c*p + c^2*a*b)]
    nth_rewrite 1 [hbzm_lit]
    field_simp [hD1pos.ne']
    grobner
