/-
Main.lean — Executable entry point for wubu_proofs

Prints a status summary of all proofs and their current state.
-/

import WubuProofs

def main : IO Unit := do
  IO.println "=" * 60
  IO.println "  Wubu Nesting Formal Proofs — Status"
  IO.println "  Lean 4.29.1 + mathlib4"
  IO.println "=" * 60
  IO.println ""
  IO.println "  📐 Proof 1: PoincaréBall.lean"
  IO.println "     Theorem: exp_0^c(log_0^c(y)) = y"
  IO.println "     Status: Partial (core identity using Real.tanh_arctanh)"
  IO.println ""
  IO.println "  📐 Proof 2: MobiusAdd.lean"
  IO.println "     Theorem: Möbius addition preserves the ball"
  IO.println "     Status: 1D case proven (nlinarith). nD case proven in MobiusAddND.lean."
  IO.println ""
  IO.println "  📐 Proof 2b: MobiusAddND.lean"
  IO.println "     Theorem: Möbius addition preserves the ball (n dimensions)"
  IO.println "     Status: PROVEN via (1−ca)(1−cb)·[(1+cp)² + c²(ab−p²)] factorization"
  IO.println ""
  IO.println "  📐 Proof 3: MLACompression.lean"
  IO.println "     Theorem: MLA KV compression error bound"
  IO.println "     Status: Sketch (requires SVD/Eckart-Young from mathlib)"
  IO.println ""
  IO.println "  📐 Proof 4: HyperbolicGyration.lean"
  IO.println "     Theorem: gyration preserves the ball"
  IO.println "     Status: Sketch (1D identity, nD SO(n) argument)"
  IO.println ""
  IO.println "  Build: lake build"
  IO.println "  Verify: lake build wubu_proofs"
