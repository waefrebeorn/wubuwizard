/-
WubuProofs.lean — Entry point for the Wubu Nested Hyperbolic Proofs library.

This library contains Lean 4 formal proofs of the core mathematical
claims underpinning the WuBu Nesting framework:

  1. PoincaréBall.lean — exp_0^c(log_0^c(y)) = y identity
  2. MobiusAdd.lean — Möbius addition preserves the Poincaré ball
  3. MLACompression.lean — Low-rank KV compression error bound
  4. HyperbolicGyration.lean — gyration preserves the unit ball
-/

import WubuProofs.PoincareBall
import WubuProofs.MobiusAdd
import WubuProofs.MobiusAddND
import WubuProofs.MLACompression
import WubuProofs.HyperbolicGyration
import WubuProofs.GyrationND
import WubuProofs.KleinDeck
import WubuProofs.QuaternionRotation
import WubuProofs.FiberBundle
import WubuProofs.NestedHyperbolicSpaces
import WubuProofs.HolographicOptimizer
-- PowerTower.lean NOT imported: it was written against a different mathlib
-- version and does not compile on 4.29.1 (Real.rpow_lt_rpow_of_exponent_lt
-- argument order + type mismatches at lines 68,91,92,96,97,101,111,112).
-- Left in the tree for a future mathlib-port; see VAULT/github-agent-assessment.md.
