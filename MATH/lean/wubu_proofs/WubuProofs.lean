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
