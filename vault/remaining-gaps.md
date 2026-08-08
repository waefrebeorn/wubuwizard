# vault/remaining-gaps.md — May 27, 2026

## ALL GAPS RESOLVED ✅

| Gap | Status | Notes |
|-----|--------|-------|
| `dump_ref` runtime error | ✅ FIXED | `llama_model_free` → `llama_model_free` API fix |
| `run-harness.sh` proxy | ✅ PATCHED | Now uses `serve_local.py` (local CPU) |
| NES PPU tile/nametable | ✅ DONE | Already had full `render_scanline` + palette + iNES loader |
| `test-hermes-headless.sh` proxy | ✅ PATCHED | Now uses `serve_local.py` with MODEL + OMP |

### Current state

**Parity:** Cos-sim 0.974 vs llama.cpp reference — IQ2_M quantization floor. Cannot reach >0.99 without higher-precision model (Q3_K/Q4_K/F16 not available on this machine).

**NES emulator:** Fully functional — builds clean, has 6502 CPU, PPU with tile/nametable rendering + palette, iNES ROM loader (mapper 0), self-play AI, ANSI ASCII output.

**Test scripts:**
- `test-hermes-integration.sh` — uses `serve_local.py` ✅
- `test-hermes-headless.sh` — uses `serve_local.py` ✅ (just patched)
- `test-512k-suite.sh` — tests KV cache, attention, memory, RoPE, NES build

### Next possible work (no urgent gaps)
- Run NES emulator with a SMB ROM for 512K context test
- Profile CPU-inference hotspots and optimize
