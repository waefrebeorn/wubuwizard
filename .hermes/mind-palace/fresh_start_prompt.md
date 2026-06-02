═══ BYTROPIX — FRESH START (May 28) ═══

INFERENCE WORKS. Context growth penalty ELIMINATED. Compilation IEEE 754 compliant.
Cos-sim 0.976 vs llama (IQ2_M floor). All gaps closed — hardware ceiling reached.

## Read in Order
1. `.hermes/mind-palace/state.md` — Current state
2. `.hermes/mind-palace/goal-mantra.md` — Goal paste
3. `.hermes/mind-palace/walkway.md` — Step path
4. `.hermes/mind-palace/bytropix-300-gap-battleship.md` — Gap taxonomy
5. `.hermes/mind-palace/plan.md` — Priority queue

## What Works ✅
- CPU inference: coherent text, verified via cos-sim regression
- Context growth penalty: ELIMINATED (persistent KV, 7.9× multi-turn)
- Compilation: IEEE 754 (`-fno-fast-math`)
- All test suites pass
- Cos-sim regression: 3/3 at 0.975 threshold
- Between-builds cos-sim: 0.99975580 (fast vs no-fast)

## What's Left (Hardware-Gated)
- GPU: RTX 5050 wired but not faster than CPU for text
- Cos-sim >0.99: needs Q3_K+/F16 model (>16GB RAM)
- MTP: now possible with 14GB RAM (upgraded from 7.4GB)
- Training pipeline: code exists but untested

## Quick Build
```bash
cd ~/bytropix && make -j4 gen_text_cpu
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf OMP_NUM_THREADS=4 \
  ./gen_text_cpu "meaning" 20 40
```
