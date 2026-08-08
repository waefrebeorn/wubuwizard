# bytropix — Entry Point (Jun 2, 2026)

**Inference WORKS.** Context growth penalty ELIMINATED. Compilation IEEE 754.
See `.hermes/mind-palace/state.md` for status.

## Hardware
- CPU: AMD Ryzen 7 7445HS (12 cores, 24 threads)
- RAM: 14GB DDR5 (14336MB WSL2)
- Platform: WSL2
- GPU: None available yet; fusing CPU + GPU paths remains the goal (`bytropix-gpu-acceleration`).

## Build
```bash
make gen_text_cpu -j4                     # CPU inference binary
make clean && make gen_text_cpu            # Full rebuild
```

## Run Inference
```bash
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  ./gen_text_cpu "The capital of France is" 20 40

# ChatML mode
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 CHAT=1 \
  ./gen_text_cpu "Write a paragraph about cats." 64 40
```

## Run Server (Persistent KV)
```bash
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  python3 tools/serve_local.py --port 8001 --persist
```

## Run Tests
```bash
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  THRESHOLD=0.975 \
  bash tools/test-cos-sim-regression.sh    # 3 prompts
bash tools/test-512k-suite.sh              # 6 tests
bash tools/test-hermes-integration.sh      # 9 tests
```

## File Layout
```
src/          — Core: ssm, moe, model, gguf_reader, cuda_kernels, vision
include/      — Headers
tools/        — gen_text.c, serve_local.py, test_*, dump_*
data/         — embeddings, tokenizer
.hermes/      — Mind palace, vault
vault/        — Context growth penalty analysis, legacy docs
/models/      — GGUF model files
```

## Key Docs
- `.hermes/mind-palace/goal-mantra.md` — Goal paste
- `.hermes/mind-palace/state.md` — Current state
- `.hermes/mind-palace/plan.md` — Priority queue
- `.hermes/mind-palace/walkway.md` — Step path
- `.hermes/mind-palace/bytropix-300-gap-battleship.md` — Gap taxonomy
- `README.md` — Project overview
