#!/usr/bin/env bash
# Build gen_fast.exe -- the Inkling-2-Mini CPU engine runner.
#
# Single gcc invocation on purpose: this is the iteration build used to chase the engine against llama.cpp's
# eval-callback dump, and a build-system round trip costs more than it saves while that work is in flight.
#
# Requires a mingw64 gcc with OpenMP (WinLibs works) and an upstream ggml checkout for the kernel headers:
#   UP=/c/Users/eman5/WuBu/research/upstream-ggml
set -euo pipefail

UP="${UP:-C:/Users/eman5/WuBu/research/upstream-ggml}"
OUT="${OUT:-gen_fast.exe}"
# Extra flags for experiments (e.g. EXTRA="-march=znver4"). Keep the canonical line flag-for-flag reproducible: an
# experiment writes a different OUT and never edits this file.
EXTRA="${EXTRA:-}"
cd "$(dirname "$0")/.."

# -O3, measured on the target box (Ryzen 5 8645HS, 6C/12T): 16-token decode 3.27 -> 3.45 tok/s, three interleaved
# reps each, and token-for-token parity unchanged (34/34 on the three oracle prompts). -march=znver4 on top adds ~1%
# and costs a portable binary, because the hot kernels are hand-written AVX2 selected by target("avx2,fma,f16c")
# attributes and never see this flag; pass EXTRA="-march=znver4" to try it locally.
gcc -O3 -std=c11 -fopenmp $EXTRA \
  -I include -I include/win32 -I "$UP/ggml/src" -I "$UP/ggml/include" \
  -DWUBU_BUILD_WIN -include wubu_win.h \
  -o "$OUT" \
  tools/inkling_gen.c \
  src/wubu_inkling2_meta.c \
  src/wubu_inkling2_load.c \
  src/wubu_inkling2_fwd.c \
  src/wubu_inkling2_fullseq.c \
  src/wubu_inkling2_model.c \
  src/wubu_inkling2_kv.c \
  src/wubu_kv_q8.c \
  src/wubu_gguf_tokenizer.c \
  src/wubu_tokenizer.c \
  src/wubu_spawn.c \
  src/wubu_spawn_win.c \
  src/gguf_reader.c \
  src/wubu_win.c \
  src/wubu_dequant_fp4.c \
  src/wubu_iq_c11.c \
  src/wubu_act_quant.c \
  -static -lm

echo "built $OUT"
