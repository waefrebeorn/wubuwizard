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
cd "$(dirname "$0")/.."

gcc -O2 -std=c11 -fopenmp \
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
