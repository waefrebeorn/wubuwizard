#!/bin/bash
# test-iq1-m.sh — Test IQ1_M quantization support
# 
# IQ1_M (1.9 bpw) is not yet supported in bytropix.
# This script documents what would be needed and skips gracefully.
#
# Prerequisites for IQ1_M:
# 1. quantized IQ1_M dot product kernel in quantized_dot_generic.c
# 2. IQ1_M dequant in dequant_iq2_xxs.c (or separate file)
# 3. GGUF type enum entry in gguf_reader.h
# 4. Loading support in gguf_reader.c
# 5. A model quantized to IQ1_M by llama.cpp quantize tool
#
# Expected model sizes (11B params):
#   IQ1_M:  ~7.0 GB  (1.9 bpw)
#   IQ2_XXS:~7.2 GB  (2.0625 bpw)  ← partially supported
#   IQ2_M:  ~7.9 GB  (2.125 bpw)   ← current default
#   Q3_K_M: ~12.6 GB (3.5 bpw)     ← too large for 16GB RAM
#
# Since IQ1_M reduces model size by only ~0.9 GB vs IQ2_M but
# significantly degrades quality (cos-sim likely < 0.95), this is
# low priority. Focus should be on Q3_K+ support via hardware upgrade.

echo "=== IQ1_M Support Test ==="
echo ""
echo "IQ1_M quantization: NOT IMPLEMENTED"
echo "Priority: LOW (quality loss > memory savings)"
echo ""
echo "Current memory usage:"
free -h | head -2
echo ""
echo "Current quantized dot product types:"
ls -la src/quantized_dot_generic.c src/dequant_iq2_xxs.c 2>/dev/null
echo ""
echo "To add IQ1_M support, start with:"
echo "  1. Add GGML_TYPE_IQ1_M to ggml_type enum in include/gguf_reader.h"
echo "  2. Add iq1_m dot product kernel to src/quantized_dot_generic.c"
echo "  3. Add iq1_m dequant to src/dequant_iq1_m.c"
echo "  4. Wire loading in src/gguf_reader.c"
echo ""
echo "SKIP: IQ1_M not needed on current hardware (11GB RAM OK for IQ2_M)"
exit 0
