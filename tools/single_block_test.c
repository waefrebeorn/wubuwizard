/**
 * single_block_test.c — Test ONE Q5_K block in detail.
 * Compares intermediate values (d, dmin, scales, high bits) between
 * our dequant and ggml's dequant.
 *
 * Build:
 *   g++ -std=c++11 -O2 -I/home/wubu/llama.cpp/ggml/include \
 *       -o single_block_test tools/single_block_test.c src/gguf_reader.o \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lggml-base -lggml-cpu -lm -lstdc++
 *
 * Usage:
 *   ./single_block_test model.gguf file_offset block_index
 *   (block_index = which 256-element block to examine, default 0)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_fp16.h"

extern "C" {
#include "ggml.h"
}

extern "C" void gguf_dequantize(const uint8_t *data, int ggml_type, int64_t n_elems, float *output);

// Our static dequant functions — we call gguf_dequantize instead
// but for single-block detail, we inline our logic

/* f16_to_f32: use wubu_fp16.h */

#define Q5_K_BLOCK_SIZE 176
#define QK_K 256

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf file_offset [block_index]\n", argv[0]);
        return 1;
    }
    const char *fname = argv[1];
    long file_offset = atol(argv[2]);
    int target_block = (argc > 3) ? atoi(argv[3]) : 0;

    // Read single block
    uint8_t raw_block[Q5_K_BLOCK_SIZE];
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    fseek(f, file_offset + target_block * Q5_K_BLOCK_SIZE, SEEK_SET);
    size_t nr = fread(raw_block, 1, Q5_K_BLOCK_SIZE, f);
    fclose(f);
    if (nr != Q5_K_BLOCK_SIZE) {
        fprintf(stderr, "Read %zu bytes, expected %d\n", nr, Q5_K_BLOCK_SIZE);
        return 1;
    }

    printf("=== Block %d at file offset %ld ===\n", target_block, 
           file_offset + target_block * Q5_K_BLOCK_SIZE);

    // Print first 16 raw bytes
    printf("Raw bytes [0:16]: ");
    for (int i = 0; i < 16; i++) printf("%02x ", raw_block[i]);
    printf("\n");

    // Extract d and dmin
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, raw_block, 2);
    memcpy(&dmin_bits, raw_block + 2, 2);
    float d_our = f16_to_f32(d_bits);
    float dmin_our = f16_to_f32(dmin_bits);
    
    printf("d:      %.10f  bits=0x%04x\n", d_our, d_bits);
    printf("dmin:   %.10f  bits=0x%04x\n", dmin_our, dmin_bits);

    // Print scales
    const uint8_t *scales = raw_block + 4;
    printf("scales[12]: ");
    for (int i = 0; i < 12; i++) printf("%02x ", scales[i]);
    printf("\n");

    // Print all 8 scale pairs from get_scale_min_k4
    printf("\nScale pairs (get_scale_min_k4):\n");
    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        get_scale_min_k4(j, scales, &sc, &m);
        float d1 = d_our * sc;
        float m1 = dmin_our * m;
        printf("  pair %d: sc=%d m=%d  d1=%.6f m1=%.6f\n",
               j, sc, m, d1, m1);
    }

    // Check high bits in qh
    const uint8_t *qh = raw_block + 16;
    printf("\nqh[32]: ");
    for (int i = 0; i < 32; i++) printf("%02x ", qh[i]);
    printf("\n");

    // Dequant with our gguf_dequantize
    float our_256[256];
    float ref_256[256];
    
    gguf_dequantize(raw_block, 13, 256, our_256);
    
    const struct ggml_type_traits *traits = ggml_get_type_traits((enum ggml_type)13);
    traits->to_float(raw_block, ref_256, 256);

    // Compare element by element
    printf("\nFirst 32 elements comparison:\n");
    printf("idx  our_val       ref_val       diff         our_nibble our_high ref_high\n");
    for (int i = 0; i < 32; i++) {
        float diff = fabsf(our_256[i] - ref_256[i]);
        // Determine nibble+high values
        int ql_idx = 0; // first group
        uint8_t lo = raw_block[48 + ql_idx + i/2];
        // Actually let's just print
        printf("%3d  %+.10f  %+.10f  %.10f%s\n",
               i, our_256[i], ref_256[i], diff,
               diff > 1e-5 ? " ***" : "");
    }

    // Compare ALL 256 elements
    int mismatch_count = 0;
    for (int i = 0; i < 256; i++) {
        if (fabsf(our_256[i] - ref_256[i]) > 1e-5f) mismatch_count++;
    }
    printf("\nTotal mismatches in block: %d/256\n", mismatch_count);

    // Find which scale pairs have mismatches
    printf("\nMismatches by 32-element group:\n");
    for (int g = 0; g < 8; g++) {
        int g_mismatch = 0;
        float our_range_min = 1e30, our_range_max = -1e30;
        float ref_range_min = 1e30, ref_range_max = -1e30;
        for (int i = 0; i < 32; i++) {
            int idx = g * 32 + i;
            if (fabsf(our_256[idx] - ref_256[idx]) > 1e-5f) g_mismatch++;
            if (our_256[idx] < our_range_min) our_range_min = our_256[idx];
            if (our_256[idx] > our_range_max) our_range_max = our_256[idx];
            if (ref_256[idx] < ref_range_min) ref_range_min = ref_256[idx];
            if (ref_256[idx] > ref_range_max) ref_range_max = ref_256[idx];
        }
        printf("  group %d (elems %d-%d): %d/32 mismatches  our[%.4f,%.4f] ref[%.4f,%.4f]\n",
               g, g*32, g*32+31, g_mismatch, our_range_min, our_range_max, ref_range_min, ref_range_max);
    }

    return mismatch_count > 0 ? 1 : 0;
}
