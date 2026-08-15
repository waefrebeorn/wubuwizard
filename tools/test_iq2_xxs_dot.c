/**
 * test_iq2_xxs_dot.c — Compare CPU vs GPU IQ2_XXS dot products for a single block.
 * Build with GPU support.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_fp16.h"

/* f16_to_f32: use wubu_fp16.h */

static const uint64_t iq2xxs_grid[256] = {
    #include "iq2xxs_grid_data.inc"
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

#define QK_K 256

// CPU dot product
float cpu_iq2_xxs_dot(const uint8_t *block, const float *x) {
    uint16_t d_bits;
    memcpy(&d_bits, block, 2);
    float d = f16_to_f32_local(d_bits);
    const uint16_t *qs16 = (const uint16_t *)(block + 2);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float total = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        memcpy(aux32, qs16 + 4*ib32, 2*sizeof(uint32_t));
        float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t *)(&iq2xxs_grid[aux8[l]]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 8; j++) {
                float val = db * (float)grid[j];
                if (signs & kmask_iq2xs[j]) total += -val * x[base + j];
                else total += val * x[base + j];
            }
        }
    }
    return total;
}

// Correct CPU F16→F32 (matches ldexpf approach)
float f16_to_f32_correct(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant =  v        & 0x3FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    
    // Load the GGUF model to get a real MoE weight
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "Failed to open model\n"); return 1; }
    
    // Read an IQ2_XXS block from blk.0.ffn_gate_exps.weight
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) { fprintf(stderr, "Could not find gate tensor\n"); gguf_close(ctx); return 1; }
    
    printf("Gate tensor: type=%d dims=[%lld, %lld] offset=%lld\n", 
           t->ggml_type, (long long)t->dims[0], (long long)t->dims[1], (long long)t->data_offset);
    
    // Read the first block of the first expert
    uint8_t block[66];
    gguf_read_raw(ctx, t, 0, block, 66);
    
    uint16_t d_bits;
    memcpy(&d_bits, block, 2);
    printf("F16 d bits: 0x%04x\n", d_bits);
    printf("F16 d (local):  %.10e\n", f16_to_f32_local(d_bits));
    printf("F16 d (correct): %.10e\n", f16_to_f32_correct(d_bits));
    
    // Create a random-ish input vector
    float x[256];
    srand(42);
    for (int i = 0; i < 256; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    
    float cpu_result = cpu_iq2_xxs_dot(block, x);
    printf("CPU IQ2_XXS dot: %.10f\n", cpu_result);
    
    // Also check many blocks of the first column (first expert, column 0)
    printf("\nFirst 8 blocks of expert 0, column 0:\n");
    int blk_sz = 66;
    float cpu_total_col = 0;
    for (int b = 0; b < 8; b++) {
        uint8_t blk[66];
        gguf_read_raw(ctx, t, (int64_t)b * blk_sz, blk, 66);
        float d = cpu_iq2_xxs_dot(blk, x + b * 256);
        cpu_total_col += d;
        printf("  Block %d: %.10f\n", b, d);
    }
    printf("Total column 0: %.10f\n", cpu_total_col);
    
    // Check F16 d values for all blocks in column 0
    printf("\nF16 d values for all 8 blocks of column 0:\n");
    int denormal_count = 0;
    for (int b = 0; b < 8; b++) {
        uint8_t blk[66];
        gguf_read_raw(ctx, t, (int64_t)b * blk_sz, blk, 66);
        memcpy(&d_bits, blk, 2);
        float d_old = f16_to_f32_local(d_bits);
        float d_new = f16_to_f32_correct(d_bits);
        int is_denorm = ((d_bits >> 10) & 0x1F) == 0;
        printf("  Block %d: d_bits=0x%04x old=%.10e new=%.10e %s\n", 
               b, d_bits, d_old, d_new, is_denorm ? "(DENORM)" : "");
        if (is_denorm) denormal_count++;
    }
    printf("Denormal count: %d/8\n", denormal_count);
    
    // Compare old vs new F16 conversion for ALL blocks in column
    printf("\nComparing old vs new F16 conversion (all blocks in column 0):\n");
    float diff_max = 0;
    for (int b = 0; b < 8; b++) {
        uint8_t blk[66];
        gguf_read_raw(ctx, t, (int64_t)b * blk_sz, blk, 66);
        memcpy(&d_bits, blk, 2);
        float d_old = f16_to_f32_local(d_bits);
        float d_new = f16_to_f32_correct(d_bits);
        float diff = fabsf(d_old - d_new);
        if (diff > diff_max) { diff_max = diff; printf("  Block %d: old=%.10e new=%.10e diff=%.10e\n", b, d_old, d_new, diff); }
    }
    printf("Max diff between old/new F16: %.10e\n", diff_max);
    
    gguf_close(ctx);
    return 0;
}
