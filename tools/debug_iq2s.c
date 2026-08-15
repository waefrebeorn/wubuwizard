#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "wubu_fp16.h"

// Copied from gguf_reader.c for local use
#define QK_K 256
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
// iq2s_grid table (1024 entries, each uint64 packs 8 int8 values)
static uint64_t iq2s_grid[1024] = {
#include "iq2s_grid.inc"
};

static void reference_dequant_iq2s_block(const uint8_t *block, float *output) {
    uint16_t d_bits;
    memcpy(&d_bits, block, 2);
    float d = wubu_f16_to_f32(d_bits);
    
    const uint8_t *qs = block + 2;        // 64 bytes
    const uint8_t *qh = block + 66;       // 8 bytes
    const uint8_t *scales = block + 74;   // 8 bytes
    const uint8_t *signs = qs + 32;       // last 32 bytes of qs
    
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        float db[2];
        db[0] = d * (0.5f + (float)(scales[ib32] & 0x0F)) * 0.25f;
        db[1] = d * (0.5f + (float)(scales[ib32] >>   4)) * 0.25f;
        
        for (int l = 0; l < 4; l++) {
            float dl = db[l/2];
            uint16_t grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
            const int8_t *grid = (const int8_t *)(&iq2s_grid[grid_idx]);
            
            for (int j = 0; j < 8; j++) {
                float val = dl * (float)grid[j];
                if (signs[l] & kmask_iq2xs[j]) val = -val;
                output[j] = val;
            }
            output += 8;
        }
        qs += 4;
        signs += 4;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    const char *path = argv[1];
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    
    // Find first IQ2_S tensor
    gguf_tensor_info *iq2s_tensor = NULL;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].ggml_type == GGML_TYPE_IQ2_S) {
            iq2s_tensor = &ctx->tensors[i];
            break;
        }
    }
    
    if (!iq2s_tensor) { fprintf(stderr, "No IQ2_S tensor found\n"); gguf_close(ctx); return 1; }
    
    int64_t n_elems = 1;
    for (int d = 0; d < iq2s_tensor->n_dims; d++) n_elems *= iq2s_tensor->dims[d];
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    
    printf("Tensor: %s, n_elems=%ld, n_blocks=%ld\n", iq2s_tensor->name, n_elems, n_blocks);
    printf("data_offset=%lu, data_blob_offset=%lu, file_pos=%lu\n", 
           iq2s_tensor->data_offset, ctx->data_blob_offset, 
           ctx->data_blob_offset + iq2s_tensor->data_offset);
    
    // Seek to raw data and read ALL blocks raw
    uint64_t tensor_pos = ctx->data_blob_offset + iq2s_tensor->data_offset;
    fseek(ctx->file, tensor_pos, SEEK_SET);
    
    // Read raw blocks using 82-byte block size
    size_t raw_size = n_blocks * 82;
    uint8_t *raw = malloc(raw_size);
    if (!raw) { fprintf(stderr, "malloc failed\n"); gguf_close(ctx); return 1; }
    size_t n_read = fread(raw, 1, raw_size, ctx->file);
    printf("fread: requested %zu bytes, got %zu bytes\n", raw_size, n_read);
    
    if (n_read != raw_size) {
        printf("WARNING: fread short read! EOF or error.\n");
    }
    
    // Dequantize block 0 manually and print debug info
    printf("\n=== Block 0 debug ===\n");
    {
        const uint8_t *block = raw;
        uint16_t d_bits;
        memcpy(&d_bits, block, 2);
        float d = wubu_f16_to_f32(d_bits);
        printf("d_bits=0x%04x, d=%f\n", d_bits, d);
        
        printf("raw[0..81] hex: ");
        for (int i = 0; i < 82; i++) printf("%02x", raw[i]);
        printf("\n");
        
        printf("qh bytes (offset 66-73): ");
        for (int i = 0; i < 8; i++) printf("%02x ", raw[66+i]);
        printf("\n");
        
        printf("scales bytes (offset 74-81): ");
        for (int i = 0; i < 8; i++) printf("%02x ", raw[74+i]);
        printf("\n");
        
        float *block_out = malloc(256 * sizeof(float));
        reference_dequant_iq2s_block(block, block_out);
        
        // Stats for this block
        double sum = 0, sum2 = 0;
        float bmin = block_out[0], bmax = block_out[0];
        for (int i = 0; i < 256; i++) {
            float v = block_out[i];
            sum += v;
            sum2 += v*v;
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
        }
        printf("Block 0: %d elems, min=%.3f, max=%.3f, mean=%.3f, stddev=%.3f\n",
               256, bmin, bmax, sum/256, sqrt(sum2/256 - (sum/256)*(sum/256)));
        printf("Block 0 first 8: ");
        for (int i = 0; i < 8; i++) printf("%+.4f ", block_out[i]);
        printf("\n");
        free(block_out);
    }
    
    // Now dequantize ALL blocks using gguf_read_tensor_f32
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) { free(raw); gguf_close(ctx); return 1; }
    
    // Reset file position (gguf_read_tensor_f32 does fseek itself)
    int n_read2 = gguf_read_tensor_f32(ctx, iq2s_tensor, buf, n_elems);
    printf("\nDequantized %d elements via gguf_read_tensor_f32\n", n_read2);
    
    // Now manually dequantize first few blocks from raw buffer and compare
    printf("\n=== Comparison: raw-vs-gguf first block ===\n");
    float *ref_block = malloc(256 * sizeof(float));
    reference_dequant_iq2s_block(raw, ref_block);
    
    int mismatches = 0;
    double max_diff = 0;
    for (int i = 0; i < 256 && i < n_read2; i++) {
        float diff = fabs(ref_block[i] - buf[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            mismatches++;
            if (mismatches <= 5)
                printf("  diff at [%d]: ref=%+.6f got=%+.6f (diff=%f)\n", i, ref_block[i], buf[i], diff);
        }
    }
    printf("Block 0: max_diff=%f, mismatches=%d/256\n", max_diff, mismatches);
    
    // Now check multiple blocks for correctness
    printf("\n=== Per-block stats for first 5 blocks ===\n");
    for (int b = 0; b < 5 && b < n_blocks; b++) {
        const uint8_t *block = raw + b * 82;
        
        uint16_t d_bits;
        memcpy(&d_bits, block, 2);
        float d = wubu_f16_to_f32(d_bits);
        
        float *bout = malloc(256 * sizeof(float));
        reference_dequant_iq2s_block(block, bout);
        
        double sum = 0, sum2 = 0;
        float bmin = bout[0], bmax = bout[0];
        int extreme = 0;
        for (int i = 0; i < 256; i++) {
            float v = bout[i];
            sum += v; sum2 += v*v;
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
            if (fabs(v) > 10) extreme++;
        }
        printf("Block %d: d=%+.6f min=%+.1f max=%+.1f mean=%.3f std=%.3f extreme=%d\n",
               b, d, bmin, bmax, sum/256, sqrt(sum2/256 - (sum/256)*(sum/256)), extreme);
        
        // Compare with gguf output
        mismatches = 0;
        max_diff = 0;
        for (int i = 0; i < 256; i++) {
            float diff = fabs(bout[i] - buf[b*256 + i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 0.01f) mismatches++;
        }
        if (max_diff > 0.01f) {
            printf("  -> MISMATCH with gguf output: max_diff=%f mismatches=%d/256\n", max_diff, mismatches);
        } else {
            printf("  -> MATCHES gguf output\n");
        }
        
        free(bout);
    }
    
    free(ref_block);
    free(buf);
    free(raw);
    gguf_close(ctx);
    return 0;
}
