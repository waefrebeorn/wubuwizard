#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_fp16.h"

#define QK_K 256

/* f16_to_f32: use wubu_fp16.h */

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Scan all tensors
    int q5k_count = 0;
    int denorm_count = 0;
    int total_blocks = 0;
    
    for (int i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        if (t->ggml_type != GGML_TYPE_Q5_K) continue;
        q5k_count++;
        
        int64_t n_elems = t->dims[0] * t->dims[1];
        if (t->n_dims > 2) n_elems *= t->dims[2];
        if (t->n_dims > 3) n_elems *= t->dims[3];
        
        int n_blocks = (n_elems + QK_K - 1) / QK_K;
        const uint8_t *data = blob + t->data_offset;
        
        int local_denorm = 0;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *block = data + b * 176;
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block, 2);
            memcpy(&dmin_bits, block + 2, 2);
            
            int d_exp = (d_bits >> 10) & 0x1F;
            int dmin_exp = (dmin_bits >> 10) & 0x1F;
            
            float d_cpu = f16_to_f32(d_bits);
            float dmin_cpu = f16_to_f32(dmin_bits);
            
            // GPU: flush to zero
            float d_gpu = (d_exp == 0) ? 0.0f : f16_to_f32(d_bits);
            float dmin_gpu = (dmin_exp == 0) ? 0.0f : f16_to_f32(dmin_bits);
            
            if (d_exp == 0 && d_bits != 0) {
                if (local_denorm == 0) {
                    printf("  Q5_K tensor '%s' block %d: d=0x%04X (exp=0) cpu=%e gpu=%e\n",
                           t->name, b, d_bits, d_cpu, d_gpu);
                }
                local_denorm++;
                denorm_count++;
            }
            if (dmin_exp == 0 && dmin_bits != 0) {
                if (local_denorm == 0) {
                    printf("  Q5_K tensor '%s' block %d: dmin=0x%04X (exp=0) cpu=%e gpu=%e\n",
                           t->name, b, dmin_bits, dmin_cpu, dmin_gpu);
                }
                local_denorm++;
                denorm_count++;
            }
        }
        total_blocks += n_blocks;
        printf("  Q5_K tensor '%s': %d blocks, %d with denormals (%s)\n",
               t->name, n_blocks, local_denorm,
               local_denorm > 0 ? "AFFECTED" : "clean");
    }
    
    printf("\nSummary: %d Q5_K tensors, %d blocks, %d denormal blocks, %d denorm values\n",
           q5k_count, total_blocks, denorm_count, denorm_count);
    
    gguf_close(ctx);
    return 0;
}
