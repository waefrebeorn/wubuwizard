#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_fp16.h"

#define QK_K 256

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Check IQ2_XXS tensors for denormal F16 d values
    int iq2_count = 0;
    int denorm_d = 0;
    int total_blocks = 0;
    
    for (int i = 0; i < ctx->n_tensors && iq2_count < 3; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        if (t->ggml_type != GGML_TYPE_IQ2_XXS) continue;  // GGML_TYPE_IQ2_XXS = 16
        iq2_count++;
        
        int64_t n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        int n_blocks = (n_elems + QK_K - 1) / QK_K;
        const uint8_t *data = blob + t->data_offset;
        
        printf("IQ2_XXS tensor '%s': %ld elems, %d blocks\n", t->name, (long)n_elems, n_blocks);
        
        int local_denorm = 0;
        for (int b = 0; b < n_blocks && b < 100000; b++) {
            const uint8_t *block = data + b * 66;  // IQ2_XXS: 66 bytes/block
            uint16_t d_bits;
            memcpy(&d_bits, block, 2);
            int d_exp = (d_bits >> 10) & 0x1F;
            
            if (d_exp == 0 && d_bits != 0) {
                float d_cpu = f16_to_f32(d_bits);
                if (local_denorm < 3) {
                    printf("  block %d: d=0x%04X (exp=0) cpu=%e gpu=0.0\n", b, d_bits, d_cpu);
                }
                local_denorm++;
                denorm_d++;
            }
        }
        total_blocks += n_blocks;
        printf("  %d/%d blocks have denormal d (%.1f%%)\n", local_denorm, n_blocks<100000?n_blocks:100000, 100.0*local_denorm/(n_blocks<100000?n_blocks:100000));
    }
    
    printf("\nTotal: %d IQ2_XXS tensors checked, %d/%d blocks with denormal d\n", iq2_count, denorm_d, total_blocks);
    
    gguf_close(ctx);
    return 0;
}
