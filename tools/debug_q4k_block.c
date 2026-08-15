#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_fp16.h"

int main() {
    const char *path = "/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (!t) { fprintf(stderr, "not found\n"); return 1; }
    printf("attn_q: type=%d dims=%lld %lld\n", t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);

    const uint8_t *weight_q = blob + t->data_offset;
    
    /* Print first block raw bytes */
    printf("\nFirst block (144 bytes):\n");
    for (int i = 0; i < 144; i++) {
        if (i % 16 == 0) printf("\n%03d: ", i);
        printf("%02x ", weight_q[i]);
    }
    printf("\n");
    
    /* Extract d and dmin */
    uint16_t d_bits = *(uint16_t*)weight_q;
    uint16_t dmin_bits = *(uint16_t*)(weight_q + 2);
    printf("\nd_bits = 0x%04x, dmin_bits = 0x%04x\n", d_bits, dmin_bits);
    printf("d = %f, dmin = %f\n", f16_to_f32_cpu(d_bits), f16_to_f32_cpu(dmin_bits));
    
    /* Print scales */
    const uint8_t *scales = weight_q + 4;
    printf("\nScales (12 bytes):\n");
    for (int i = 0; i < 12; i++) printf("%02x ", scales[i]);
    printf("\n");
    
    /* Print qs start */
    const uint8_t *qs = weight_q + 16;
    printf("\nQS first 32 bytes:\n");
    for (int i = 0; i < 32; i++) {
        if (i % 16 == 0) printf("\n%03d: ", i);
        printf("%02x ", qs[i]);
    }
    printf("\n");
    
    /* Now test reference dequant on first 256 elements */
    float ref[256];
    gguf_dequantize(weight_q, t->ggml_type, 256, ref);
    printf("\nReference first 16:\n");
    for (int i = 0; i < 16; i++) printf("  [%d] = %.6f\n", i, ref[i]);
    
    gguf_close(ctx);
    return 0;
}
