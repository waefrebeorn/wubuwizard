/* spot.c -- hand-dequant first 8 F16 of time_embed.2.weight and compare
 * to gguf_read_tensor_f32; also print raw bytes. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float h2f(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) u = (sign << 31) | (127 - 15 + 1) << 23 | (mant << 13);
    else if (exp == 31) u = (sign << 31) | 0xFF << 23 | (mant << 13);
    else u = (sign << 31) | (127 - 15 + exp) << 23 | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
}
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    gguf_tensor_info *ti = gguf_find_tensor(g, "model.diffusion_model.time_embed.2.weight");
    printf("data_offset=%llu type=%d\n", (unsigned long long)ti->data_offset, ti->ggml_type);
    const uint8_t *raw = (const uint8_t *)g->data_blob + ti->data_offset;
    printf("raw first 16 bytes:");
    for (int i = 0; i < 16; i++) printf(" %02x", raw[i]);
    printf("\n");
    for (int i = 0; i < 8; i++) {
        uint16_t h;
        memcpy(&h, raw + i * 2, 2);
        printf("raw[%d] = %04x -> hand %g", i, h, h2f(h));
        /* canonical */
        float *f = (float *)malloc(1638400 * 4);
        gguf_read_tensor_f32(g, ti, f, 1638400);
        printf("  canonical %g  %s\n", f[i], f[i] == h2f(h) ? "MATCH" : "DIFFER");
        free(f);
    }
    return 0;
}
