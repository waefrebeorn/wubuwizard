/* expert_probe -- dump the engine's own decoding of one expert block, next to its raw bytes.
 *
 * Purpose: pin the bit packing of a quant type from evidence. Guessing at the layout from the C source
 * produced five wrong readings; the engine's decoder is known to work, so its first outputs plus the raw
 * bytes are enough to solve the packing by construction.
 *
 * usage: expert_probe <model.gguf> <tensor-name> [n_values]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_reader.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <model> <tensor> [n]\n", argv[0]); return 2; }
    const int n_show = argc > 3 ? atoi(argv[3]) : 8;
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    gguf_tensor_info *t = gguf_find_tensor(ctx, argv[2]);
    if (!t) { fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }

    /* 256 elements = one block for every K-quant / IQ type here */
    const size_t n_elem = 256, n_bytes = 4096;
    uint8_t *raw = (uint8_t *)calloc(1, n_bytes);
    float *out = (float *)calloc(n_elem, sizeof(float));
    if (!raw || !out) return 1;

    /* Read just the head of the tensor: gguf_read_raw_tensor wants the whole 196 MB, which is pointless
     * here. The data blob starts at the 32-aligned end of the tensor directory. */
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "reopen failed\n"); return 1; }
    /* (long) fseek truncates above 4 GB on Windows - this offset is ~7.96e9, so it must be the 64-bit form.
     * Same trap that produced garbage Q4_K values from gguf_probe earlier in this port. */
    if (_fseeki64(fp, (long long)(11978688ULL + t->data_offset), SEEK_SET) != 0) {
        fprintf(stderr, "seek failed\n"); fclose(fp); return 1;
    }
    if (fread(raw, 1, n_bytes, fp) != n_bytes) { fprintf(stderr, "short read\n"); fclose(fp); return 1; }
    fclose(fp);
    gguf_dequantize(raw, t->ggml_type, (int64_t)n_elem, out);

    printf("tensor %s  type=%d  n_dims=%d  dims=%lld,%lld,%lld  data_offset=%llu\n",
           argv[2], t->ggml_type, t->n_dims,
           (long long)t->dims[0], (long long)t->dims[1], (long long)t->dims[2],
           (unsigned long long)t->data_offset);
    printf("first %d dequantized values:\n", n_show);
    for (int i = 0; i < n_show; ++i) printf("  [%d] %.9g\n", i, out[i]);
    printf("raw bytes 0..73:\n ");
    for (int i = 0; i < 74; ++i) printf(" %02x", raw[i]);
    printf("\n");
    free(raw); free(out); gguf_close(ctx);
    return 0;
}
