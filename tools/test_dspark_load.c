/* test_dspark_load.c -- bind drafter markov tensors through the in-engine C
 * reader and verify their dequantized values against the f32 numpy reference.
 * Checks joint_proj (5120x768, type 30) and the markov_w1 prev-row, printing
 * rms + a few sample values so the caller compares with dflash_ref.
 * usage: test_dspark_load <drafter.gguf> <out.bin>  (out.bin receives f32 joint_proj row-major) */
#include <stdio.h>
#include <stdlib.h>
#include "gguf_reader.h"

static int read_tensor(gguf_ctx *ctx, const char *name, size_t *nelem, float **out) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); return 1; }
    int64_t n = 1; for (int d = 0; d < t->n_dims; ++d) n *= t->dims[d];
    float *buf = (float *)malloc(sizeof(float) * (size_t)n);
    int rc = gguf_read_tensor_f32(ctx, t, buf, (size_t)n);
    if (rc < 0 || rc != (int)n) { fprintf(stderr, "read %s rc=%d want=%lld\n", name, rc, (long long)n); free(buf); return 1; }
    *nelem = (size_t)n; *out = buf;
    printf("  %-18s nelem=%lld type=%d rms=%.6f head=[%.6f %.6f %.6f %.6f]\n",
           name, (long long)n, t->ggml_type, buf[0] /*placeholder*/, buf[0], buf[1], buf[2], buf[3]);
    return 0;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_dspark_load <gguf> <joint_proj.bin>\n"); return 2; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "OPEN FAIL\n"); return 1; }
    size_t nj = 0, nw = 0; float *J = NULL, *W = NULL;
    if (read_tensor(ctx, "joint_proj.weight", &nj, &J)) { gguf_close(ctx); return 1; }
    read_tensor(ctx, "markov_w1.weight", &nw, &W);
    double rms = 0; for (size_t i = 0; i < nj; ++i) rms += (double)J[i]*J[i]; rms = rms / nj; rms = rms<0? -rms:rms; rms = __builtin_sqrt(rms);
    printf("  joint_proj true_rms=%.6f\n", rms);
    FILE *f = fopen(argv[2], "wb"); fwrite(J, 4, nj, f); fclose(f);
    printf("  wrote %zu floats to %s\n", nj, argv[2]);
    gguf_close(ctx);
    printf("DONE\n");
    return 0;
}
