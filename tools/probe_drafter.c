/* probe_drafter.c -- open the DSpark drafter GGUF with the in-engine C reader
 * and read the markov-head tensors, proving the C loader path handles the
 * drafter (incl. markov_w1/w2/joint_proj quant type 30).
 * usage: probe_drafter <drafter.gguf> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_reader.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: probe_drafter <gguf>\n"); return 2; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "PROBE OPEN FAIL\n"); return 1; }
    static const char *names[] = {
        "joint_proj.weight","markov_w1.weight","markov_w2.weight",
        "markov_ctx_mix.weight","markov_ctx_norm.weight",
        "markov_w2.bias","mask_embd.weight","output_norm.weight", NULL };
    int nf = 0;
    for (int i = 0; names[i]; ++i) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, names[i]);
        if (!t) { printf("  %-26s MISSING\n", names[i]); continue; }
        int64_t nelem = 1; for (int d = 0; d < t->n_dims; ++d) nelem *= t->dims[d];
        int64_t cap = nelem < 64 ? 4096 : 64;   /* probe a small slice unless tiny */
        float *buf = (float *)malloc(sizeof(float)*(cap<nelem?cap:nelem));
        int64_t want = cap < nelem ? cap : nelem;
        int rc = gguf_read_tensor_f32(ctx, t, buf, want);
        printf("  %-26s dims=%d ne=[", names[i], t->n_dims);
        printf("] type=%d nelem=%lld read=%d first=[", t->ggml_type, (long long)nelem, rc);
        for (int d = 0; d < t->n_dims; ++d) printf("%lld%s", (long long)t->dims[d], d+1<t->n_dims?",":"");
        for (int k = 0; k < 4 && k < (int)(want); ++k) printf("%.4g%s", buf[k], k+1<4&&k+1<(int)want?",":"");
        if (want>4) printf("...");
        printf("]\n");
        ++nf; free(buf);
    }
    gguf_close(ctx);
    printf("PROBE tensors-found=%d\n", nf);
    return nf >= 4 ? 0 : 1;
}
