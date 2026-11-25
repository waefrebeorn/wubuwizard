/* iq_bench -- standalone throughput of the engine's quant row kernels.
 *
 * The in-situ number (3.06 GMAC for one token's MoE in 0.30 s = 20 GFLOPS on one core) is an aggregate over 384
 * small kernel calls with routing, activations and thread handoffs between them. This isolates the kernel itself on
 * a real tensor from the real file, so "the kernel is slow" and "we call it badly" stop being the same hypothesis.
 *
 * usage: iq_bench <tensor-name-or-type> [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"
#include "wubu_iq_c11.h"
#include "wubu_iq2q8.h"

int in2_tensor_to_f32(gguf_ctx *, const wubu_in2_tensor_t *, int64_t *);
const unsigned char *wubu_inkling2_map_ptr(gguf_ctx *, const wubu_in2_tensor_t *);

int main(int argc, char **argv)
{
    const char *path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    const int want_ty = (argc > 1) ? atoi(argv[1]) : 16;
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    /* pick a layer whose gate_up expert matrix has the requested type */
    int il = -1;
    for (int i = 6; i < 18; ++i) {
        wubu_in2_tensor_t *t = &m.layer[i].gate_up;
        if (t && t->info && t->info->ggml_type == want_ty) { il = i; break; }
    }
    if (il < 0) { fprintf(stderr, "no core layer with gate_up type %d\n", want_ty); return 4; }
    wubu_in2_tensor_t *t = &m.layer[il].gate_up;
    const int n_in = cfg.d_model, n_out = 2 * cfg.d_ff_expert;
    const size_t rb = gguf_raw_size(t->info->ggml_type, n_in);
    const unsigned char *base = wubu_inkling2_map_ptr(ctx, t);
    if (!base) { fprintf(stderr, "no mapping\n"); return 5; }

    /* one activation, quantised exactly as the engine does it */
    float *x = (float *)malloc(sizeof(float) * n_in);
    for (int i = 0; i < n_in; ++i) x[i] = (float)((i * 37 % 101) - 50) / 50.0f;
    in2_q8k_blk *act = (in2_q8k_blk *)malloc(sizeof(in2_q8k_blk) * (size_t)(n_in / 256));
    in2_q8k_quantise(x, n_in / 256, act);
    float *out = (float *)malloc(sizeof(float) * n_out);
    const void *ap[1] = { act };
    float *op[1] = { out };

    wubu_iq_rows(t->info->ggml_type, base, rb, n_in, ap, 1, op, 0, n_out);   /* warm */
    /* Cycle through every expert slice, not one matrix: a 1.3 MB matrix stays in L2/L3 across reps and reports the
     * compute ceiling (35 GFLOPS), which is not what a decode step does - it streams a different expert every call. */
    const int NE = cfg.n_expert;
    const size_t slice = rb * (size_t)n_out;
    const clock_t c0 = clock();
    for (int r = 0; r < reps * NE; ++r) {
        const int e = r % NE;
        wubu_iq_rows(t->info->ggml_type, base + slice * (size_t)e, rb, n_in, ap, 1, op, 0, n_out);
    }
    const double sec = (double)(clock() - c0) / (double)CLOCKS_PER_SEC;
    const int total = reps * NE;

    const double macs = (double)total * (double)n_out * (double)n_in;
    printf("ty=%d layer=%d n_in=%d n_out=%d calls=%d over %d experts  %.3f s  %.2f GFLOPS  %.2f GB/s\n",
           t->info->ggml_type, il, n_in, n_out, total, NE, sec, macs * 2.0 / sec / 1e9,
           (double)total * (double)n_out * (double)rb / sec / 1e9);
    free(x); free(act); free(out); gguf_close(ctx);
    return 0;
}
