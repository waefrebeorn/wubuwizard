/* inkling_fwd_check -- EG-118 step 3 instrument.
 * Embeds one token, norms it with blk.0.attn_norm, dumps the result, and prints the first four rows of
 * attn_q @ x. An independent Q6_K implementation recomputes those four dot products from the same file,
 * so agreement validates the dequantiser, the row-major layout and the matvec together.
 */
#include <stdio.h>
#include <stdlib.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out);
int  in2_embed(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
               int token, float *out);

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    const int token = argc > 2 ? atoi(argv[2]) : 1234;
    const int d = 2304;

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    float *x = (float *)calloc(d, sizeof(float));
    int rc = in2_embed(ctx, &m, &cfg, token, x);
    if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "embed rc=%d\n", rc); return 4; }

    /* blk.0 attn_norm is F32 (2304) */
    float *an = in2_tensor_to_f32(ctx, &m.layer[0].attn_norm, NULL);
    if (!an) { fprintf(stderr, "attn_norm load failed\n"); return 5; }
    float *xn = (float *)calloc(d, sizeof(float));
    in2_rmsnorm(x, an, d, cfg.rms_eps, xn);

    FILE *df = fopen("x_normed.bin", "wb");
    if (df) { fwrite(xn, sizeof(float), d, df); fclose(df); }

    printf("token %d\n", token);
    printf("embed   x[0..3] : %+.6f %+.6f %+.6f %+.6f\n", x[0], x[1], x[2], x[3]);
    printf("attn_norm[0..3]: %+.6f %+.6f %+.6f %+.6f\n", an[0], an[1], an[2], an[3]);
    printf("normed  x[0..3] : %+.6f %+.6f %+.6f %+.6f\n", xn[0], xn[1], xn[2], xn[3]);

    /* attn_q is (2304, 3072): dequantise, then four rows. */
    int64_t ne = 0;
    float *wq = in2_tensor_to_f32(ctx, &m.layer[0].wq, &ne);
    if (!wq) { fprintf(stderr, "attn_q load failed\n"); return 6; }
    printf("attn_q elements %lld (expect %d)\n", (long long)ne, 2304 * 3072);
    float y[4] = {0};
    in2_matvec(wq, xn, d, 4, y);
    printf("attn_q row0..3 : %+.8e %+.8e %+.8e %+.8e\n", y[0], y[1], y[2], y[3]);
    free(wq); free(x); free(xn); free(an);
    gguf_close(ctx);
    return 0;
}
