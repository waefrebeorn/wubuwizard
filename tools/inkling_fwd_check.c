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
    /* Static short conv check: run attn_post_conv over 3 deterministic tokens and dump the result.
     * Verifiable in Python from the same weights, which pins tap order and the 1/sqrt(k+1) block. */
    extern void in2_shortconv(const float *, int, int, const float *, int, float *);
    {
        const int T = 3, C = 2304;
        float *pc = in2_tensor_to_f32(ctx, &m.layer[0].attn_post_conv, NULL);
        if (pc) {
            float *in = (float *)calloc((size_t)T * C, sizeof(float));
            float *out = (float *)calloc((size_t)T * C, sizeof(float));
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c)
                    in[(size_t)t * C + c] = xn[c] * (float)(t + 1) * (float)((c % 7) - 3) * 0.01f;
            in2_shortconv(in, T, C, pc, 4, out);
            printf("shortconv t2 c0..3: %+.6f %+.6f %+.6f %+.6f\n", out[2*C+0], out[2*C+1], out[2*C+2], out[2*C+3]);
            FILE *cf = fopen("conv_check.bin", "wb");
            if (cf) { fwrite(in, sizeof(float), (size_t)T*C, cf); fclose(cf); }
            FILE *of = fopen("conv_out.bin", "wb");
            if (of) { fwrite(out, sizeof(float), (size_t)T*C, of); fclose(of); }
            free(in); free(out); free(pc);
        }
    }

    /* Relative-position bias check: attn_rel(x) -> clip(rel_clip) -> mix the 16 learned distance
     * profiles. Dumps the 24x1024 table for an independent recomputation. */
    {
        extern void in2_l2_clip(float *, const float *, int, int, float);
        extern void in2_rel_bias_table(const float *, const float *, int, int, int, float *);
        const int NH = 24, DREL = 16, EXT = 1024;
        float *wrel = in2_tensor_to_f32(ctx, &m.layer[0].wrel, NULL);
        float *rclip = in2_tensor_to_f32(ctx, &m.layer[0].rel_clip, NULL);
        float *P = in2_tensor_to_f32(ctx, &m.layer[0].rel_bias, NULL);
        if (wrel && rclip && P) {
            float *rs = (float *)calloc(NH * DREL, sizeof(float));
            float *bias = (float *)calloc((size_t)NH * EXT, sizeof(float));
            in2_matvec(wrel, xn, d, NH * DREL, rs);
            in2_l2_clip(rs, rclip, NH, DREL, 8.0f);
            in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);
            printf("rel_bias h0 d0..3: %+.6f %+.6f %+.6f %+.6f\n", bias[0], bias[1], bias[2], bias[3]);
            printf("rel_bias h23 d0..3: %+.6f %+.6f %+.6f %+.6f\n", bias[23*EXT], bias[23*EXT+1], bias[23*EXT+2], bias[23*EXT+3]);
            FILE *bf = fopen("relbias.bin", "wb");
            if (bf) { fwrite(rs, sizeof(float), NH*DREL, bf); fwrite(bias, sizeof(float), (size_t)NH*EXT, bf); fclose(bf); }
            free(rs); free(bias);
        }
        free(wrel); free(rclip); free(P);
    }

    /* MoE check: router selection/weights, then one expert end-to-end. */
    {
        extern int in2_router(const float *, const float *, const float *, int, int, int, float, int *, float *);
        extern int in2_dequant_slice(gguf_ctx *, const wubu_in2_tensor_t *, int64_t, int64_t, float *);
        extern void in2_expert_act(const float *, const float *, int, float *);
        const int NE = 128, TOPK = 8, DFF = 1152;
        float *Wr = in2_tensor_to_f32(ctx, &m.layer[0].router, NULL);
        float *cb = in2_tensor_to_f32(ctx, &m.layer[0].exp_bias, NULL);
        int idx[8]; float w[8];
        int got = in2_router(xn, Wr, cb, NE, d, TOPK, 8.0f, idx, w);
        printf("router: %d experts ->", got);
        for (int k = 0; k < got; ++k) printf(" e%d=%.6f", idx[k], w[k]);
        printf("\n");
        double wsum = 0; for (int k = 0; k < got; ++k) wsum += w[k];
        printf("router weight sum (expect 8.0): %.6f\n", wsum);

        const int e = idx[0];
        float *gu = (float *)calloc((size_t)2 * DFF * d, sizeof(float));
        float *dn = (float *)calloc((size_t)d * DFF, sizeof(float));
        float *g  = (float *)calloc(2 * DFF, sizeof(float));
        float *act = (float *)calloc(DFF, sizeof(float));
        float *eo = (float *)calloc(d, sizeof(float));
        float *dsc = in2_tensor_to_f32(ctx, &m.layer[0].down_scale, NULL);
        float *gsc = in2_tensor_to_f32(ctx, &m.layer[0].gate_up_scale, NULL);
        if (gu && dn && g && act && eo) {
            in2_dequant_slice(ctx, &m.layer[0].gate_up, (int64_t)e * 2 * DFF * d, (int64_t)2 * DFF * d, gu);
            in2_dequant_slice(ctx, &m.layer[0].down,    (int64_t)e * d * DFF,    (int64_t)d * DFF,    dn);
            const float gs = gsc ? gsc[e] : 1.0f, ds = dsc ? dsc[e] : 1.0f;
            in2_matvec(gu, xn, d, 2 * DFF, g);
            for (int i = 0; i < 2 * DFF; ++i) g[i] *= gs;          /* per-expert quant scale */
            in2_expert_act(g, g + DFF, DFF, act);
            in2_matvec(dn, act, DFF, d, eo);
            for (int i = 0; i < d; ++i) eo[i] *= ds;
            printf("expert e%d scale %.8g/%.8g  gate_up[0..2]=%.6f %.6f %.6f  out[0..3]=%.6f %.6f %.6f %.6f\n",
                   e, (double)gs, (double)ds, g[0], g[1], g[2], eo[0], eo[1], eo[2], eo[3]);
            FILE *mf = fopen("moe.bin", "wb");
            if (mf) { fwrite(act, sizeof(float), DFF, mf); fwrite(eo, sizeof(float), d, mf); fclose(mf); }
        }
        free(gu); free(dn); free(g); free(act); free(eo); free(dsc); free(gsc); free(Wr); free(cb);
    }

    /* Full sliding layer (blk.0) over 1 and 2 tokens. */
    {
        extern int in2_layer_sliding(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                                     int, float *, int, int, const float *, int, const float *);
        for (int N = 1; N <= 2; ++N) {
            float *hh = (float *)calloc((size_t)N * d, sizeof(float));
            for (int t = 0; t < N; ++t)
                for (int i = 0; i < d; ++i) hh[(size_t)t * d + i] = x[i] * (1.0f + 0.1f * t);
            int rc2 = in2_layer_sliding(ctx, &m, &cfg, 0, hh, N, 0, NULL, 0, NULL);
            double ss = 0, mx = 0; int bad = 0;
            for (int i = 0; i < N * d; ++i) {
                double vq = hh[i];
                if (!(vq == vq)) bad++;
                ss += vq * vq; if (fabs(vq) > mx) mx = fabs(vq);
            }
            printf("layer0 N=%d rc=%d rms=%.6f max|h|=%.4f nan=%d  h[0..3]=%.6f %.6f %.6f %.6f\n",
                   N, rc2, sqrt(ss / (N * d)), mx, bad, hh[0], hh[1], hh[2], hh[3]);
            FILE *lf = fopen(N == 1 ? "layer0_n1.bin" : "layer0_n2.bin", "wb");
            if (lf) { fwrite(hh, sizeof(float), (size_t)N * d, lf); fclose(lf); }
            free(hh);
        }
    }

    free(wq); free(x); free(xn); free(an);
    gguf_close(ctx);
    return 0;
}
