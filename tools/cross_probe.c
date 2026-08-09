/* cross_probe.c -- verify the cross-attn k/v path: project ctx (cond vs
 * empty) through to_k/to_v of attn2 in block 1.1, compare outputs.
 * If k_cond == k_empty, the CLIP context is wrong. */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "gguf_reader.h"
#include "wubu_sd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g->data_blob) gguf_buffer_data(g);
    wubu_sd_clip_t *clip = wubu_sd_clip_load(g);
    float ctx[77*768], pool[768], ectx[77*768], epool[768];
    wubu_sd_clip_encode(clip, "a serene anime landscape, soft morning light, detailed", ctx, pool);
    wubu_sd_clip_encode(clip, "", ectx, epool);
    /* CLIP context stats + diff */
    double m1=0,m2=0,d2=0;
    for (int i = 0; i < 77*768; i++) { m1+=ctx[i]; m2+=ectx[i]; d2+=(ctx[i]-ectx[i])*(ctx[i]-ectx[i]); }
    m1/=77*768; m2/=77*768;
    printf("clip ctx: cond mean=%+.4f empty mean=%+.4f diff rms=%+.4f\n", m1, m2, sqrt(d2/(77*768)));
    /* project through attn2.to_k of input_blocks.1.1 */
    gguf_tensor_info *ti = gguf_find_tensor(g, "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight");
    const uint8_t *wk = (const uint8_t *)g->data_blob + ti->data_offset;
    float kc[77*320], ke[77*320];
    extern void wubu_sd_linear_q(const float *x, const void *W, int type, int M, int K, int N, float *y);
    wubu_sd_linear_q(ctx, wk, ti->ggml_type, 77, 768, 320, kc);
    wubu_sd_linear_q(ectx, wk, ti->ggml_type, 77, 768, 320, ke);
    double dk=0, rmsk=0;
    for (int i = 0; i < 77*320; i++) { dk+=(kc[i]-ke[i])*(kc[i]-ke[i]); rmsk+=kc[i]*kc[i]; }
    printf("attn2.to_k: cond-vs-empty diff rms=%+.4f (k rms=%+.4f) %s\n",
           sqrt(dk/(77*320)), sqrt(rmsk/(77*320)),
           sqrt(dk/(77*320)) > 0.1*sqrt(rmsk/(77*320)) ? "CONTEXT FLOWS" : "WEAK");
    return 0;
}
