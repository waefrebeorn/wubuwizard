/* test_lfm2_load_gguf.c — load the REAL LFM2.5-2.6B GGUF with
 * our modified loader ("make it load whatever"). */
#include "lfm2_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    lfm2_model_t m;
    if (!lfm2_load(argv[1], &m)) {
        fprintf(stderr, "LOAD FAILED\n");
        return 1;
    }
    printf("LOADED: d_model=%d layers=%d q_heads=%d kv_heads=%d "
           "head_dim=%d ff_dim=%d vocab=%d conv_dim=%d rope_theta=%.0f\n",
           m.d_model, m.n_layers, m.n_q_heads, m.n_kv_heads,
           m.head_dim, m.ff_dim, m.vocab_size, m.conv_dim, m.rope_theta);
    printf("embed=%s embed_norm=%s\n",
           m.embed ? "OK" : "MISSING",
           m.embed_norm ? "OK" : "MISSING");
    int ok_layers = 0, missing = 0;
    for (int l = 0; l < m.n_layers; l++) {
        lfm2_layer_t *L = &m.layers[l];
        if (m.is_conv[l]) {
            if (L->in_proj && L->conv_w && L->out_proj) ok_layers++;
            else missing++;
        } else {
            if (L->q_proj && L->k_proj && L->v_proj && L->o_proj &&
                L->w1 && L->w2 && L->w3) ok_layers++;
            else missing++;
        }
    }
    printf("layers ok=%d missing=%d\n", ok_layers, missing);
    lfm2_free(&m);
    return (ok_layers > 0 && missing == 0) ? 0 : 1;
}
