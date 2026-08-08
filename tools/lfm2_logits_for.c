/* lfm2_logits_for.c — print logits for specific tokens, on a prompt.
 * Usage: lfm2_logits_for <model.gguf> <print-tok>... -- <prompt-tok>...
 * Everything before -- is a token whose logit gets printed; after -- is
 * the prompt (token ids). */
#include "wubu_lfm2.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <model> <tok>... -- <prompt-tok>...\n", argv[0]); return 1; }
    int dash = 2;
    while (dash < argc && strcmp(argv[dash], "--")) dash++;
    if (dash >= argc - 1) { fprintf(stderr, "no prompt after --\n"); return 1; }
    int n_print = dash - 2;
    int T = argc - dash - 1;
    lfm2_model_t m;
    memset(&m, 0, sizeof(m));
    if (!lfm2_load(argv[1], &m)) { fprintf(stderr, "load failed\n"); return 1; }
    float *emb = (float *)malloc((size_t)T * m.d_model * sizeof(float));
    for (int t = 0; t < T; t++) {
        int id = atoi(argv[dash + 1 + t]);
        if (m.embed) {
            memcpy(emb + (size_t)t * m.d_model, m.embed + (size_t)id * m.d_model,
                   m.d_model * sizeof(float));
        } else if (m.q_embed) {
            const uint8_t *row = m.q_embed + (size_t)id * m.embed_bytes_per_row;
            gguf_dequantize(row, m.q_embed_type, m.d_model, emb + (size_t)t * m.d_model);
        }
    }
    float *logits = (float *)malloc((size_t)m.vocab_size * sizeof(float));
    if (!lfm2_forward(&m, emb, 1, T, logits)) { fprintf(stderr, "forward failed\n"); return 1; }
    for (int a = 2; a < dash; a++) {
        int id = atoi(argv[a]);
        if (id >= 0 && id < m.vocab_size) printf("tok %d = %.3f\n", id, logits[id]);
    }
    return 0;
}
