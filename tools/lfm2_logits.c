/* lfm2_logits.c — dump top-k logits for a prompt using lfm2 forward.
 * Usage: lfm2_logits <model.gguf> <tok0> [tok1 ...]
 * Token ids are passed as integers (byte-level BPE ids from the vocab). */
#include "wubu_lfm2.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <tok>...\n", argv[0]); return 1; }
    lfm2_model_t m;
    memset(&m, 0, sizeof(m));
    if (!lfm2_load(argv[1], &m)) { fprintf(stderr, "load failed\n"); return 1; }
    int T = argc - 2;
    float *emb = (float *)malloc((size_t)T * m.d_model * sizeof(float));
    for (int t = 0; t < T; t++) {
        int id = atoi(argv[2 + t]);
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
    /* top-k */
    int k = 12;
    int *idx = (int *)malloc((size_t)m.vocab_size * sizeof(int));
    for (int i = 0; i < m.vocab_size; i++) idx[i] = i;
    for (int a = 0; a < k; a++) {
        int best = a;
        for (int i = a + 1; i < m.vocab_size; i++)
            if (logits[idx[i]] > logits[idx[best]]) best = i;
        int t = idx[a]; idx[a] = idx[best]; idx[best] = t;
        printf("  [%d]=%.2f\n", idx[a], logits[idx[a]]);
    }
    return 0;
}
