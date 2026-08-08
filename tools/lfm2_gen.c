/* lfm2_gen.c -- autoregressive generation for LFM2.5 (backup brain).
 * Reads an initial token-ID sequence on argv, runs lfm2_forward with
 * temperature + top-p (nucleus) sampling for N steps, prints token IDs.
 * Tokenization is handled by a Python glue using LFM2.5's tokenizer.json.
 * Self-contained: depends only on wubu_lfm2.h. */
#include "wubu_lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <lfm2_dir> <maxtokens> [temp] [topp] [tok0 ...]\n", argv[0]); return 1; }
    srand((unsigned)time(NULL));
    const char *dir = argv[1];
    int max_new = atoi(argv[2]);
    float temp = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    float topp = (argc > 4) ? (float)atof(argv[4]) : 0.95f;
    int arg0 = 5;

    lfm2_model_t m;
    if (!lfm2_load(dir, &m)) { fprintf(stderr, "lfm2: load failed\n"); return 1; }

    int *prompt = NULL; int np = 0;
    for (int i = arg0; i < argc; i++) { prompt = realloc(prompt, (np+1)*sizeof(int)); prompt[np++] = atoi(argv[i]); }
    if (np == 0) { fprintf(stderr, "need >=1 seed token\n"); return 1; }

    int cap = np + max_new + 1;
    int *seq = (int *)malloc(cap * sizeof(int));
    for (int i = 0; i < np; i++) seq[i] = prompt[i];
    int T = np;

    float *emb = (float *)malloc((size_t)m.d_model * cap * sizeof(float));
    float *logits = (float *)malloc(m.vocab_size * sizeof(float));

    printf("<lfm2_gen start d=%d layers=%d vocab=%d max_new=%d temp=%.2f topp=%.2f>\n",
           m.d_model, m.n_layers, m.vocab_size, max_new, temp, topp);
    fflush(stdout);

    int produced = 0;
    for (int step = 0; step < max_new; step++) {
        for (int t = 0; t < T; t++) {
            if (m.embed) {
                const float *row = m.embed + (size_t)seq[t] * m.d_model;
                memcpy(emb + (size_t)t * m.d_model, row, m.d_model * sizeof(float));
            } else if (m.q_embed) {
                /* quantized embed: dequantize ONE row per token */
                const uint8_t *row = m.q_embed + (size_t)seq[t] * m.embed_bytes_per_row;
                gguf_dequantize(row, m.q_embed_type, m.d_model,
                                emb + (size_t)t * m.d_model);
            }
        }
        if (!lfm2_forward(&m, emb, 1, T, logits)) { fprintf(stderr, "lfm2: forward failed at step %d\n", step); break; }

        int nan = 0;
        for (int i = 0; i < m.vocab_size; i++) {
            float v = logits[i];
            if (isnan(v) || isinf(v)) { nan++; logits[i] = -1e30f; }
        }
        if (nan) { fprintf(stderr, "lfm2: %d nan logits at step %d\n", nan, step); break; }

        int sample;
        if (temp <= 0.0001f) {
            int am = 0; float mx = -1e30f;
            for (int i = 0; i < m.vocab_size; i++) if (logits[i] > mx) { mx = logits[i]; am = i; }
            sample = am;
        } else {
            float maxv = -1e30f; for (int i = 0; i < m.vocab_size; i++) if (logits[i] > maxv) maxv = logits[i];
            float sum = 0.0f;
            for (int i = 0; i < m.vocab_size; i++) { float e = expf((logits[i]-maxv)/temp); logits[i] = e; sum += e; }
            for (int i = 0; i < m.vocab_size; i++) logits[i] /= sum;
            float r = (float)rand() / (float)RAND_MAX;
            float cdf = 0.0f; sample = 0;
            for (int i = 0; i < m.vocab_size; i++) {
                cdf += logits[i];
                if (cdf >= r) { sample = i; break; }
            }
        }

        printf("T%d:%d ", T, sample);
        fflush(stdout);
        seq[T] = sample;
        T++;
        produced++;
        if (sample == 0) break;
    }
    printf("\n<lfm2_gen done produced=%d total_T=%d>\n", produced, T);
    fflush(stdout);

    lfm2_free(&m);
    free(prompt); free(seq); free(emb); free(logits);
    return 0;
}
