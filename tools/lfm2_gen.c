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
        /* incremental decode: the first call pre-fills the whole prompt
         * (T=np), every later call feeds ONE new token at kv_len — the
         * attention reads the KV cache, the conv uses its saved state. */
        int feed_t = (step == 0) ? T : 1;
        int feed_idx = (step == 0) ? 0 : (T - 1);   /* the last emitted token */
        for (int t = 0; t < feed_t; t++) {
            int tok = seq[feed_idx + t];
            if (m.embed) {
                const float *row = m.embed + (size_t)tok * m.d_model;
                memcpy(emb + (size_t)t * m.d_model, row, m.d_model * sizeof(float));
            } else if (m.q_embed) {
                /* quantized embed: dequantize ONE row per token */
                const uint8_t *row = m.q_embed + (size_t)tok * m.embed_bytes_per_row;
                gguf_dequantize(row, m.q_embed_type, m.d_model,
                                emb + (size_t)t * m.d_model);
            }
        }
        if (!lfm2_forward(&m, emb, 1, feed_t, logits)) { fprintf(stderr, "lfm2: forward failed at step %d\n", step); break; }
        seq = realloc(seq, (T + 2) * sizeof(int));   /* room for the new token */

        if (getenv("LFM2_TOPLOGITS") && step == 0) {
            int top[5]; float tv[5];
            for (int i = 0; i < 5; i++) { top[i] = -1; tv[i] = -1e30f; }
            for (int i = 0; i < m.vocab_size; i++) {
                for (int j = 0; j < 5; j++) {
                    if (logits[i] > tv[j]) {
                        for (int k = 4; k > j; k--) { top[k] = top[k-1]; tv[k] = tv[k-1]; }
                        top[j] = i; tv[j] = logits[i]; break;
                    }
                }
            }
            fprintf(stderr, "TOP5: ");
            for (int j = 0; j < 5; j++) fprintf(stderr, "%d:%.4f ", top[j], tv[j]);
            fprintf(stderr, "\n");
        }

        int nan = 0;
        for (int i = 0; i < m.vocab_size; i++) {
            float v = logits[i];
            if (v != v || v > 1e30f || v < -1e30f) { nan = 1; break; }
        }
        if (nan) { fprintf(stderr, "lfm2: NaN in logits at step %d\n", step); break; }

        /* repetition penalty (log-space): damp the last REP_WIN tokens so
         * the LFM2.5's greedy/sampling loops ("(((((" / "PeriodPeriod")
         * break. The model's degeneration margin is large — llama.cpp's
         * default ~0.1 log is useless here; 1.0 reliably escalates out. */
        {
            float rep_pen = 1.0f;
            const char *rp = getenv("LFM2_REP_PEN");
            if (rp) rep_pen = (float)atof(rp);
            if (rep_pen > 0.0f && T > 1) {
                int from = (T > 64) ? T - 64 : 0;
                for (int i = from; i < T; i++) logits[seq[i]] -= rep_pen;
            }
        }

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
