/* inkling_gen -- greedy generation.
 *
 * Incremental (KV-cached) decode by default, ~6x faster than recomputing the prefix. It is bit-identical to the full
 * recompute path on every prompt tested (max|diff| = 0.0), which is what promoted it; IN2_NO_KV=1 forces the
 * full-recompute path for A/B checking of any cache change.
 *
 * usage: inkling_gen <n_gen> <token> [token ...]
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"
#include "wubu_gguf_tokenizer.h"
#include "wubu_tokenizer.h"

int in2_forward_seq(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                    const int *, int, float *, float *);
int in2_forward_at(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                   const int *, int, int, float *, float *);

/* SentencePiece marks a leading space with U+2581, byte-level BPE with U+0120; a byte-fallback token is <0xNN>.
 * Same rules as tools/detok.c so the text a run prints is the text detok would print for the same ids. */
static void emit_piece(const char *s)
{
    if (!s) return;
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
        char *end = NULL;
        const long b = strtol(s + 3, &end, 16);
        if (end && *end == '>' && b >= 0 && b < 256) { putchar((int)b); return; }
    }
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { putchar(' '); p += 3; }   /* U+2581 SentencePiece */
        else if (p[0] == 0xC4 && p[1] == 0xA0) { putchar(' '); p += 2; }                /* U+0120 byte-level BPE */
        else if (p[0] == 0xC4 && p[1] == 0x8A) { putchar('\n'); p += 2; }      /* U+010A byte-level newline */
        else if (p[0] == 0xC4 && p[1] == 0x89) { putchar('\t'); p += 2; }      /* U+0109 byte-level tab */
        else { putchar(*p++); }
    }
}

int main(int argc, char **argv)
{
    const char *path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    if (argc < 3) { fprintf(stderr, "usage: %s <n_gen> <token> [token ...]\n", argv[0]); return 2; }
    const int n_gen = atoi(argv[1]);

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    if (getenv("IN2_CFG"))
        fprintf(stderr, "CFG d=%d layers=%d core=[%d,%d) loops=%d n_expert=%d used=%d d_ff=%d d_ff_expert=%d vocab=%d\n",
                cfg.d_model, cfg.n_layer, cfg.loop_core_start, cfg.loop_core_end, cfg.n_loops,
                cfg.n_expert, cfg.n_expert_used, cfg.d_ff, cfg.d_ff_expert, cfg.vocab_size);
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    /* Two ways in: a list of token ids, or --text and a prompt encoded with the .gguf's own tokenizer. The ids a text
     * prompt produces are the ones llama.cpp's /tokenize returns, so the tokenizer is testable against the oracle
     * instead of assumed. */
    const int text_mode = (argc >= 4 && strcmp(argv[2], "--text") == 0);
    int n_prompt = text_mode ? 0 : argc - 2;
    int *toks = NULL;
    wubu_tokenizer_t tok_text;
    memset(&tok_text, 0, sizeof tok_text);
    if (text_mode) {
        if (!wubu_tokenizer_init_from_gguf(&tok_text, path)) { fprintf(stderr, "tokenizer init failed\n"); return 5; }
        const int cap0 = 4096;
        toks = (int *)calloc((size_t)cap0 + (size_t)n_gen, sizeof(int));
        if (!toks) return 6;
        n_prompt = wubu_tokenizer_encode(&tok_text, argv[3], toks, cap0);
        if (n_prompt <= 0) { fprintf(stderr, "prompt encode failed (%d)\n", n_prompt); return 7; }
        for (int i = 0; i < n_prompt; ++i) printf("tok[%d]=%d\n", i, toks[i]);
    }
    const int cap = n_prompt + n_gen;
    if (!text_mode) {
        toks = (int *)calloc((size_t)cap, sizeof(int));
        for (int i = 0; i < n_prompt; ++i) toks[i] = atoi(argv[2 + i]);
    }
    for (int i = 0; i < n_prompt; ++i) printf("prompt[%d]=%d\n", i, toks[i]);

    clock_t t_step = clock(); double t_total = 0.0; int used_cache = 0;
    float *h = (float *)calloc((size_t)cap * cfg.d_model, sizeof(float));
    float *logits = (float *)calloc(cfg.vocab_size, sizeof(float));

    for (int step = 0; step < n_gen; ++step) {
        const clock_t t_step0 = clock(); (void)t_step0;
        const int n = n_prompt + step;
        /* With a cache the prompt is prefilled once and each step pushes a single token at its real position;
         * without one the whole prefix is recomputed, which is the O(n^2) behaviour this is meant to replace.
         * Cache on by default now: the full-recompute path was the safe default while the cache could hide a bug, but
         * 34/34 oracle tokens match with it enabled, and the two paths produce bit-identical hidden states. IN2_NO_KV
         * forces the full recompute for debugging. */
        const int incremental = getenv("IN2_NO_KV") == NULL && step > 0;
        const int rc = incremental
            ? in2_forward_at(ctx, &m, &cfg, &toks[n - 1], 1, n - 1, h, logits)
            : in2_forward_at(ctx, &m, &cfg, toks, n, 0, h, logits);
        if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "step %d rc=%d\n", step, rc); return rc; }
        used_cache |= incremental;
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        toks[n] = best;
        const double dt = (double)(clock() - t_step) / (double)CLOCKS_PER_SEC;
        t_step = clock();
        t_total += dt;
        printf("gen[%d] id=%d logit=%.4f  %.1fs  (%.2f tok/s at prefix %d)\n",
               step, best, bv, dt, dt > 0 ? 1.0 / dt : 0.0, n);
        if (getenv("IN2_TOPK")) {
            /* Top-k of the same distribution, so a near-tie can be judged against llama-server's
             * completion_probabilities instead of only by the argmax. */
            const int K = atoi(getenv("IN2_TOPK"));
            static int used[64];
            for (int r = 0; r < K && r < 64; ++r) {
                int bi = -1; float bvv = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) {
                    int skip = 0;
                    for (int q = 0; q < r; ++q) if (used[q] == i) { skip = 1; break; }
                    if (!skip && logits[i] > bvv) { bvv = logits[i]; bi = i; }
                }
                used[r] = bi;
                fprintf(stderr, "TOPK %d %d %.4f\n", r, bi, bvv);
            }
        }
        fflush(stdout);
    }
    if (n_gen > 0 && t_total > 0)
        fprintf(stderr, "TIMING: %d tokens in %.1fs = %.2f tok/s (%s)\n",
                n_gen, t_total, (double)n_gen / t_total, used_cache ? "KV cache" : "full recompute");
    /* The ids alone do not say whether the output is coherent, which is the whole question a generation run is
     * asking. Decode with the vocab inside the same .gguf, in detok's rules, so a human can judge the result here. */
    {
        wubu_gguf_tokenizer_data_t tk;
        memset(&tk, 0, sizeof tk);
        if (wubu_gguf_tokenizer_extract(path, &tk)) {
            printf("GEN_TEXT: ");
            for (int i = n_prompt; i < n_prompt + n_gen; ++i)
                if (toks[i] >= 0 && toks[i] < tk.n_tokens) emit_piece(tk.tokens[toks[i]]);
            printf("\n");
            printf("FULL_TEXT: ");
            for (int i = 0; i < n_prompt + n_gen; ++i)
                if (toks[i] >= 0 && toks[i] < tk.n_tokens) emit_piece(tk.tokens[toks[i]]);
            printf("\n");
            wubu_gguf_tokenizer_free(&tk);
        } else {
            fprintf(stderr, "tokenizer extract failed; only ids available\n");
        }
    }
    printf("IDS:");
    for (int i = 0; i < n_prompt + n_gen; ++i) printf(" %d", toks[i]);
    printf("\n");
    free(h); free(logits); free(toks); gguf_close(ctx);
    return 0;
}
