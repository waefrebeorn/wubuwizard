/**
 * profile_experts.c — Profile expert selection stability.
 * Measures: (1) overlap from token N to N+1, (2) expert usage frequency.
 *
 * Usage: MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./profile_experts "prompt" 50
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LAYERS 40
static int recorder_buf[MAX_LAYERS * MAX_EXPERT_RECORDER_TOKENS][N_ACTIVE_EXPTS];

static int overlap_count(const int *a, const int *b) {
    int count = 0;
    for (int i = 0; i < N_ACTIVE_EXPTS; i++)
        for (int j = 0; j < N_ACTIVE_EXPTS; j++)
            if (a[i] == b[j]) { count++; break; }
    return count;
}

int main(int argc, char **argv) {
    const char *model_path = getenv("MODEL");
    if (!model_path) { fprintf(stderr, "MODEL env var required\n"); return 1; }
    const char *prompt = argc > 1 ? argv[1] : "The capital of France is";
    int n_gen = argc > 2 ? atoi(argv[2]) : 30;
    if (n_gen > MAX_EXPERT_RECORDER_TOKENS) n_gen = MAX_EXPERT_RECORDER_TOKENS;

    wubu_model_t model;
    memset(&model, 0, sizeof(model));
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Model init failed\n"); return 1;
    }
    model.enable_moe = true;
    int n_layers = model.n_layers;
    model.expert_recorder = recorder_buf;
    model.expert_recorder_tokens = 0;

    int tokens[MAX_EXPERT_RECORDER_TOKENS + 1];
    tokens[0] = model.vocab_size - 4;
    float *logits = (float *)malloc(model.vocab_size * sizeof(float));

    printf("Prompt: \"%s\", %d tokens, %d layers\n", prompt, n_gen, n_layers);

    double t0 = clock();
    for (int i = 0; i < n_gen; i++) {
        wubu_model_forward(&model, &tokens[i], 1, 1, logits);
        int best = 0;
        float bv = logits[0];
        for (int j = 1; j < model.vocab_size; j++)
            if (logits[j] > bv) { bv = logits[j]; best = j; }
        tokens[i + 1] = best;
    }
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("Done: %d tokens in %.1fs (%.1f tok/s)\n", n_gen, dt, n_gen / dt);

    // ---- Per-layer stability ----
    printf("\n=== Per-Layer Stability (token N->N+1 overlap) ===\n");
    printf("Layer|Type|Overlap/8|Samples\n");
    double global_avg = 0;
    for (int l = 0; l < n_layers; l++) {
        int total = 0, np = 0;
        for (int t = 0; t < model.expert_recorder_tokens - 1 && t < n_gen - 1; t++) {
            total += overlap_count(
                recorder_buf[l * MAX_EXPERT_RECORDER_TOKENS + t],
                recorder_buf[l * MAX_EXPERT_RECORDER_TOKENS + (t + 1)]);
            np++;
        }
        double avg = np > 0 ? (double)total / np : 0;
        global_avg += avg;
        printf("%5d|%s|%.2f|%d\n", l, model.layers[l].is_ssm?"SSM":"GQA", avg, np);
    }
    printf("  AVG: %.2f/8 across %d layers\n", global_avg / n_layers, n_layers);

    // ---- Expert usage frequency ----
    printf("\n=== Expert Usage Frequency ===\n");
    int expert_freq[N_EXPERTS] = {0};
    int total_sel = 0;
    for (int l = 0; l < n_layers; l++)
        for (int t = 0; t < model.expert_recorder_tokens && t < n_gen; t++) {
            int idx = l * MAX_EXPERT_RECORDER_TOKENS + t;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = recorder_buf[idx][k];
                if (e >= 0 && e < N_EXPERTS) { expert_freq[e]++; total_sel++; }
            }
        }

    // Sort
    typedef struct { int id; int freq; } ef_t;
    ef_t sorted[N_EXPERTS];
    for (int i = 0; i < N_EXPERTS; i++) { sorted[i].id = i; sorted[i].freq = expert_freq[i]; }
    for (int i = 0; i < N_EXPERTS; i++)
        for (int j = i+1; j < N_EXPERTS; j++)
            if (sorted[j].freq > sorted[i].freq) {
                ef_t tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }

    printf("Top-10:\n");
    for (int k = 0; k < 10; k++)
        printf("  #%d: Expert %d (%d times, %.2f%%)\n", k+1, sorted[k].id,
               sorted[k].freq, 100.0*sorted[k].freq/total_sel);

    int cum = 0, n50=0, n80=0, n95=0;
    for (int k = 0; k < N_EXPERTS; k++) {
        cum += sorted[k].freq;
        double pct = 100.0*cum/total_sel;
        if (!n50 && pct>=50) n50 = k+1;
        if (!n80 && pct>=80) n80 = k+1;
        if (!n95 && pct>=95) n95 = k+1;
    }
    printf("Coverage: top %d=50%%, top %d=80%%, top %d=95%%\n", n50, n80, n95);
    int zero = 0;
    for (int i = 0; i < N_EXPERTS; i++) if (expert_freq[i]==0) zero++;
    printf("Never used: %d/%d (%.0f%%)\n", zero, N_EXPERTS, 100.0*zero/N_EXPERTS);

    // ---- Per-layer pruning analysis ----
    printf("\n=== Per-Layer Pruning Analysis ===\n");
    printf("Layer|Type|Top-10%%|Top-25%%|Top-50%%\n");
    for (int l = 0; l < n_layers && l < 10; l++) {
        int layer_freq[N_EXPERTS] = {0};
        int layer_sel = 0;
        for (int t = 0; t < model.expert_recorder_tokens && t < n_gen; t++) {
            int idx = l * MAX_EXPERT_RECORDER_TOKENS + t;
            for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                int e = recorder_buf[idx][k];
                if (e >= 0 && e < N_EXPERTS) { layer_freq[e]++; layer_sel++; }
            }
        }
        ef_t lsorted[N_EXPERTS];
        for (int i = 0; i < N_EXPERTS; i++) { lsorted[i].id = i; lsorted[i].freq = layer_freq[i]; }
        for (int i = 0; i < N_EXPERTS; i++)
            for (int j = i+1; j < N_EXPERTS; j++)
                if (lsorted[j].freq > lsorted[i].freq) {
                    ef_t tmp = lsorted[i]; lsorted[i] = lsorted[j]; lsorted[j] = tmp;
                }
        int lcum = 0, l10=0, l25=0, l50=0;
        for (int k = 0; k < N_EXPERTS; k++) {
            lcum += lsorted[k].freq;
            double pct = 100.0*lcum/layer_sel;
            if (!l10 && pct>=10) l10 = k+1;
            if (!l25 && pct>=25) l25 = k+1;
            if (!l50 && pct>=50) l50 = k+1;
        }
        printf("%5d|%s|%d|%d|%d\n", l, model.layers[l].is_ssm?"SSM":"GQA", l10, l25, l50);
    }

    // ---- Assessment ----
    double overall = global_avg / n_layers;
    printf("\n=== DEMOSCENE ASSESSMENT ===\n");
    printf("Stability: %.2f/8 (%.0f%% experts same from token N to N+1)\n",
           overall, 100.0*overall/8);
    if (overall >= 5.0)
        printf("HIGH: hash-predict viable. Prefetch during SSM.\n");
    else if (overall >= 3.5)
        printf("MODERATE: hybrid prefetch (predict 4, router handles 4).\n");
    else
        printf("LOW: focus on pruning/expert-reduction instead.\n");

    printf("Pruning potential: %d/%d experts cover 95%% of usage.\n", n95, N_EXPERTS);
    printf("If we prune to %d experts: model shrinks ", n95);
    printf("from ~10.7GB to ~%.1fGB (%.0f%% reduction)\n",
           10.7 * n95 / N_EXPERTS, 100.0 - 100.0 * n95 / N_EXPERTS);

    free(logits);
    wubu_model_free(&model);
    return 0;
}
