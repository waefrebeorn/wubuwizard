/**
 * delta_sparsity.c — Measure hidden state delta sparsity between consecutive tokens.
 * If delta is sparse, we can compute delta @ W (cheap) instead of h_curr @ W (expensive).
 *
 * Usage: MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./delta_sparsity "prompt" 20
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *model_path = getenv("MODEL");
    if (!model_path) { fprintf(stderr, "MODEL required\n"); return 1; }
    const char *prompt = argc > 1 ? argv[1] : "The capital of France is";
    int n_gen = argc > 2 ? atoi(argv[2]) : 20;

    wubu_model_t model;
    memset(&model, 0, sizeof(model));
    if (!wubu_model_init(&model, model_path)) return 1;
    model.enable_moe = true;
    int vs = model.vocab_size;

    // Allocate two hidden state buffers
    float *h_prev = (float *)malloc(D_MODEL * sizeof(float));
    float *h_curr = (float *)malloc(D_MODEL * sizeof(float));
    memset(h_prev, 0, D_MODEL * sizeof(float));

    int tokens[n_gen + 1];
    tokens[0] = model.vocab_size - 4;
    float *logits = (float *)malloc(vs * sizeof(float));

    printf("Prompt: \"%s\", %d tokens\n", prompt, n_gen);
    printf("=== Hidden State Delta Sparsity ===\n");
    printf("Token|Sparsity(0.01)|Sparsity(0.05)|Sparsity(0.10)|Max|Delta|RMS|Delta\n");

    for (int i = 0; i < n_gen; i++) {
        // Forward without output proj to capture h
        model.skip_output_proj = true;
        model.save_last_hidden = h_curr;
        wubu_model_forward(&model, &tokens[i], 1, 1, logits);
        model.skip_output_proj = false;
        
        // Get next token via full output proj
        wubu_model_forward(&model, &tokens[i], 1, 1, logits);
        int best = 0;
        for (int j = 1; j < vs; j++)
            if (logits[j] > logits[best]) best = j;
        tokens[i + 1] = best;

        if (i > 0) {
            // Compute delta stats
            float max_delta = 0, rms_delta = 0;
            int count_001 = 0, count_005 = 0, count_010 = 0;
            float rms_h = 0;
            for (int d = 0; d < D_MODEL; d++) {
                float delta = fabsf(h_curr[d] - h_prev[d]);
                float hv = fabsf(h_curr[d]);
                if (delta > max_delta) max_delta = delta;
                rms_delta += delta * delta;
                rms_h += hv * hv;
                if (delta > 0.01f * (hv + 0.001f)) count_001++;
                if (delta > 0.05f * (hv + 0.001f)) count_005++;
                if (delta > 0.10f * (hv + 0.001f)) count_010++;
            }
            rms_delta = sqrtf(rms_delta / D_MODEL);
            rms_h = sqrtf(rms_h / D_MODEL);
            double sp_001 = 100.0 * count_001 / D_MODEL;
            double sp_005 = 100.0 * count_005 / D_MODEL;
            double sp_010 = 100.0 * count_010 / D_MODEL;

            printf("%5d|%7.1f%%|%7.1f%%|%7.1f%%|%.4f|%.4f|%.4f\n",
                   i, sp_001, sp_005, sp_010, max_delta, rms_delta, rms_h);
        }

        memcpy(h_prev, h_curr, D_MODEL * sizeof(float));
    }

    free(h_prev);
    free(h_curr);
    free(logits);
    wubu_model_free(&model);
    return 0;
}
