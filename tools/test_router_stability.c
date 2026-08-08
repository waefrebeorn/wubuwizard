/**
 * test_router_stability.c — Measure router sensitivity to input perturbation.
 *
 * Loads only the router weight (ffn_gate_inp, [2048,256] F32 = 2MB), no
 * expert weights needed. Generates random normed/normed2 with controlled
 * perturbation (simulating attn_out = 10-20% of residual), and compares
 * top-8 expert overlap.
 *
 * Usage: ./test_router_stability <layer> [noise_levels]
 *   layer: 0-39 (default 0)
 *   noise_levels: comma-separated percentages (default 5,10,15,20,30,50)
 */
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MODEL_PATH "/home/wubu2/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf"

static int compare_int(const void *a, const void *b) {
    return *(const int *)a - *(const int *)b;
}

// Compute top-k overlap between two sets of indices
static double overlap_pct(const int *a, const int *b, int k) {
    int copy_a[8], copy_b[8];
    memcpy(copy_a, a, k * sizeof(int));
    memcpy(copy_b, b, k * sizeof(int));
    qsort(copy_a, k, sizeof(int), compare_int);
    qsort(copy_b, k, sizeof(int), compare_int);

    int common = 0, i = 0, j = 0;
    while (i < k && j < k) {
        if (copy_a[i] < copy_b[j]) i++;
        else if (copy_a[i] > copy_b[j]) j++;
        else { common++; i++; j++; }
    }
    return (double)common / k * 100.0;
}

// Simple softmax → top-k router (matches wubu_moe_router_only logic)
static void router_on_input(const float *x, const float *gate_inp,
                            int *topk_out, float *all_scores) {
    // x @ gate_inp: [2048] @ [2048, 256] -> [256]
    for (int e = 0; e < N_EXPERTS; e++) {
        float sum = 0.0f;
        for (int i = 0; i < D_MODEL; i++)
            sum += x[i] * gate_inp[i + e * D_MODEL];
        all_scores[e] = sum;
    }

    // Softmax
    float max_s = all_scores[0];
    for (int e = 1; e < N_EXPERTS; e++)
        if (all_scores[e] > max_s) max_s = all_scores[e];
    float sum_exp = 0.0f;
    for (int e = 0; e < N_EXPERTS; e++)
        sum_exp += expf(all_scores[e] - max_s);
    float inv_sum = 1.0f / (sum_exp + 1e-30f);

    float softmax_vals[N_EXPERTS];
    for (int e = 0; e < N_EXPERTS; e++)
        softmax_vals[e] = expf(all_scores[e] - max_s) * inv_sum;

    // Top-8 (worst-first bubble)
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
        topk_out[k] = k;
    }
    for (int i = 0; i < N_ACTIVE_EXPTS-1; i++)
        for (int j = i+1; j < N_ACTIVE_EXPTS; j++)
            if (softmax_vals[topk_out[i]] > softmax_vals[topk_out[j]]) {
                int t = topk_out[i]; topk_out[i] = topk_out[j]; topk_out[j] = t;
            }

    for (int e = N_ACTIVE_EXPTS; e < N_EXPERTS; e++) {
        if (softmax_vals[e] > softmax_vals[topk_out[0]]) {
            topk_out[0] = e;
            int pos = 0;
            while (pos + 1 < N_ACTIVE_EXPTS && softmax_vals[topk_out[pos]] > softmax_vals[topk_out[pos+1]]) {
                int t = topk_out[pos]; topk_out[pos] = topk_out[pos+1]; topk_out[pos+1] = t;
                pos++;
            }
        }
    }
}

int main(int argc, char **argv) {
    int layer = (argc > 1) ? atoi(argv[1]) : 0;
    const char *noise_str = (argc > 2) ? argv[2] : "5,10,15,20,30,50";

    // Parse noise levels
    int n_noise = 0;
    float noise_vals[16];
    char buf[256];
    strncpy(buf, noise_str, sizeof(buf));
    char *tok = strtok(buf, ",");
    while (tok && n_noise < 16) {
        noise_vals[n_noise++] = atof(tok);
        tok = strtok(NULL, ",");
    }

    // Open GGUF — read only the router weight (no data blob needed)
    gguf_ctx *ctx = gguf_open(MODEL_PATH);
    if (!ctx) {
        fprintf(stderr, "Failed to open %s\n", MODEL_PATH);
        return 1;
    }

    char name[256];
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "Missing tensor %s\n", name);
        gguf_close(ctx);
        return 1;
    }
    int64_t n_router = (int64_t)D_MODEL * N_EXPERTS; // 2048*256
    float *gate_inp = (float *)malloc(n_router * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, gate_inp, n_router)) {
        fprintf(stderr, "Failed to read %s\n", name);
        free(gate_inp);
        gguf_close(ctx);
        return 1;
    }
    gguf_close(ctx);

    fprintf(stderr, "Loaded blk.%d.ffn_gate_inp.weight: %ld floats (%.1f MB)\n",
            layer, (long)n_router, (double)n_router * sizeof(float) / 1e6);

    printf("=== Router Stability Test — Layer %d ===\n", layer);
    printf("Router weight: [%d, %d] = %d params (%.1f MB)\n",
           D_MODEL, N_EXPERTS, D_MODEL * N_EXPERTS, 
           (double)D_MODEL * N_EXPERTS * sizeof(float) / 1e6);
    printf("Noise levels: ");
    for (int i = 0; i < n_noise; i++)
        printf("%.0f%%%s", noise_vals[i], i < n_noise - 1 ? ", " : "");
    printf("\n\n");

    srand(time(NULL));
    int N_TRIALS = 200;
    float *x = (float *)malloc(D_MODEL * sizeof(float));

    int idx_normed[N_ACTIVE_EXPTS];
    int idx_noisy[N_ACTIVE_EXPTS];
    float scores1[N_EXPERTS];
    float scores2[N_EXPERTS];

    printf("%-8s  %-18s  %-18s  %-18s\n",
           "Noise%", "Top-8 Overlap %", "Avg Score Δ", "Max Score Δ");
    printf("%-8s  %-18s  %-18s  %-18s\n",
           "-------", "-----------------", "-----------------", "-----------------");

    for (int ni = 0; ni < n_noise; ni++) {
        float noise_frac = noise_vals[ni] / 100.0f;
        double avg_overlap = 0.0;
        double avg_score_diff = 0.0;
        double max_score_diff = 0.0;

        for (int t = 0; t < N_TRIALS; t++) {
            // Generate random normed (unit sphere)
            float norm = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                norm += x[i] * x[i];
            }
            norm = sqrtf(norm);
            for (int i = 0; i < D_MODEL; i++)
                x[i] /= norm;

            // noisy = normed + noise_frac * random direction
            float noisy[D_MODEL];
            float nrm = 0.0f;
            for (int i = 0; i < D_MODEL; i++) {
                noisy[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                nrm += noisy[i] * noisy[i];
            }
            nrm = sqrtf(nrm);
            float scale = noise_frac / nrm;
            for (int i = 0; i < D_MODEL; i++)
                noisy[i] = x[i] + noisy[i] * scale;

            // Run router on both
            router_on_input(x, gate_inp, idx_normed, scores1);
            router_on_input(noisy, gate_inp, idx_noisy, scores2);

            // Overlap
            avg_overlap += overlap_pct(idx_normed, idx_noisy, N_ACTIVE_EXPTS);

            // Score diffs
            float total_diff = 0.0f;
            float max_diff = 0.0f;
            for (int e = 0; e < N_EXPERTS; e++) {
                float d = fabsf(scores1[e] - scores2[e]);
                total_diff += d;
                if (d > max_diff) max_diff = d;
            }
            avg_score_diff += total_diff / N_EXPERTS;
            if (max_diff > max_score_diff) max_score_diff = max_diff;
        }

        avg_overlap /= N_TRIALS;
        avg_score_diff /= N_TRIALS;

        printf("%-8.0f  %-18.1f  %-18.4f  %-18.4f\n",
               noise_vals[ni], avg_overlap, avg_score_diff, max_score_diff);
    }

    free(x);
    free(gate_inp);
    return 0;
}

// unused but kept for reference
