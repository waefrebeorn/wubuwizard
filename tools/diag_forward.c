/**
 * diag_forward.c — Run bytropix single-token forward, dump per-layer hidden states
 * Build: gcc -O3 -I include -o diag_forward diag_forward.c \
 *            src/wubu_model_cpu.o src/wubu_moe_cpu.o \
 *            $(filter-out src/wubu_moe.o,$(CORE_OBJ)) \
 *            src/wubu_tokenizer.o -lm -fopenmp -lopenblas -ljson-c
 *
 * Use: MODEL=/path/to/model.gguf ./diag_forward
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#define DUMP_DIR "/tmp/diag_dump"

static double cos_sim_f(const float *a, const float *b, int n) {
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        n1  += (double)a[i] * (double)a[i];
        n2  += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(n1) * sqrt(n2) + 1e-30);
}

static double max_diff_f(const float *a, const float *b, int n) {
    double md = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
    }
    return md;
}

int main(int argc, char **argv) {
    const char *model_path = getenv("MODEL");
    if (!model_path) {
        fprintf(stderr, "Set MODEL env var\n");
        return 1;
    }

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;
    // MoE memory limit
    mdl.moe_max_layers = 0; // all layers

    int D = D_MODEL;
    int token_id = 248044; // BOS
    if (argc > 1) token_id = atoi(argv[1]);

    // Get embedding
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { fprintf(stderr, "No embedding file\n"); return 1; }
        fseek(f, (long long)token_id * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(x, mdl.token_embd + (long long)token_id * D, D * sizeof(float));
    }

    // Save input embedding
    mkdir(DUMP_DIR, 0755);
    FILE *f = fopen(DUMP_DIR "/embd.bin", "wb");
    fwrite(x, sizeof(float), D, f);
    fclose(f);
    
    double n = 0;
    for (int i = 0; i < D; i++) n += (double)x[i] * (double)x[i];
    printf("EMBD: norm=%.4f range=[%.4f,%.4f]\n", sqrt(n), x[0], x[D-1]);

    // Run forward one layer at a time, dumping hidden states
    float *h = (float *)malloc(D * sizeof(float));
    memcpy(h, x, D * sizeof(float));

    // Skip model's built-in layer loop — manually run layers
    // Instead, use wubu_model_forward_from_embd which is the full pipeline
    // But that doesn't expose per-layer states.
    // Let me intercept by modifying it or using the individual layer functions.

    // Approach: run wubu_model_forward_from_embd, then diff the intermediate dumps
    // from the DUMP_LAYER mechanism.
    
    // Use the existing PROFILE mechanism: DUMP_LAYER env var
    printf("Run with: MODEL=... DUMP_LAYER=N ./gen_text_cpu 'token' 1 40\n");
    printf("  to dump per-layer hidden states to /tmp/debug_hidden_before_l.bin\n\n");

    // Also sanity-check: run full forward and dump logits
    float *logits = (float *)malloc(mdl.vocab_size * sizeof(float));
    mdl.skip_output_proj = false;
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Check logits
    int nan_cnt = 0, inf_cnt = 0;
    float minv = 1e30f, maxv = -1e30f;
    double sum = 0, sum2 = 0;
    for (int i = 0; i < mdl.vocab_size; i++) {
        if (isnan(logits[i])) nan_cnt++;
        if (isinf(logits[i])) inf_cnt++;
        if (logits[i] < minv) minv = logits[i];
        if (logits[i] > maxv) maxv = logits[i];
        sum += logits[i];
        sum2 += (double)logits[i] * (double)logits[i];
    }
    double mean = sum / mdl.vocab_size;
    double var = sum2 / mdl.vocab_size - mean * mean;
    printf("LOGITS: range=[%.4f,%.4f] mean=%.6f var=%.6f nan=%d inf=%d\n",
           minv, maxv, mean, var, nan_cnt, inf_cnt);

    // Top-10
    float tv[10]; int top[10];
    for (int k = 0; k < 10; k++) tv[k] = -1e30f;
    for (int i = 0; i < mdl.vocab_size; i++) {
        if (logits[i] > tv[9]) {
            tv[9] = logits[i]; top[9] = i;
            for (int k = 8; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float t = tv[k]; tv[k] = tv[k+1]; tv[k+1] = t;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                }
            }
        }
    }
    printf("TOP-10:\n");
    for (int k = 0; k < 10; k++)
        printf("  [%d] = %.4f\n", top[k], (double)tv[k]);

    // Save logits
    f = fopen(DUMP_DIR "/logits.bin", "wb");
    fwrite(logits, sizeof(float), mdl.vocab_size, f);
    fclose(f);
    printf("Logits saved to " DUMP_DIR "/logits.bin\n");

    free(logits);
    free(h);
    free(x);
    wubu_model_free(&mdl);
    return 0;
}
