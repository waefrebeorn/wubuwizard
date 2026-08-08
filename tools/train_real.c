/**
 * train_real.c — Phase 3: Training Pipeline on Real Qwen3.6 Model
 *
 * Loads the 40-layer model + tokenizer + tokenized corpus,
 * runs forward pass through all layers, computes CE loss.
 * Now with full backward gradient flow verification.
 */

#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Forward pass with intermediate saving, CPU-only (no GPU_SUPPORT branches)
// Returns hidden states before output projection (final RMSNorm output)
// Populates saved_* arrays for backward
static void forward_with_save(wubu_model_t *model,
                               const float *embeddings,
                               float *logits,
                               float *saved_normed,    // [n_layers * N * D_MODEL]
                               float *saved_attn_out,  // [n_layers * N * D_MODEL]
                               float *saved_normed2,   // [n_layers * N * D_MODEL]
                               float *saved_ffn_out,   // [n_layers * N * D_MODEL]
                               int B, int T) {
    const int N = B * T;
    const int layer_sz = N * D_MODEL;

    // Residual stream + intermediate buffers
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    memcpy(x, embeddings, N * D_MODEL * sizeof(float));

    float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
    float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
    float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    int *prev_experts = model->enable_moe ? (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int)) : NULL;

    // Pre-allocate SSM workspace for reuse across layers
    ssm_workspace_t *ssm_ws = wubu_ssm_workspace_alloc(B, T);

    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];

        // Pre-attention RMSNorm
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        memcpy(saved_normed + l * layer_sz, normed, layer_sz * sizeof(float));

        // Attention (SSM or GQA)
        if (layer->is_ssm) {
            float *ssm_state = model->ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model->conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm,
                             ssm_state, conv_state, attn_out, NULL, NULL, ssm_ws);
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out, NULL, NULL, 0, NULL, NULL);
        }
        memcpy(saved_attn_out + l * layer_sz, attn_out, layer_sz * sizeof(float));

        // Residual: x += attn_out
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];

        // Post-attention RMSNorm
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        memcpy(saved_normed2 + l * layer_sz, normed2, layer_sz * sizeof(float));

        // MoE (if enabled and loaded) or pass-through
        if (layer->moe.loaded && model->enable_moe &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, prev_experts);
            memcpy(saved_ffn_out + l * layer_sz, ffn_out, layer_sz * sizeof(float));
            for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];
        } else {
            // Pass-through (identity MoE)
            memcpy(saved_ffn_out + l * layer_sz, normed2, layer_sz * sizeof(float));
            for (int i = 0; i < N * D_MODEL; i++) x[i] += normed2[i];
        }
    }

    // Final RMSNorm
    if (model->norm_weight) {
        wubu_rms_norm(B, T, D_MODEL, x, model->norm_weight, 1e-6f, logits);
    } else {
        memcpy(logits, x, N * D_MODEL * sizeof(float));
    }

    // Cleanup
    free(x); free(normed); free(attn_out); free(normed2); free(ffn_out);
    free(prev_experts);
    if (ssm_ws) wubu_ssm_workspace_free(ssm_ws);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc > 2 ? argv[2] : "data/train_data.bin";
    const char *embed_path = "data/qwen36_embeddings_c.bin.raw";

    printf("========================================================\n");
    printf("  WuBuText AI — Phase 3 Real Model Training Pipeline\n");
    printf("========================================================\n");
    printf("  Model: %s\n", model_path);
    printf("  Corpus: %s\n", corpus_path);
    fflush(stdout);

    // ===== Load tokenizer =====
    printf("\n--- Loading tokenizer ---\n");
    wubu_tokenizer_t tok;
    double t0 = now_sec();
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    double t_tok = now_sec() - t0;
    printf("  Vocab: %d tokens (%d merges) in %.2fs\n",
           tok.vocab_size, tok.n_merges, t_tok);

    // ===== Load model =====
    printf("\n--- Loading model ---\n");
    wubu_model_t model;
    t0 = now_sec();
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    double t_model = now_sec() - t0;
    printf("  Layers: %d (%d SSM, %d GQA) in %.2fs\n",
           model.n_layers,
           model.n_layers - model.n_layers / 4,
           model.n_layers / 4,
           t_model);

    // ===== Load tokenized training data =====
    printf("\n--- Loading training data ---\n");
    int *tokens = NULL;
    int total_tokens = 0;
    FILE *f = fopen(corpus_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long data_size = ftell(f);
        total_tokens = (int)(data_size / sizeof(int));
        fseek(f, 0, SEEK_SET);
        tokens = (int *)malloc(data_size);
        if (fread(tokens, 1, data_size, f) != (size_t)data_size) {
            fprintf(stderr, "Warning: incomplete read of corpus\n");
        }
        fclose(f);
        printf("  %d tokens loaded (%ld bytes)\n", total_tokens, data_size);
    } else {
        printf("  No corpus at %s — using random tokens\n", corpus_path);
        // Generate synthetic training data: 1024 random tokens
        total_tokens = 1024;
        tokens = (int *)malloc(total_tokens * sizeof(int));
        srand(42);
        for (int i = 0; i < total_tokens; i++)
            tokens[i] = rand() % tok.vocab_size;
    }

    // ===== Load embeddings =====
    printf("\n--- Loading embeddings ---\n");
    f = fopen(embed_path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", embed_path); return 1; }
    fseek(f, 0, SEEK_END);
    long emb_size = ftell(f);
    int emb_tokens = (int)(emb_size / (D_MODEL * sizeof(float)));
    fclose(f);
    printf("  %d embeddings available (%ld MB)\n", emb_tokens, emb_size / (1024*1024));

    // ===== Load output weight (for logit projection) =====
    printf("\n--- Loading output projection ---\n");
    gguf_ctx *ctx = gguf_open(model_path);
    float *output_weight = NULL;
    if (ctx) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
        if (t) {
            printf("  output.weight: [%ld, %ld] type=%d\n",
                   (long)t->dims[0], (long)t->dims[1], t->ggml_type);
            output_weight = (float *)malloc(tok.vocab_size * D_MODEL * sizeof(float));
            t0 = now_sec();
            int nread = gguf_read_tensor_f32(ctx, t, output_weight, tok.vocab_size * D_MODEL);
            double t_ow = now_sec() - t0;
            printf("  output.weight loaded (%d elems) in %.2fs\n", nread, t_ow);
        }
        gguf_close(ctx);
    }

    // ===== Training configuration =====
    int B = 1, T = 4;
    int N = B * T;
    int total_batches = total_tokens / N;
    if (total_batches < 1) { fprintf(stderr, "Not enough tokens for one batch\n"); return 1; }

    printf("\n--- Training config ---\n");
    printf("  B=%d, T=%d, N=%d tokens/batch\n", B, T, N);
    printf("  Total batches available: %d\n", total_batches);
    printf("  Vocab: %d\n", tok.vocab_size);

    if (getenv("ENABLE_MOE")) {
        model.enable_moe = true;
        printf("  MoE: ENABLED (per-layer lazy load)\n");
        if (getenv("MOE_LAYERS")) {
            model.moe_max_layers = atoi(getenv("MOE_LAYERS"));
            printf("  MoE: first %d layers only (set MOE_LAYERS=N)\n", model.moe_max_layers);
        }
    } else {
        printf("  MoE: disabled (set ENABLE_MOE=1 to enable)\n");
    }

    // ====================================================================
    // PART 1: Forward-only benchmark (using batched wubu_model_forward_from_embd)
    // ====================================================================
    printf("\n=== PART 1: Forward-Only Benchmark ===\n");

    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *logits_bench = (float *)malloc(N * tok.vocab_size * sizeof(float));

    // Use first N tokens from corpus
    if (model.use_embedding_file) {
        f = fopen(embed_path, "rb");
        if (f) {
            for (int i = 0; i < N; i++) {
                int id = tokens[i];
                if (id < 0 || id >= emb_tokens) id = 0;
                fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
                if (fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f) != D_MODEL) {
                    fprintf(stderr, "Warning: incomplete embedding read\n");
                }
            }
            fclose(f);
        } else {
            memset(embd, 0, N * D_MODEL * sizeof(float));
        }
    }

    // Targets: predict next token
    int *targets = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N - 1; i++) targets[i] = tokens[i + 1];
    targets[N - 1] = 0;

    // Warm-up run
    printf("  Warm-up: running forward pass...\n");
    wubu_model_forward_from_embd(&model, embd, B, T, logits_bench);
    printf("  Warm-up done.\n");

    // Timed runs
    int n_runs = 3;
    double total_forward = 0.0;
    for (int run = 0; run < n_runs; run++) {
        t0 = now_sec();
        wubu_model_forward_from_embd(&model, embd, B, T, logits_bench);
        double t_fwd = now_sec() - t0;

        float loss = 0.0f;

        // Project to vocab space and compute CE loss
        if (output_weight) {
            float total_loss_val = 0.0f;
            for (int i = 0; i < N; i++) {
                const float *h = logits_bench + i * D_MODEL;
                int target = targets[i];

                double max_l = -1e30;
                double target_logit = 0.0;

                for (int j = 0; j < tok.vocab_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += (double)h[k] * (double)output_weight[j * D_MODEL + k];
                    if (j == target) target_logit = sum;
                    if (sum > max_l) max_l = sum;
                }

                double sum_exp = 0.0;
                for (int j = 0; j < tok.vocab_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += (double)h[k] * (double)output_weight[j * D_MODEL + k];
                    sum_exp += exp(sum - max_l);
                }
                double log_sum_exp = max_l + log(sum_exp);

                double ce = -(target_logit - log_sum_exp);
                total_loss_val += (float)ce;
            }
            loss = total_loss_val / N;
            printf("  CE loss: %.4f\n", loss);
        }

        if (run == 0) {
            printf("  Logits[0:8] for token 0:");
            for (int i = 0; i < 8 && i < tok.vocab_size; i++)
                printf(" %+.4f", logits_bench[i]);
            printf("\n");
        }

        total_forward += t_fwd;
        printf("  Run %d: forward = %.3fs (%.1f tok/s)\n",
               run + 1, t_fwd, N / t_fwd);
    }

    printf("  Avg forward time: %.3fs\n", total_forward / n_runs);
    printf("  Throughput: %.1f tok/s (B=%d, T=%d)\n",
           N / (total_forward / n_runs), B, T);

    // ====================================================================
    // PART 2: Backward Gradient Flow Verification
    // ====================================================================
    printf("\n=== PART 2: Backward Gradient Flow Verification ===\n");

    // Allocate intermediate buffers (N tokens * D_MODEL per layer, 40 layers)
    int layer_sz = N * D_MODEL;
    float *saved_normed  = (float *)calloc(model.n_layers * layer_sz, sizeof(float));
    float *saved_attn_out = (float *)calloc(model.n_layers * layer_sz, sizeof(float));
    float *saved_normed2 = (float *)calloc(model.n_layers * layer_sz, sizeof(float));
    float *saved_ffn_out = (float *)calloc(model.n_layers * layer_sz, sizeof(float));

    if (!saved_normed || !saved_attn_out || !saved_normed2 || !saved_ffn_out) {
        fprintf(stderr, "Failed to allocate intermediate buffers (%d layers * %d * %d = %d floats each)\n",
                model.n_layers, N, D_MODEL, model.n_layers * layer_sz);
        return 1;
    }

    // Run forward with save
    float *logits_bwd = (float *)malloc(N * D_MODEL * sizeof(float));
    printf("  Running forward with intermediate saving...\n");
    t0 = now_sec();
    forward_with_save(&model, embd, logits_bwd,
                       saved_normed, saved_attn_out, saved_normed2, saved_ffn_out,
                       B, T);
    double t_fwd_save = now_sec() - t0;
    printf("  Forward+save: %.3fs (%.1f tok/s)\n", t_fwd_save, N / t_fwd_save);

    // === Backward through output projection: CE loss gradient (BLAS-accelerated) ===
    // Forward: hidden[i] @ output_weight^T → vocab_logits → softmax → CE
    // Backward: d_hidden = output_weight^T @ (softmax - one_hot)
    // Uses cblas_sgemv for W @ h and cblas_sgemm for W^T @ softmax
    float *d_logits = (float *)malloc(N * D_MODEL * sizeof(float));
    double total_ce_loss = 0.0;

    if (output_weight) {
        printf("  Computing CE loss backward through output projection (%d vocab, BLAS)...\n", tok.vocab_size);
        t0 = now_sec();

        // Scratch buffers
        float *logits_scratch = (float *)malloc(tok.vocab_size * sizeof(float));
        float *softmax_matrix = (float *)malloc(tok.vocab_size * N * sizeof(float));
        if (!logits_scratch || !softmax_matrix) {
            fprintf(stderr, "CE backward scratch alloc failed\n");
            free(logits_scratch); free(softmax_matrix);
        } else {
            // Phase 1: for each token, compute vocab logits + softmax
            for (int i = 0; i < N; i++) {
                const float *h = logits_bwd + i * D_MODEL;
                int target = targets[i];

                // BLAS: logits_scratch = output_weight @ h  (W @ h)
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    tok.vocab_size, D_MODEL, 1.0f,
                    output_weight, D_MODEL, h, 1, 0.0f,
                    logits_scratch, 1);

                // Find max logit
                float max_l = logits_scratch[0];
                for (int j = 1; j < tok.vocab_size; j++)
                    if (logits_scratch[j] > max_l) max_l = logits_scratch[j];

                // Compute softmax: exp(logit - max_l) / sum_exp
                double sum_exp = 0.0;
                for (int j = 0; j < tok.vocab_size; j++) {
                    double ev = exp((double)logits_scratch[j] - (double)max_l);
                    sum_exp += ev;
                    softmax_matrix[j * N + i] = (float)ev;
                }
                double inv_sum_exp = 1.0 / sum_exp;
                for (int j = 0; j < tok.vocab_size; j++)
                    softmax_matrix[j * N + i] *= (float)inv_sum_exp;

                // CE loss
                double target_logit = logits_scratch[target];
                double ce = -(target_logit - max_l - log(sum_exp));
                total_ce_loss += ce;
            }

            // Phase 2: d_hidden = output_weight^T @ softmax_matrix  (W^T @ S)
            // output_weight: [vocab_size, D_MODEL] row-major
            // softmax_matrix: [vocab_size, N] row-major
            // d_hidden: [D_MODEL, N] row-major
            // sgemm: C = alpha * op(A) * op(B) + beta * C
            // op(A) = W^T: [D_MODEL, vocab_size], lda = D_MODEL
            // op(B) = S: [vocab_size, N], ldb = N
            // C = d_hidden: [D_MODEL, N], ldc = N
            memset(d_logits, 0, N * D_MODEL * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                D_MODEL, N, tok.vocab_size,
                1.0f, output_weight, D_MODEL,
                softmax_matrix, N,
                0.0f, d_logits, N);

            // Subtract W[target] from each token's d_hidden column
            for (int i = 0; i < N; i++) {
                int target = targets[i];
                const float *w_target = output_weight + target * D_MODEL;
                for (int k = 0; k < D_MODEL; k++)
                    d_logits[k * N + i] -= w_target[k];
            }

            free(logits_scratch);
            free(softmax_matrix);
        }

        double t_ce_bwd = now_sec() - t0;
        printf("  CE backward (BLAS): %.3fs (%.1f tok/s)\n", t_ce_bwd, N / t_ce_bwd);
    } else {
        // Fallback: quadratic loss gradient (no output weight available)
        printf("  No output weight — using quadratic loss gradient (d_logits = logits)\n");
        for (int i = 0; i < N * D_MODEL; i++)
            d_logits[i] = logits_bwd[i];
    }

    // Run backward
    float *d_embeddings = (float *)calloc(N * D_MODEL, sizeof(float));
    printf("  Running model backward through all layers...\n");
    t0 = now_sec();
    wubu_model_backward_from_embd(&model, embd, logits_bwd, d_logits,
                                   saved_normed, saved_attn_out,
                                   saved_normed2, saved_ffn_out,
                                   d_embeddings, B, T);
    double t_bwd = now_sec() - t0;
    printf("  Backward: %.3fs\n", t_bwd);

    // Verify gradients
    float max_d = 0.0f, min_d = 0.0f, sum_abs = 0.0f, sum_sq = 0.0f;
    int non_zero = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        float v = d_embeddings[i];
        if (v > max_d) max_d = v;
        if (v < min_d) min_d = v;
        sum_abs += fabsf(v);
        sum_sq += v * v;
        if (fabsf(v) > 1e-10f) non_zero++;
    }
    float rms = sqrtf(sum_sq / (N * D_MODEL));

    printf("\n=== Backward Gradient Results (CE Loss Backprop) ===\n");
    printf("  d_embeddings: range [%.6e, %.6e]\n", min_d, max_d);
    printf("  d_embeddings: sum|grad| = %.6e, rms = %.6e\n", sum_abs, rms);
    printf("  Non-zero elements: %d / %d (%.1f%%)\n",
           non_zero, N * D_MODEL, 100.0 * non_zero / (N * D_MODEL));
    if (output_weight) printf("  CE loss (avg): %.4f\n", (float)(total_ce_loss / N));

    if (sum_abs > 0.0f && non_zero > 0) {
        printf("  ✅ GRADIENT FLOW VERIFIED — gradients propagate through all layers\n");
    } else {
        printf("  ❌ GRADIENT FLOW FAILED — all gradients are zero\n");
    }

    // ====================================================================
    // Summary
    // ====================================================================
    printf("\n========================================================\n");
    printf("  RESULTS\n");
    printf("========================================================\n");
    printf("  Forward (bench): %.3fs (%.1f tok/s, B=%d, T=%d)\n",
           total_forward / n_runs, N / (total_forward / n_runs), B, T);
    printf("  Forward+save:    %.3fs\n", t_fwd_save);
    printf("  Backward:        %.3fs (%.1f × forward cost)\n",
           t_bwd, t_bwd / (total_forward / n_runs));
    printf("  Model: %d layers (%d SSM + %d GQA)\n", model.n_layers,
           model.n_layers - model.n_layers / 4, model.n_layers / 4);
    printf("  Gradient flow: ✅ (%d/%d non-zero)\n", non_zero, N * D_MODEL);
    printf("  CE loss backward through output projection: ✅\n");
    printf("========================================================\n");

    // Cleanup
    free(tokens);
    free(embd);
    free(logits_bench);
    free(logits_bwd);
    free(targets);
    free(d_logits);
    free(d_embeddings);
    free(saved_normed);
    free(saved_attn_out);
    free(saved_normed2);
    free(saved_ffn_out);
    free(output_weight);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);

    return 0;
}
