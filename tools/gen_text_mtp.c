/**
 * gen_text_mtp.c — MTP speculative decode with verification.
 * DRAFT_N=2 (blog: 83% acceptance at 2, 50% at 4). Max 2 recommended.
 *
 * LOADS TWO MODEL FILES:
 *   Main: main GGUF (regular model, layers 0-39)
 *   MTP:  MTP GGUF (blk.40 head only — streams from file, no blob)
 *
 * These are DIFFERENT quantizations — base weights are only in the regular file.
 *
 * Flow per iteration:
 *   1. Emit main model's prediction (argmax of last_logits)
 *   2. Generate 2 draft tokens via MTP head (blk.40 from MTP model)
 *   3. Checkpoint model state
 *   4. Forward main_token through main -> verify_logits
 *   5. Check draft[1] against main's prediction:
 *      - MATCH: emit draft[1] (2 tokens for 1 main forward)
 *      - MISMATCH: rollback, emit main's real prediction instead
 *   6. Update h_39 + last_logits for next iteration
 *
 * Build: make gen_text_mtp
 * Usage: MTP=1 OMP_NUM_THREADS=16 ./gen_text_mtp "Hello" 64
 * Without MTP: behaves like gen_text (1 token per step)
 * With MTP: 2-draft speculative decode with accept/reject
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#define DRAFT_N 2

static volatile int g_stop = 0;
static void handle_sigint(int sig) { (void)sig; g_stop = 1; }
static double wall_clock(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int argmax(float *logits, int vs) {
    int mi = 0; float mv = logits[0];
    for (int i = 1; i < vs; i++) if (logits[i] > mv) { mv = logits[i]; mi = i; }
    return mi;
}

static void get_embd(wubu_model_t *mdl, int token, float *out, FILE *emb_file) {
    int D = D_MODEL;
    if (mdl->use_embedding_file && emb_file) {
        fseek(emb_file, (long long)token * D * sizeof(float), SEEK_SET);
        size_t n = fread(out, sizeof(float), D, emb_file);
        if (n != (size_t)D) memset(out, 0, D * sizeof(float));
    } else if (mdl->token_embd) {
        memcpy(out, mdl->token_embd + (long long)token * D, D * sizeof(float));
    } else {
        memset(out, 0, D * sizeof(float));
    }
}

int main(int argc, char **argv) {
    const char *main_model = "/home/wubu2/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf";
    const char *mtp_model  = "/home/wubu2/models/qwen3.6-35b-a3b-MTP-UD-IQ2_M.gguf";
    const char *env_mm = getenv("MODEL");
    const char *env_mtp = getenv("MTP_MODEL");
    if (env_mm) main_model = env_mm;
    if (env_mtp) mtp_model = env_mtp;
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int D = D_MODEL;
    int use_mtp = getenv("MTP") != NULL;
    int verbose = getenv("VERBOSE") != NULL;

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    // ====== Load MAIN model (regular GGUF, layers 0-39) ======
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, main_model)) return 1;
    mdl.enable_moe = true;

    // ====== Load MTP head from MTP model file (no blob — stream from file) ======
    gguf_ctx *mtp_ctx = NULL;
    if (use_mtp) {
        mtp_ctx = gguf_open(mtp_model);
        if (!mtp_ctx) {
            fprintf(stderr, "Failed to open MTP model: %s\n", mtp_model);
            use_mtp = 0;
        } else {
            fprintf(stderr, "Opened MTP model: %s (no full buffer — stream from file)\n", mtp_model);
            if (!wubu_mtp_load(&mdl.mtp, mtp_model, mtp_ctx, NULL)) {
                fprintf(stderr, "MTP head not available in MTP model file\n");
                gguf_close(mtp_ctx);
                mtp_ctx = NULL;
                use_mtp = 0;
            } else {
                fprintf(stderr, "MTP head loaded (%d draft tokens, file-streaming mode)\n", DRAFT_N);
            }
        }
    }

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, main_model)) { wubu_model_free(&mdl); return 1; }

    int vs = mdl.vocab_size;
    int prompt_tokens[1024];
    int n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
    if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);

    // Embeddings for prompt
    float *embd = (float *)malloc(n_prompt * D * sizeof(float));
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) { free(embd); wubu_model_free(&mdl); return 1; }
    }
    for (int i = 0; i < n_prompt; i++)
        get_embd(&mdl, prompt_tokens[i], embd + i * D, emb_file);

    // Prefill: get logits + h_39 via save_last_hidden
    float *logits_buf = (float *)malloc(n_prompt * vs * sizeof(float));
    float h_39[D];
    double t0 = wall_clock();
    mdl.save_last_hidden = h_39;
    wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits_buf);
    mdl.save_last_hidden = NULL;
    double t_prefill = wall_clock() - t0;
    fprintf(stderr, "Prefill: %.2fs (%.1f tok/s)\n", t_prefill, n_prompt / t_prefill);

    // Print prompt
    { char buf[2048]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 2048);
      if (nc > 0) fprintf(stderr, "Input: %s\n", buf); }

    // Stack-allocate the working buffers (D_MODEL=2048, vs=248320 — logits are too big for stack)
    float *logits_out = (float *)malloc(vs * sizeof(float));
    float *cur = (float *)malloc(D * sizeof(float));
    float *mtp_logits = (float *)malloc(vs * sizeof(float));
    float logit_correction[256] = {0};  // EMA correction for first 256 logits
    int correction_n = 0;  // count of EMA updates, for adaptive decay
    float prev_cur[D];
    int total_gen = 0;
    int total_accepted = 0;
    int total_attempted = 0;

    // After prefill: last_logits predicts first generated token
    // prev_cur = embedding of the LAST PROMPT token (for MTP's first draft)
    float *last_logits = logits_buf + (n_prompt - 1) * vs;
    get_embd(&mdl, prompt_tokens[n_prompt - 1], prev_cur, emb_file);
    memcpy(cur, prev_cur, D * sizeof(float));

    // Bootstrap MTP EMA correction using prefill data
    if (use_mtp && mdl.mtp.loaded && n_prompt > 1) {
        fprintf(stderr, "Bootstrapping MTP EMA correction...\n");
        float *prefill_logits = logits_buf + (n_prompt - 1) * vs;
        mdl.mtp.cache_len = 0;
        wubu_mtp_draft_forward(&mdl, h_39, prev_cur, 1, mtp_logits);
        int matched = 0;
        for (int v = 0; v < 256 && v < vs; v++) {
            float diff = prefill_logits[v] - mtp_logits[v];
            logit_correction[v] = 0.8f * diff;  // Full initial correction (decayed 0.8)
            if (fabsf(diff) > 0.1f) matched++;
        }
        // Check if draft[0] matches argmax after correction
        for (int v = 0; v < vs; v++) mtp_logits[v] += logit_correction[v % 256];
        int bootstrap_draft = 0; float bv = mtp_logits[0];
        for (int v = 1; v < vs; v++) if (mtp_logits[v] > bv) { bv = mtp_logits[v]; bootstrap_draft = v; }
        int main_arg = 0; float mv = prefill_logits[0];
        for (int v = 1; v < vs; v++) if (prefill_logits[v] > mv) { mv = prefill_logits[v]; main_arg = v; }
        fprintf(stderr, "  Bootstrap: correction_top100_mag=%.2f, draft_match=%s\n",
               sqrtf(matched > 1 ? 0.01f * matched : 0.0f),
               bootstrap_draft == main_arg ? "YES" : "NO");
        // Reset MTP cache for actual decode loop
        mdl.mtp.cache_len = 0;
    }

    // ============================================================
    // Main decode loop
    // ============================================================
    while (total_gen < max_tokens && !g_stop) {
        // ====== Emit main model's prediction ======
        int main_token = argmax(last_logits, vs);
        {
            char piece[256];
            int nc = wubu_tokenizer_decode(&tok, &main_token, 1, piece, 256);
            if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", main_token);
            fflush(stdout);
        }
        total_gen++;
        if (main_token == tok.eos_id || main_token == tok.bos_id) break;
        if (total_gen >= max_tokens || g_stop) break;

        // ====== Attempt speculative decode ======
        int accepted_drafts = 0;
        (void)accepted_drafts;
        if (use_mtp && mdl.mtp.loaded) {
            total_attempted++;

            // Reset MTP KV cache for fresh draft generation
            mdl.mtp.cache_len = 0;

            // Generate draft[0] using prev_cur (token before the one we just predicted)
            wubu_mtp_draft_forward(&mdl, h_39, prev_cur, 1, mtp_logits);

            // Save raw MTP logits before correction (for EMA update)
            float mtp_raw[256];
            int mtp_raw_n = (vs < 256) ? vs : 256;
            for (int v = 0; v < mtp_raw_n; v++) mtp_raw[v] = mtp_logits[v];

            // Apply online logit correction to compensate for quantization bias
            for (int v = 0; v < vs; v++) mtp_logits[v] += logit_correction[v % 256];

            int draft0 = argmax(mtp_logits, vs);

            if (draft0 == main_token) {
                // draft[0] matches! Update EMA correction.
                // Use adaptive alpha: fast convergence early, stable after.
                float alpha = correction_n < 10 ? 0.2f : 0.05f;
                if (mtp_raw_n > 0) {
                    for (int v = 0; v < mtp_raw_n; v++) {
                        float diff = last_logits[v] - mtp_raw[v];
                        logit_correction[v] = (1.0f - alpha) * logit_correction[v] + alpha * diff;
                    }
                    correction_n++;
                }
                // Generate draft[1]
                float mid_cur[D];
                get_embd(&mdl, main_token, mid_cur, emb_file);

                wubu_mtp_draft_forward(&mdl, h_39, mid_cur, 1, mtp_logits);

                float mtp_raw1[256];
                int mtp_raw1_n = (vs < 256) ? vs : 256;
                for (int v = 0; v < mtp_raw1_n; v++) mtp_raw1[v] = mtp_logits[v];

                for (int v = 0; v < vs; v++) mtp_logits[v] += logit_correction[v % 256];

                int draft1 = argmax(mtp_logits, vs);

                // Checkpoint before verifying draft[1]
                if (!wubu_model_checkpoint(&mdl)) {
                    if (verbose) fprintf(stderr, "\n[MTP] Checkpoint failed\n");
                    goto normal_advance;
                }

                // Forward main_token through main model
                mdl.save_last_hidden = h_39;
                wubu_model_forward_from_embd(&mdl, mid_cur, 1, 1, logits_out);
                mdl.save_last_hidden = NULL;
                int main_next = argmax(logits_out, vs);

                // Update EMA from draft[1] verification
                float alpha1 = correction_n < 10 ? 0.2f : 0.05f;
                if (mtp_raw1_n > 0) {
                    for (int v = 0; v < mtp_raw1_n; v++) {
                        float diff = logits_out[v] - mtp_raw1[v];
                        logit_correction[v] = (1.0f - alpha1) * logit_correction[v] + alpha1 * diff;
                    }
                }

                if (main_next == draft1) {
                    // BOTH drafts accepted
                    total_accepted++;
                    {
                        char piece[256];
                        int nc = wubu_tokenizer_decode(&tok, &draft1, 1, piece, 256);
                        if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", draft1);
                        fflush(stdout);
                    }
                    total_gen++;
                    if (verbose) fprintf(stderr, "\n[MTP] 2/2 accepted\n");

                    get_embd(&mdl, draft1, cur, emb_file);
                    memcpy(prev_cur, mid_cur, D * sizeof(float));

                    if (total_gen < max_tokens && !g_stop &&
                        draft1 != tok.eos_id && draft1 != tok.bos_id) {
                        mdl.save_last_hidden = h_39;
                        wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
                        mdl.save_last_hidden = NULL;
                    }
                    last_logits = logits_out;
                } else {
                    // draft[1] REJECTED — rollback
                    wubu_model_rollback(&mdl);
                    mdl.save_last_hidden = h_39;
                    wubu_model_forward_from_embd(&mdl, mid_cur, 1, 1, logits_out);
                    mdl.save_last_hidden = NULL;

                    int real_next = argmax(logits_out, vs);
                    {
                        char piece[256];
                        int nc = wubu_tokenizer_decode(&tok, &real_next, 1, piece, 256);
                        if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", real_next);
                        fflush(stdout);
                    }
                    total_gen++;
                    if (verbose) fprintf(stderr, "\n[MTP] 1/2 accepted (draft[1] rejected: main=%d draft=%d)\n",
                                         main_next, draft1);

                    get_embd(&mdl, real_next, cur, emb_file);
                    memcpy(prev_cur, mid_cur, D * sizeof(float));
                    last_logits = logits_out;
                }
            } else {
                // draft[0] doesn't match main model
                if (mtp_raw_n > 0) {
                    for (int v = 0; v < mtp_raw_n; v++) {
                        float diff = last_logits[v] - mtp_raw[v];
                        logit_correction[v] = logit_correction[v] + 0.5f * diff;
                    }
                }
                if (verbose) fprintf(stderr, "\n[MTP] draft[0] mismatch: main=%d draft=%d\n", main_token, draft0);
                goto normal_advance;
            }
        } else {
            normal_advance:
            // Normal single-token advance
            get_embd(&mdl, main_token, cur, emb_file);
            memcpy(prev_cur, cur, D * sizeof(float));

            mdl.save_last_hidden = h_39;
            wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
            mdl.save_last_hidden = NULL;
            last_logits = logits_out;
        }

        if (main_token == tok.eos_id || main_token == tok.bos_id) break;
    }

    printf("\n");

    double t_total = wall_clock() - t0;
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Total: %d tok in %.2fs (%.1f tok/s)\n", total_gen + n_prompt, t_total,
           (total_gen + n_prompt) / t_total);
    fprintf(stderr, "Decode: %d tok in %.2fs (%.1f tok/s)\n", total_gen, t_total - t_prefill,
           total_gen / (t_total - t_prefill));
    if (total_attempted > 0) {
        fprintf(stderr, "MTP: %d attempts, %d accepted (%.0f%%), draft=%d\n",
               total_attempted, total_accepted,
               100.0 * total_accepted / total_attempted, DRAFT_N);
    }

    free(embd); free(logits_buf); free(logits_out); free(cur); free(mtp_logits);
    if (emb_file) fclose(emb_file);
    if (mtp_ctx) gguf_close(mtp_ctx);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
