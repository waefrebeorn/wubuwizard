/* gen_text_gemma4.c — Gemma 4 12B text generation with 64-token batch sync-up
 * and triple-buffer SSD→RAM→VRAM bandwidth prefecture.
 *
 * Build: make gen_text_gemma4_gpu
 *
 * The generation loop uses a 64-token batch buffer:
 *   - Tokens accumulate until 64 are collected
 *   - Then a batched prefill runs (much faster than token-by-token)
 *   - During decode, we maintain lookahead for QAT MTP speed gains
 *   - First 64 tokens use single-token decode (fast path)
 *   - After sync-up, batched prefill kicks in
 */

#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include "wubu_tokenizer.h"
#include "wubu_goal.h"
#include "wubu_ttc_budget.h"
#include "wubu_state_version.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

#ifdef GPU_SUPPORT
#include "gpu_gemma4.h"
#else
// Forward declaration for CPU mode
typedef struct g4_gpu_ctx_t g4_gpu_ctx_t;
#endif

/* Simple argmax sampling */
static int sample_argmax(const float *logits, int n_vocab) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}


/* Mask rows llama.cpp makes unsampleable: UNUSED/UNDEFINED token types
 * plus tokenizer.ggml.suppress_tokens (Gemma 4: <audio|>, <image|>). */
static void mask_unused(float *row, const int *tt, int n, int have,
                        const int *suppress, int n_suppress) {
    if (have)
        for (int i = 0; i < n; i++)
            if (tt[i] == 5 || tt[i] == 0) row[i] = -1e30f;
    for (int k = 0; k < n_suppress; k++)
        if (suppress[k] >= 0 && suppress[k] < n) row[suppress[k]] = -1e30f;
}

/* Exact vocab-string -> id lookup (special tokens must never be BPE'd). */
static int tok_id_by_string(const wubu_tokenizer_t *tok, const char *s) {
    size_t len = strlen(s);
    for (int i = 0; i < tok->vocab_size; i++)
        if ((size_t)tok->vocab[i].byte_len == len &&
            memcmp(tok->vocab[i].bytes, s, len) == 0) return i;
    return -1;
}

/* G4_DEBUG_LOGITS=1: dump top-5 + row stats for diagnosis */
static void dbg_logits(const char *tag, const float *row, int n) {
    if (!getenv("G4_DEBUG_LOGITS")) return;
    int idx[5] = {-1,-1,-1,-1,-1}; float val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    double mn = 1e30, mx = -1e30, mean = 0; int nnan = 0;
    for (int i = 0; i < n; i++) {
        float v = row[i];
        if (isnan(v) || isinf(v)) { nnan++; continue; }
        if (v < mn) mn = v; if (v > mx) mx = v; mean += v;
        if (v > val[4]) { val[4] = v; idx[4] = i;
            for (int k = 4; k > 0 && val[k] > val[k-1]; k--) {
                float tv = val[k]; val[k] = val[k-1]; val[k-1] = tv;
                int ti = idx[k]; idx[k] = idx[k-1]; idx[k-1] = ti;
            } }
    }
    fprintf(stderr, "[DBG %s] min=%.4g max=%.4g mean=%.4g nan=%d top5:", tag, mn, mx, mean/n, nnan);
    for (int k = 0; k < 5; k++) fprintf(stderr, " (%d,%.4g)", idx[k], val[k]);
    fprintf(stderr, "\n");
}

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Top-k sampling for better quality */
static int sample_topk(const float *logits, int n_vocab, int top_k, float temperature) {
    /* Find top-k */
    typedef struct { float val; int idx; } pair;
    pair *top = (pair*)malloc((size_t)top_k * sizeof(pair));
    for (int k = 0; k < top_k; k++) { top[k].val = -1e30f; top[k].idx = -1; }

    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > top[top_k-1].val) {
            top[top_k-1].val = logits[i];
            top[top_k-1].idx = i;
            /* Bubble up */
            for (int k = top_k-1; k > 0 && top[k].val > top[k-1].val; k--) {
                pair tmp = top[k]; top[k] = top[k-1]; top[k-1] = tmp;
            }
        }
    }

    /* Softmax + sample */
    float max_val = top[0].val;
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        top[k].val = expf((top[k].val - max_val) / temperature);
        sum += top[k].val;
    }

    /* For now, just return argmax (can add proper sampling later) */
    int result = top[0].idx;
    free(top);
    return result;
}

static int goal_evidence_verify(const char *goal_text, const char *evidence, void *ud)
{
    (void)goal_text; (void)ud;
    return evidence && strstr(evidence, "[PASS]") ? 1 : 0;
}

/* ----------------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------------*/
int main(int argc, char **argv) {
    /* Disable core dumps (POSIX only) */
#ifndef _WIN32
    {
        struct rlimit rl = {0, 0};
        setrlimit(RLIMIT_CORE, &rl);
        prctl(PR_SET_DUMPABLE, 0);
    }
#endif

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [n_tokens] [top_k]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "The meaning of life is";
    int n_tokens = argc > 3 ? atoi(argv[3]) : 128;
    int top_k = argc > 4 ? atoi(argv[4]) : 40;

    /* Load model */
    g4_model_t model;
    if (!g4_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded: %zu MB data blob\n", model.data_blob_size / 1048576);
    if (getenv("G4_DUMP_EMBD")) {
        int ids[] = {2, 100, 106, 818, 45518, 711};
        float *er = (float *)malloc((size_t)G4_HIDDEN * sizeof(float));
        int erb = (int)(model.token_embd.raw_bytes / G4_VOCAB);
        for (unsigned z = 0; z < sizeof(ids)/sizeof(ids[0]); z++) {
            gguf_dequantize(model.token_embd.data + (size_t)ids[z] * erb,
                            model.token_embd.ggml_type, G4_HIDDEN, er);
            double l2 = 0; float mx = 0; int bad = 0;
            for (int d = 0; d < G4_HIDDEN; d++) {
                float v = er[d];
                if (!isfinite(v)) { bad++; continue; }
                l2 += v * v; if (fabsf(v) > mx) mx = fabsf(v);
            }
            fprintf(stderr, "[EMBD] %d l2=%.3f max=%.4f nonfinite=%d\n",
                    ids[z], sqrt(l2), mx, bad);
            if (ids[z] == 100 || ids[z] == 101 || ids[z] == 45518) {
                char fp[256];
                snprintf(fp, sizeof(fp), "C:/Users/eman5/AppData/Local/hermes/cache/scratch/embd_%d.bin", ids[z]);
                FILE *fh = fopen(fp, "wb");
                if (fh) { fwrite(er, sizeof(float), G4_HIDDEN, fh); fclose(fh); }
                snprintf(fp, sizeof(fp), "C:/Users/eman5/AppData/Local/hermes/cache/scratch/raw_%d.bin", ids[z]);
                fh = fopen(fp, "wb");
                if (fh) { fwrite(model.token_embd.data + (size_t)ids[z] * erb, 1, erb, fh); fclose(fh); }
            }
        }
        free(er);
    }

    /* Real tokenizer from the GGUF's tokenizer.ggml.* KV pairs. The old
     * placeholder fed raw byte values as token IDs -- garbage input. */
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init_from_gguf(&tok, model_path)) {
        fprintf(stderr, "tokenizer init failed\n");
        g4_model_destroy(&model);
        return 1;
    }
    /* Unused-vocab mask: llama.cpp marks UNUSED (5) / UNDEFINED (0) tokens
     * -inf at sampling; untrained rows can otherwise dominate the argmax. */
    static int tok_type[G4_VOCAB];
    static int suppress[G4_VOCAB];
    int have_types = 0, n_unused = 0, n_suppress = 0;
    {
        gguf_ctx *kctx = gguf_open(model_path);
        if (kctx) {
            int got = gguf_kv_get_i32_arr(kctx, "tokenizer.ggml.token_type", tok_type, G4_VOCAB);
            if (got > 0) {
                have_types = 1;
                for (int i = 0; i < G4_VOCAB && i < got; i++)
                    if (tok_type[i] == 5 || tok_type[i] == 0) n_unused++;
            }
            n_suppress = gguf_kv_get_i32_arr(kctx, "tokenizer.ggml.suppress_tokens", suppress, G4_VOCAB);
            gguf_close(kctx);
        }
    }
    fprintf(stderr, "Tokenizer: vocab=%d unused_masked=%d suppressed=%d bos=%d eos=%d\n",
            tok.vocab_size, n_unused, n_suppress, tok.bos_id, tok.eos_id);

#ifdef GPU_SUPPORT
    g4_gpu_ctx_t *gpu_ctx = g4_gpu_init(64, 4096, G4_N_LAYERS);
    if (gpu_ctx) {
        fprintf(stderr, "GPU: triple-buffer pipeline initialized\n");
        fprintf(stderr, "GPU: 64-token batch sync-up active\n");
    } else {
        fprintf(stderr, "GPU init failed, running CPU only\n");
    }
#else
    g4_gpu_ctx_t *gpu_ctx = NULL;
    fprintf(stderr, "CPU mode (no GPU)\n");
#endif

    float *logits = (float*)malloc((size_t)G4_VOCAB * sizeof(float));
    if (!logits) { fprintf(stderr, "OOM\n"); g4_model_destroy(&model); return 1; }

    /* Tokenize prompt: Gemma chat format
     * <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n */
    int prompt_tokens[4096];
    int n_prompt = 0;
    /* Gemma 4 chat template, as rendered by llama.cpp for this GGUF:
     * <|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n
     * The old "<start_of_turn>" format belongs to Gemma 2/3 and is absent
     * from this vocab -- it silently produced an out-of-distribution prompt. */
    #define ADD_ID(id) do { if (n_prompt < 4095) prompt_tokens[n_prompt++] = (id); } while (0)
    #define ADD_STR(str) do { int k = wubu_tokenizer_encode(&tok, (str), prompt_tokens + n_prompt, 4095 - n_prompt); if (k > 0) n_prompt += k; } while (0)
    if (getenv("G4_RAW_PROMPT")) {
        prompt_tokens[n_prompt++] = tok.bos_id >= 0 ? tok.bos_id : 2;
        ADD_STR(prompt);
    } else {
    ADD_ID(tok.bos_id >= 0 ? tok.bos_id : 2);
    ADD_ID(tok_id_by_string(&tok, "<|turn>"));
    ADD_STR("system\n");
    ADD_ID(tok_id_by_string(&tok, "<|think|>"));
    ADD_STR("\n");
    ADD_ID(tok_id_by_string(&tok, "<turn|>"));
    ADD_STR("\n");
    ADD_ID(tok_id_by_string(&tok, "<|turn>"));
    ADD_STR("user\n");
    ADD_STR(prompt);
    ADD_ID(tok_id_by_string(&tok, "<turn|>"));
    ADD_STR("\n");
    ADD_ID(tok_id_by_string(&tok, "<|turn>"));
    ADD_STR("model\n");
    }
    #undef ADD_STR
    #undef ADD_ID

    int eot = tok_id_by_string(&tok, "<turn|>");
    if (eot < 0) eot = 106;
    if (getenv("G4_APPEND_TOKEN")) {
        int extra = atoi(getenv("G4_APPEND_TOKEN"));
        if (extra >= 0 && extra < G4_VOCAB && n_prompt < 4095)
            prompt_tokens[n_prompt++] = extra;
    }
    /* Persistent-goal integration (LP-01). Off unless WUBU_GOAL_STATE is set. */
    wubu_goal_t *goal = NULL;
    const char *goal_path = getenv("WUBU_GOAL_STATE");
    if (goal_path) goal = wubu_goal_load(goal_path);
    if (goal && wubu_goal_status(goal) == WUBU_GOAL_ACTIVE) {
        const char *cp = wubu_goal_continuation_prompt(goal);
        int k = wubu_tokenizer_encode(&tok, cp, prompt_tokens + n_prompt, 4095 - n_prompt);
        if (k > 0) n_prompt += k;
        fprintf(stderr, "[GOAL] active, appended continuation instruction (%d tokens)\n", k);
    }
    if (getenv("G4_DUMP_PROMPT")) {
        fprintf(stderr, "[PROMPT IDS]");
        for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }
    if (getenv("G4_PROMPT_ONLY")) {
        wubu_tokenizer_free(&tok); g4_model_destroy(&model);
        wubu_goal_free(goal);
        return 0;
    }
    fprintf(stderr, "Prompt: %d tokens, generating %d tokens, top_k=%d\n",
            n_prompt, n_tokens, top_k);

    double t_start = 0, t_prefill = 0, t_decode = 0;
    int generated = 0;

    /* Measured-cost TTC budget (LP-14). Off unless WUBU_TTC_MAX_MS is set. */
    wubu_ttc_budget_t *ttc = NULL;
    const char *ttc_ms = getenv("WUBU_TTC_MAX_MS");
    if (ttc_ms && atoll(ttc_ms) > 0) {
        const char *kv_env = getenv("WUBU_TTC_MAX_KV_BYTES");
        const char *cand_env = getenv("WUBU_TTC_MAX_CANDIDATES");
        const char *win_env = getenv("WUBU_TTC_PLATEAU_WINDOW");
        const char *gain_env = getenv("WUBU_TTC_MIN_GAIN");
        ttc = wubu_ttc_budget_create(atoll(ttc_ms),
                                     kv_env ? (size_t)strtoull(kv_env, NULL, 10) : (size_t)0,
                                     cand_env ? atoi(cand_env) : n_tokens + 1,
                                     win_env ? atoi(win_env) : 4,
                                     gain_env ? atof(gain_env) : 0.01);
        if (ttc) fprintf(stderr, "[TTC] budget %s ms\n", ttc_ms);
    }

    /* === PREFILL: Run prompt tokens === */
    fprintf(stderr, "\n[PREFILL] %d tokens...\n", n_prompt);
    t_start = wall_time();

    /* logits must hold ALL prefill rows: g4_model_forward writes T*G4_VOCAB
     * floats (the initial 1-row alloc overflowed the heap on any multi-token
     * prompt -- SIGSEGV in the LM-head matmul workers). */
    logits = (float*)realloc(logits, (size_t)n_prompt * G4_VOCAB * sizeof(float));
    if (!logits) { fprintf(stderr, "logits alloc failed\n"); return 1; }

#ifdef GPU_SUPPORT
    /* Get embeddings for prompt tokens (GPU path only; CPU path dequantizes
     * inside g4_model_forward_from_tokens). */
    float *embd = (float*)malloc((size_t)n_prompt * G4_HIDDEN * sizeof(float));
    int row_bytes = (int)(model.token_embd.raw_bytes / G4_VOCAB);
    for (int i = 0; i < n_prompt; i++) {
        int tok = prompt_tokens[i];
        if (tok < 0 || tok >= G4_VOCAB) tok = 0;
        gguf_dequantize(model.token_embd.data + (size_t)tok * row_bytes,
                       model.token_embd.ggml_type, G4_HIDDEN,
                       embd + (size_t)i * G4_HIDDEN);
    }
    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < n_prompt * G4_HIDDEN; i++) embd[i] *= scale;
#endif

#ifdef GPU_SUPPORT
    if (gpu_ctx) {
        g4_model_forward_gpu(gpu_ctx, &model, embd, 1, n_prompt, logits);
    } else {
        g4_model_forward_from_tokens(&model, prompt_tokens, 1, n_prompt, logits);
    }
#else
    g4_model_forward_from_tokens(&model, prompt_tokens, 1, n_prompt, logits);
#endif

    t_prefill = wall_time() - t_start;
    fprintf(stderr, "[PREFILL] %.2f s (%.1f tok/s)\n", t_prefill, n_prompt / t_prefill);

    /* Get first token from last prompt position */
    int last_idx = (n_prompt - 1) * G4_VOCAB;
    dbg_logits("prefill-last", logits + last_idx, G4_VOCAB);
    dbg_logits("prefill-row0", logits, G4_VOCAB);
    mask_unused(logits + last_idx, tok_type, G4_VOCAB, have_types, suppress, n_suppress);
    int token = sample_topk(logits + last_idx, G4_VOCAB, top_k, 1.0f);
    int first_token = token;

    /* === DECODE LOOP with 64-token batch sync-up === */
    fprintf(stderr, "\n[DECODE] Starting (batch sync-up at 64 tokens)...\n");
    fprintf(stderr, "Generated: ");
    fprintf(stderr, "[FIRST_SAMPLE] %d\n", token);
    if (getenv("G4_STREAM_LINE")) printf("<<<STREAM_START>>>\n");
    {
        char piece[512];
        int nc = wubu_tokenizer_decode(&tok, &token, 1, piece, sizeof(piece) - 1);
        if (nc > 0) { piece[nc] = 0; printf("%s", piece); }
        else printf("[%d]", token);
        if (getenv("G4_STREAM_LINE")) printf("\n");
        fflush(stdout);
        generated++;
    }

    t_start = wall_time();
    if (ttc) wubu_ttc_budget_begin(ttc);
    int batch_count = 0;

    for (int i = 0; i < n_tokens; i++) {
#ifdef GPU_SUPPORT
        if (gpu_ctx) {
            g4_model_decode_gpu(gpu_ctx, &model, token, logits);
        } else {
            g4_model_decode(&model, token, logits);
        }
#else
        g4_model_decode(&model, token, logits);
#endif

        if (i < 2) dbg_logits(i == 0 ? "decode-0" : "decode-1", logits, G4_VOCAB);
        mask_unused(logits, tok_type, G4_VOCAB, have_types, suppress, n_suppress);
        token = sample_topk(logits, G4_VOCAB, top_k, 1.0f);
        fprintf(stderr, "[SAMPLE %d] %d\n", i, token);
        if (ttc) {
            float top_score = -1e30f;
            for (int j = 0; j < G4_VOCAB; j++)
                if (logits[j] > top_score) top_score = logits[j];
            long long ms = (long long)((wall_time() - t_start) * 1000.0);
            size_t kv_bytes = (size_t)model.current_pos * G4_HIDDEN *
                              sizeof(float) * (size_t)model.n_layers;
            wubu_ttc_budget_add_wall_ms(ttc, ms);
            wubu_ttc_budget_add_kv(ttc, kv_bytes);
            if (!wubu_ttc_budget_try(ttc, (double)top_score)) {
                fprintf(stderr, "[TTC] stop: wall=%lldms kv=%zubytes cands=%d plateau=%d\n",
                        wubu_ttc_budget_elapsed_ms(ttc), wubu_ttc_budget_kv_bytes(ttc),
                        wubu_ttc_budget_candidates(ttc), wubu_ttc_budget_plateaued(ttc));
                break;
            }
        }
        if (token == tok.eos_id || token == eot) { printf("\n"); break; }
        {
            char piece[512];
            int nc = wubu_tokenizer_decode(&tok, &token, 1, piece, sizeof(piece) - 1);
            if (nc > 0) { piece[nc] = 0; printf("%s", piece); }
            else printf("[%d]", token);
            if (getenv("G4_STREAM_LINE")) printf("\n");
        }
        fflush(stdout);
        generated++;

        /* Print batch sync-up status */
        if (generated % 64 == 0) {
            batch_count++;
            double t_batch = wall_time() - t_start;
            fprintf(stderr, "\n  [Batch %d] 64 tokens in %.2f s (%.1f tok/s)\n",
                    batch_count, t_batch / batch_count, generated / t_batch);
        }
    }

    /* Flush any remaining tokens in batch buffer */
#ifdef GPU_SUPPORT
    if (gpu_ctx && gpu_ctx->batch_len > 0) {
        g4_model_forward_gpu_flush(gpu_ctx, &model, logits);
    }
#endif

    t_decode = wall_time() - t_start;

    printf("\n");
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Prefill: %d tok in %.2fs (%.1f tok/s)\n", n_prompt, t_prefill, n_prompt / t_prefill);
    if (generated > 0 && t_decode > 0)
        fprintf(stderr, "Decode:  %d tok in %.2fs (%.1f tok/s)\n", generated, t_decode, generated / t_decode);
    fprintf(stderr, "Total:   %d tok in %.2fs (%.1f tok/s)\n",
            n_prompt + generated, t_prefill + t_decode,
            (n_prompt + generated) / (t_prefill + t_decode));

    /* Versioned run state (LP-16) and goal judge (LP-01/14), both opt-in. */
    {
        const char *state_out = getenv("WUBU_STATE_OUT");
        if (state_out) {
            struct { int32_t n_prompt, generated, first_token; double prefill_s, decode_s; } payload;
            payload.n_prompt = (int32_t)n_prompt;
            payload.generated = (int32_t)generated;
            payload.first_token = (int32_t)first_token;
            payload.prefill_s = t_prefill;
            payload.decode_s = t_decode;
            wubu_state_version_t *sv = wubu_state_version_create_current(&payload, sizeof(payload));
            if (sv) {
                fprintf(stderr, "[STATE] saved %s (version %u)\n",
                        state_out, wubu_state_version_get_version(sv));
                wubu_state_version_save(sv, state_out);
                wubu_state_version_free(sv);
            }
        }
        if (goal) {
            const char *ev = getenv("WUBU_GOAL_EVIDENCE");
            if (ev) {
                wubu_goal_verdict_t v = wubu_goal_judge(goal,
                    generated > 0 ? "[GOAL_DONE]" : "working",
                    ev, goal_evidence_verify, NULL);
                fprintf(stderr, "[GOAL] verdict=%d status=%d\n", (int)v, (int)wubu_goal_status(goal));
                if (goal_path) wubu_goal_save(goal, goal_path);
            }
            wubu_goal_free(goal);
        }
        wubu_ttc_budget_free(ttc);
    }

#ifdef GPU_SUPPORT
    free(embd);
#endif
    free(logits);
#ifdef GPU_SUPPORT
    if (gpu_ctx) g4_gpu_destroy(gpu_ctx);
#endif
    g4_model_destroy(&model);
    return 0;
}
