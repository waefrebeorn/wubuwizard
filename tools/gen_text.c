/**
 * gen_text.c — Text generation with optional GPU-accelerated output projection.
 *
 * CPU-only:  make gen_text_cpu
 * GPU:       GPU=1 GPU_BATCH=16 OMP_NUM_THREADS=16 make gen_text_gpu
 *
 * Modes:
 *   Normal:   ./gen_text_cpu "prompt" max_tokens top_k
 *   Persist:  ./gen_text_cpu --persist
 *     Binary protocol on stdin/stdout:
 *     Input:  <4-byte LE text_len> <text> <4-byte LE max_tokens> <4-byte LE top_k>
 *     Output: <4-byte LE result_len> <result> <4-byte LE tokens_generated>
 *     Send text_len=0 to reset KV cache.
 *     Environment: CHAT=1 for ChatML formatting, PERSIST=1 as alternative to --persist
 *
 * Environment:
 *   MODEL=path  — GGUF model path
 *   GPU=1       — Enable GPU output projection
 *   CHAT=1      — ChatML mode (adds system/user/assistant tokens)
 *   PERSIST=1   — Persist mode (alternative to --persist flag)
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
#include <stdbool.h>
#include <unistd.h>

// GPU support — compiled only in gen_text_gpu target
#ifdef GPU_SUPPORT
#include "gpu_output_proj.h"
#else
static inline bool gpu_output_init(const void *w,int D,int V,int t){(void)w;(void)D;(void)V;(void)t;return false;}
static inline bool gpu_output_project_batch(const float *i,float *o,int T){(void)i;(void)o;(void)T;return false;}
static inline bool gpu_output_project(const float *i,float *o){(void)i;(void)o;return false;}
static inline void gpu_output_cleanup(void){}
inline int wubu_model_gpu_init(wubu_model_t *m,int mc,int cs){(void)m;(void)mc;(void)cs;return 0;}
inline void wubu_model_gpu_free(wubu_model_t *m){(void)m;}
#endif

static volatile int g_stop = 0;
static void handle_sigint(int sig) { (void)sig; g_stop = 1; }

static double clock_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int read_embedding(const wubu_model_t *mdl, int token_id, float *out, FILE *emb_file) {
    int D = D_MODEL;
    if (mdl->use_embedding_file) {
        if (emb_file && token_id >= 0 && token_id < mdl->vocab_size) {
            fseek(emb_file, (long long)token_id * D * sizeof(float), SEEK_SET);
            size_t nread = fread(out, sizeof(float), D, emb_file);
            return nread == (size_t)D ? 1 : 0;
        }
        return 0;
    } else if (mdl->token_embd && token_id >= 0 && token_id < mdl->vocab_size) {
        memcpy(out, mdl->token_embd + (long long)token_id * D, D * sizeof(float));
        return 1;
    }
    return 0;
}

// Build ChatML prompt tokens for a user message. On first turn (cache_len==0),
// includes BOS + system prompt. On subsequent turns, just user/assistant wrappers.
static int build_chat_prompt(wubu_tokenizer_t *tok, const char *text,
                             int cache_len, int *tokens, int max_tokens) {
    const int IM_START = 248045, IM_END = 248046, THINK = 248068, NL_TOKEN = 198;
    int pos = 0;
    if (cache_len == 0) {
        tokens[pos++] = tok->bos_id >= 0 ? tok->bos_id : 248044;
        tokens[pos++] = IM_START;
        int n = wubu_tokenizer_encode(tok, "system\nYou are a helpful assistant.", tokens + pos, max_tokens - pos);
        if (n > 0) pos += n;
        tokens[pos++] = IM_END; tokens[pos++] = NL_TOKEN;
    }
    tokens[pos++] = IM_START;
    int n = wubu_tokenizer_encode(tok, "user\n", tokens + pos, max_tokens - pos);
    if (n > 0) pos += n;
    n = wubu_tokenizer_encode(tok, text, tokens + pos, max_tokens - pos);
    if (n > 0) pos += n;
    tokens[pos++] = IM_END; tokens[pos++] = NL_TOKEN;
    tokens[pos++] = IM_START;
    n = wubu_tokenizer_encode(tok, "assistant\n", tokens + pos, max_tokens - pos);
    if (n > 0) pos += n;
    tokens[pos++] = THINK; tokens[pos++] = NL_TOKEN;
    return pos;
}

// Decode loop: generate tokens from last_logits, write to result buffer
static int decode_loop(wubu_model_t *mdl, wubu_tokenizer_t *tok,
                       float *last_logits, int vs,
                       int max_tokens, int top_k, int use_gpu,
                       char *result_buf, int buf_size) {
    int generated = 0;
    int result_pos = 0;
    FILE *emb_file = NULL;
    if (mdl->use_embedding_file)
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");

    while (generated < max_tokens && !g_stop) {
        int topk_idxs[256];
        int nk = top_k > 256 ? 256 : top_k;
        for (int k = 0; k < nk; k++) {
            float maxv = -1e30f; int maxi = -1;
            for (int i = 0; i < vs; i++)
                if (last_logits[i] > maxv) { maxv = last_logits[i]; maxi = i; }
            topk_idxs[k] = maxi; last_logits[maxi] = -1e30f;
        }
        int next_token = topk_idxs[0];
        if (next_token == tok->eos_id || next_token == tok->bos_id) break;

        char piece_buf[256];
        int n_chars = wubu_tokenizer_decode(tok, &next_token, 1, piece_buf, 256);
        if (n_chars > 0 && result_pos + n_chars < buf_size) {
            memcpy(result_buf + result_pos, piece_buf, n_chars);
            result_pos += n_chars;
            result_buf[result_pos] = '\0';
        }

        float x_next[D_MODEL];
        if (!read_embedding(mdl, next_token, x_next, emb_file))
            memset(x_next, 0, D_MODEL * sizeof(float));

        if (use_gpu) {
            mdl->skip_output_proj = true;
            wubu_model_forward_from_embd(mdl, x_next, 1, 1, last_logits);
            gpu_output_project(last_logits, last_logits);
        } else {
            mdl->skip_output_proj = false;
            wubu_model_forward_from_embd(mdl, x_next, 1, 1, last_logits);
        }
        generated++;
    }

    if (emb_file) fclose(emb_file);
    return result_pos;
}

//=== Persist mode ===
static int persist_main(wubu_model_t *mdl, wubu_tokenizer_t *tok) {
    int D = D_MODEL;
    int vs = mdl->vocab_size;
    int chat_mode = getenv("CHAT") != NULL;

    fprintf(stderr, "[persist] ready\n");
    fflush(stderr);

    uint32_t text_len;
    while (fread(&text_len, 4, 1, stdin) == 1) {
        if (text_len == 0) {
            // Reset KV cache
            // Can't easily reset just KV cache; free+re-init model for simplicity
            fprintf(stderr, "[persist] reset not supported in this version\n");
            fflush(stderr);
            continue;
        }

        // Read prompt
        char *text = (char *)malloc(text_len + 1);
        if (!text) break;
        if (fread(text, 1, text_len, stdin) != text_len) { free(text); break; }
        text[text_len] = '\0';

        uint32_t p_max_tokens = 128, p_top_k = 40;
        fread(&p_max_tokens, 4, 1, stdin);
        fread(&p_top_k, 4, 1, stdin);
        if (p_max_tokens == 0 || p_max_tokens > 4096) p_max_tokens = 128;
        if (p_top_k == 0 || p_top_k > 256) p_top_k = 40;

        // Tokenize
        int prompt_tokens[1024];
        int n_prompt;
        if (chat_mode) {
            n_prompt = build_chat_prompt(tok, text, mdl->gqa_cache_len,
                                          prompt_tokens, 1024);
        } else {
            n_prompt = wubu_tokenizer_encode(tok, text, prompt_tokens, 1024);
            if (n_prompt <= 0) {
                prompt_tokens[0] = tok->bos_id >= 0 ? tok->bos_id : 248044;
                n_prompt = 1;
            }
        }

        fprintf(stderr, "[persist] prompt: %d tokens (cache has %d)\n",
                n_prompt, mdl->gqa_cache_len);
        fflush(stderr);

        // Embed
        FILE *emb_file = NULL;
        if (mdl->use_embedding_file)
            emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");

        float *embd = (float *)malloc((size_t)n_prompt * D * sizeof(float));
        for (int i = 0; i < n_prompt; i++)
            if (!read_embedding(mdl, prompt_tokens[i], embd + i * D, emb_file))
                memset(embd + i * D, 0, D * sizeof(float));

        if (emb_file) fclose(emb_file);

        // Forward (prefill)
        float *logits = (float *)malloc((size_t)n_prompt * vs * sizeof(float));
        double t0 = clock_seconds();
        mdl->skip_output_proj = false;
        wubu_model_forward_from_embd(mdl, embd, 1, n_prompt, logits);

        // Decode
        char result_buf[65536];
        int result_len = decode_loop(mdl, tok,
                                      logits + (n_prompt - 1) * vs, vs,
                                      (int)p_max_tokens, (int)p_top_k, 0,
                                      result_buf, 65536);

        double elapsed = clock_seconds() - t0;
        fprintf(stderr, "[persist] done: %d tok in %.2fs\n",
                result_len > 0 ? 1 : 0, elapsed);
        fflush(stderr);

        // Send result: <4-byte len> <text> <4-byte tokens>
        // Precede with marker so Python client can skip model init stdout
        fprintf(stdout, "---BINARY---\n");
        fflush(stdout);
        uint32_t rl = (uint32_t)result_len;
        uint32_t gen = (uint32_t)(result_len > 0 ? 1 : 0); // approximate
        fwrite(&rl, 4, 1, stdout);
        if (result_len > 0) fwrite(result_buf, 1, result_len, stdout);
        fwrite(&gen, 4, 1, stdout);
        fflush(stdout);

        free(logits);
        free(embd);
        free(text);
    }
    return 0;
}

//=== Normal mode ===
int main(int argc, char **argv) {
    const char *model_path = getenv("MODEL");
    if (!model_path) model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int top_k = 40;
    int persist_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--persist") == 0) persist_mode = 1;
        else if (i == 1) prompt = argv[i];
        else if (i == 2) max_tokens = atoi(argv[i]);
        else if (i == 3) top_k = atoi(argv[i]);
    }
    if (getenv("PERSIST")) persist_mode = 1;

    signal(SIGINT, handle_sigint);

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        wubu_model_free(&mdl);
        return 1;
    }

    int D = D_MODEL;
    int vs = mdl.vocab_size;
    int chat_mode = getenv("CHAT") != NULL;
    int use_gpu = getenv("GPU") != NULL;

    if (persist_mode) {
        return persist_main(&mdl, &tok);
    }

    //=== Normal (non-persist) mode ===
    if (use_gpu) {
        int max_ctx = getenv("MAX_CTX") ? atoi(getenv("MAX_CTX")) : 262144;
        int chunk_sz = getenv("GPU_CHUNK") ? atoi(getenv("GPU_CHUNK")) : 256;
        if (!wubu_model_gpu_init(&mdl, max_ctx, chunk_sz)) {
            fprintf(stderr, "GPU GQA init failed, falling back to CPU GQA\n");
        } else {
            printf("GPU: GQA acceleration active (max_ctx=%d, chunk=%d)\n", max_ctx, chunk_sz);
        }
        use_gpu = 0;
    }

    // Tokenize
    int prompt_tokens[1024];
    int n_prompt;
    if (chat_mode) {
        n_prompt = build_chat_prompt(&tok, prompt, 0, prompt_tokens, 1024);
    } else {
        n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
        if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    }
    printf("Prompt: %d tokens\n", n_prompt);

    // Embed
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file)
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");

    float *embd = (float *)malloc((size_t)n_prompt * D * sizeof(float));
    for (int i = 0; i < n_prompt; i++)
        if (!read_embedding(&mdl, prompt_tokens[i], embd + i * D, emb_file))
            memset(embd + i * D, 0, D * sizeof(float));

    // Forward (prefill)
    float *logits = (float *)malloc((size_t)n_prompt * vs * sizeof(float));
    double t0 = clock_seconds();

    if (use_gpu) {
        mdl.skip_output_proj = true;
        mdl.enable_moe = true;
        wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
        float *hidden_batch = (float *)malloc((size_t)n_prompt * D * sizeof(float));
        for (int i = 0; i < n_prompt; i++)
            memcpy(hidden_batch + i * D, logits + i * vs, D * sizeof(float));
        if (n_prompt > 0)
            gpu_output_project_batch(hidden_batch, logits, n_prompt);
        free(hidden_batch);
    } else {
        mdl.skip_output_proj = false;
        wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
    }

    // Dump logits if DUMP_LOGITS set
    const char *dump_logits_path = getenv("DUMP_LOGITS");
    if (dump_logits_path) {
        FILE *df = fopen(dump_logits_path, "wb");
        if (df) {
            float *last_logits = logits + (n_prompt - 1) * vs;
            fwrite(last_logits, sizeof(float), vs, df);
            fclose(df);
            fprintf(stderr, "Dumped logits to %s\n", dump_logits_path);
        }
    }

    double t_prefill = clock_seconds() - t0;
    float *last_logits = logits + (n_prompt - 1) * vs;

    // Print input
    { char buf[1024]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 1024);
      if (nc > 0) printf("Input: %s\n", buf); }

    // Decode loop
    char result_buf[65536];
    int result_len = decode_loop(&mdl, &tok, last_logits, vs,
                                  max_tokens, top_k, use_gpu,
                                  result_buf, 65536);
    fwrite(result_buf, 1, result_len, stdout);
    printf("\n");

    double t_total = clock_seconds() - t0;
    printf("\n--- Stats ---\n");
    printf("Prefill: %d tok in %.2fs (%.1f tok/s)\n", n_prompt, t_prefill, n_prompt / t_prefill);
    double t_decode = t_total - t_prefill;
    if (t_decode > 0)
        printf("Decode:  %d chars in %.2fs\n", result_len, t_decode);

    free(logits); free(embd);
    if (emb_file) fclose(emb_file);
    if (use_gpu) gpu_output_cleanup();
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
