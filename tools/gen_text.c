/**
 * gen_text.c — Text generation with optional GPU-accelerated output projection.
 *
 * CPU-only:  make gen_text
 * GPU:       GPU=1 GPU_BATCH=16 OMP_NUM_THREADS=16 make gen_text_gpu
 *
 * Environment:
 *   GPU=1       — Enable GPU output projection
 *   GPU_BATCH=N — Max batch size for batched prefill (default 1)
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include "wubu_repetition.h"
#include "wubu_model_safetensors_bridge.h"
#include "wubu_tokenizer_hf.h"
#include "wubu_generate.h"  /* KB5: spec decode */
#include "wubu_prefix_cache.h"  /* KB7: prefix cache (doc 010) */
#include "wubu_kernel.h"  /* HW dispatch table */
#include "wubu_eagle.h"   /* G01: EAGLE speculative decode */
#include "wubu_smt_check.h"     /* F02: boot-time GEMV verification */
#include "wubu_lmcache.h"       /* A06: persistent KV cache */
#include "wubu_kv_adaptive.h"   /* 001: Ecco entropy-aware KV */
#include "wubu_mem_budget.h"    /* OOM-proof memory budget detector */
#include "wubu_hwcaps.h"       /* HW-accel: SIMD ladder */
#include "wubu_rambus.h"       /* HW-accel: RDRAM-interleaved KV */
#include "wubu_tandem.h"       /* HW-accel: N64 RCP two-stage pipeline */
#include "wubu_gamebud.h"      /* HW-accel: game frame-budget governor */
#include "wubu_banner.h"       /* UX: consistent banner + stats formatting */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

// GPU support — compiled only in gen_text_gpu target (-DGPU_SUPPORT)
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
static void handle_sigint(int sig) { (void)sig; g_stop = 1; fprintf(stderr, "\n[interrupt]\n"); }

static double clock_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int read_embedding(const wubu_model_t *mdl, int token_id, float *out, FILE *emb_file) {
    int D = D_MODEL;
    if (token_id < 0 || token_id >= mdl->vocab_size) token_id = 0;
    if (mdl->lazy_embd_raw) {
        /* Zero-copy BF16/F16 embedding: dequant ONE row per token from the
         * mmap'd shard (Colonel safetensors models use this). Mirrors the
         * forward's embedding read so decode gets real vectors. */
        const uint8_t *base = mdl->lazy_embd_raw + (size_t)token_id * (size_t)mdl->lazy_embd_row * 2;
        if (mdl->lazy_embd_dtype == ST_DTYPE_BF16) {
            const uint16_t *s = (const uint16_t *)base;
            for (int k = 0; k < D; k++) out[k] = st_bf16_to_f32(s[k]);
        } else if (mdl->lazy_embd_dtype == ST_DTYPE_F16) {
            const uint16_t *s = (const uint16_t *)base;
            for (int k = 0; k < D; k++) out[k] = st_f16_to_f32(s[k]);
        } else {
            memcpy(out, base, D * sizeof(float));
        }
        return 1;
    } else if (mdl->use_embedding_file && emb_file) {
        fseek(emb_file, (long long)token_id * D * sizeof(float), SEEK_SET);
        size_t nread = fread(out, sizeof(float), D, emb_file);
        return nread == (size_t)D ? 1 : 0;
    } else if (mdl->token_embd) {
        memcpy(out, mdl->token_embd + (long long)token_id * D, D * sizeof(float));
        return 1;
    } else if (mdl->token_embd_q) {
        /* Large-vocab GGUF: dequantize one row. */
        gguf_tensor_info *t_emb = gguf_find_tensor(mdl->gguf_ctx, "token_embd.weight");
        int bytes_per_token = (int)(D * sizeof(float));
        if (t_emb) {
            int64_t n_elems = 1;
            for (int d = 0; d < t_emb->n_dims; d++) n_elems *= t_emb->dims[d];
            int64_t raw = gguf_raw_size(t_emb->ggml_type, n_elems);
            /* Exact bytes per vocab row: raw / dims[1] (see wubu_model.c
             * notes — old formula dropped block overhead + used dims[1]). */
            bytes_per_token = (int)(raw / t_emb->dims[1]);
            if (bytes_per_token <= 0)
                bytes_per_token = (int)(raw / n_elems * t_emb->dims[0]);
        }
        gguf_dequantize(mdl->token_embd_q + (size_t)token_id * bytes_per_token,
                        mdl->token_embd_type, D, out);
        return 1;
    }
    return 0;
}

/* ----------------------------------------------------------------------------
 * Sampler: temperature + top-p (nucleus) + top-k, with a seeded PRNG.
 * Defaults match the Colonel tuning for Agents-A1-4B / Qwen3.6 on RTX 5070 Ti
 * (temp 0.6 / top_p 0.95 / top_k 20). All overridable via env.
 * Uses XORO-128+ (deterministic, no libc-RNG global state). */
typedef struct { uint64_t s[2]; } xrng_t;
static uint64_t xrng_next(xrng_t *r) {
    uint64_t s0 = r->s[0], s1 = r->s[1];
    uint64_t res = s0 + s1;
    r->s[0] = s1;
    s1 ^= s1 << 23; s1 ^= s1 >> 18; s1 ^= s0 ^ (s0 >> 5);
    r->s[1] = s1;
    return res + ((res >> 27) ^ s1);  // xoroshiro128+ output mix
}
static float xrng_f32(xrng_t *r) { return (float)(xrng_next(r) >> 11) * (1.0f / 9007199254740992.0f); }

/* Sort helper for top-k indices by descending logit (insertion, small k). */
static int sample_token(xrng_t *rng, const float *logits, int vocab,
                        float temp, float top_p, int top_k) {
    if (temp <= 0.0f) {  /* greedy */
        int best = 0; float bv = logits[0];
        for (int i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    /* 1. temperature */
    float *z = (float *)malloc((size_t)vocab * sizeof(float));
    float maxl = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > maxl) maxl = logits[i];
    double inv_t = 1.0 / (double)temp;
    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        float v = (logits[i] - maxl) * (float)inv_t;
        z[i] = (float)expf(v);
        sum += z[i];
    }
    /* 2. top-k truncation (cap candidates) */
    if (top_k > 0 && top_k < vocab) {
        /* find the k-th largest via partial selection */
        float kth = -1e30f;
        for (int c = 0; c < top_k; c++) {
            int bi = 0; float bv = -1e30f;
            for (int i = 0; i < vocab; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
            if (c == top_k - 1) { kth = bv; break; }
            z[bi] = -1e30f;  /* remove from further consideration */
        }
        for (int i = 0; i < vocab; i++) if (z[i] < kth) z[i] = 0.0f;
    }
    /* 3. top-p nucleus: keep the smallest set of top tokens whose cumulative
     *    probability mass reaches top_p; mask the rest to 0. */
    double cum = 0.0;
    const double tp = (double)top_p;
    for (int c = 0; c < vocab; c++) {
        /* pick the still-active (positive) max */
        int bi = 0; float bv = -1.0f;
        for (int i = 0; i < vocab; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
        if (bv <= 0.0f) break;
        cum += bv;
        z[bi] = -1.0f;               /* mark as visited (kept) */
        if (cum >= tp) {             /* threshold reached: cut all smaller */
            for (int i = 0; i < vocab; i++) if (z[i] > 0.0f) z[i] = 0.0f;
            break;
        }
    }
    /* renormalize kept mass */
    double tot = 0.0;
    for (int i = 0; i < vocab; i++) tot += z[i];
    if (tot <= 0.0) {  /* degenerate: fall back to argmax */
        free(z);
        int best = 0; float bv = logits[0];
        for (int i = 1; i < vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    double r = xrng_f32(rng) * tot;
    double acc = 0.0;
    int chosen = 0;
    for (int i = 0; i < vocab; i++) {
        acc += z[i];
        if (r <= acc) { chosen = i; break; }
    }
    free(z);
    return chosen;
}

/* Load a model checkpoint, dispatching by extension:
 *   *.safetensors -> Colonel HF model (Qwen3.6 / Agents-A1 / KAT / BTL-3)
 *                 via the F32 safetensors bridge + adapter detection.
 *   *.gguf         -> legacy wubuwizard GGUF path.
 * Returns 1 on success (model ready), 0 on failure. */
static int init_model(wubu_model_t *mdl, const char *path) {
    int is_st = (strstr(path, ".safetensors") != NULL);
    struct stat _st;
    if (!is_st && stat(path, &_st) == 0 && S_ISDIR(_st.st_mode)) is_st = 1; /* dir of shards */
    if (is_st) {
        /* wubu_model_init_auto handles both a .safetensors file AND a directory
         * holding model-NNN-of-MMM shards (and BTL-3 LoRA + ds4-ssd sidecar). */
        return wubu_model_init_auto(mdl, path) == 0 ? 1 : 0;
    }
    wubu_dims_default();   /* legacy GGUF builds use the 2048-dim defaults */
    return wubu_model_init(mdl, path);
}

int main(int argc, char **argv) {
    wubu_print_banner("Text Generation", "KV-FS namespace · Ref kernels · Spec decode");

    /* Initialize kernel dispatch table (registers CUDA backend if compiled in) */
    wubu_kernel_init();

    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *env_mp = getenv("MODEL");
    if (env_mp) model_path = env_mp;
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    /* Colonel tuning: Agents-A1-4B / Qwen3.6 on RTX 5070 Ti (16GB).
     * temp 0.6 / top_p 0.95 / top_k 20. Env-overridable. */
    float gen_temp   = getenv("TEMP")      ? (float)atof(getenv("TEMP"))      : 0.6f;
    float gen_top_p  = getenv("TOP_P")     ? (float)atof(getenv("TOP_P"))     : 0.95f;
    int   gen_top_k  = getenv("TOP_K")     ? atoi(getenv("TOP_K"))            : 20;

    if (argc > 1 && (strstr(argv[1], ".safetensors")
                     || access(argv[1], F_OK) == 0)) {
        /* argv[1] is a model path; later args are prompt/tokens */
        model_path = argv[1];
        if (argc > 2) prompt = argv[2];
        if (argc > 3) max_tokens = atoi(argv[3]);
    } else {
        if (argc > 1) prompt = argv[1];
        if (argc > 2) max_tokens = atoi(argv[2]);
    }

    signal(SIGINT, handle_sigint);

    // Disable core dumps to avoid 16GB+ crash files (Linux only; no-op on Win)
#if !defined(_WIN32)
    {
        struct rlimit rl = {0, 0};
        setrlimit(RLIMIT_CORE, &rl);
        prctl(PR_SET_DUMPABLE, 0);
    }
#endif

    wubu_model_t mdl;
    if (!init_model(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    // ---- HW-acceleration wiring (doc "tandem"/"rambus"/"gamebud"/hwcaps) ----
    // RDRAM-interleaved KV banks, game frame-budget (20ms/decode-step), tandem
    // pinned prefill/decode pools. kv_dim = kv_heads * head_dim (from dims).
    {
        int kv_h = mdl.gqa_kv_heads > 0 ? mdl.gqa_kv_heads : 8;
        int kv_dim = (mdl.gqa_head_dim > 0 ? mdl.gqa_head_dim : 128) * kv_h;
        uint64_t frame_us = 20000;  /* 20ms decode frame budget */
        int banks = 8;
        const char *rb = getenv("WUBU_RAMBUS_BANKS");
        const char *fb = getenv("WUBU_FRAME_US");
        if (rb) banks = atoi(rb);
        if (fb) frame_us = (uint64_t)strtoull(fb, NULL, 10);
        wubu_model_wire_hwaccel(&mdl, 1 /*simd detect*/, banks, kv_dim,
                                frame_us, "0-1", "2-3");
        fprintf(stderr, "[hwaccel] %s\n", wubu_model_hwaccel_str(&mdl));
    }

    // Speed optimizations for maximum context:
    // 1. Q8 KV cache (auto-selected by wubu_kv_autoselect for ctx≥4096)
    // 2. SWA window (reduces O(ctx) → O(window) attention, set via WUBU_SWA env)
    // 3. Spec decode (n-gram draft, set via WUBU_SPEC_DECODE env)
    // 4. Layer prefetch (hardware prefetch for Q4_K weight dequant)
    // 5. Split-K attention (parallelizes over KV axis for long context)
    // 6. Fused Q4_K matmul (4x bandwidth reduction vs F32 dequant)
    // All auto-enabled; override with env vars as needed.
    const char *swa_env = getenv("WUBU_SWA");
    if (!swa_env) {
        // SWA default: let the model's budget calculator set it based on KV size.
        // wubu_ssm.c auto-selects SWA=512 when cache_len > 512 (auto-KV eviction).
    }

    // GPU init (if GPU=1 env var set)
    int use_gpu = getenv("GPU") != NULL;
    if (use_gpu) {
        // Initialize integrated GPU context: GQA layers, KV cache, chunked attention
        int max_ctx = getenv("MAX_CTX") ? atoi(getenv("MAX_CTX")) : 262144;
        int chunk_sz = getenv("GPU_CHUNK") ? atoi(getenv("GPU_CHUNK")) : 256;
        if (!wubu_model_gpu_init(&mdl, max_ctx, chunk_sz)) {
            fprintf(stderr, "GPU GQA init failed, falling back to CPU GQA\n");
        } else {
            printf("GPU: GQA acceleration active (max_ctx=%d, chunk=%d)\n", max_ctx, chunk_sz);
        }
        // Initialize GPU output projection with model's output.weight (Q4_K)
        if (mdl.output_weight_q && gpu_output_init(mdl.output_weight_q, D_MODEL, mdl.vocab_size, mdl.output_weight_type)) {
            printf("GPU: Output projection active (Q4_K via cuBLAS)\n");
        } else {
            fprintf(stderr, "GPU: Output projection init failed, using CPU\n");
            use_gpu = 0;
        }
    }

    wubu_tokenizer_t tok;
    /* For safetensors/Colonel models, prefer the HF tokenizer.json in the
     * same directory. Fall back to the GGUF tokenizer otherwise. */
    wubu_tok_hf_t *hf_tok = NULL;
    {
        char hf_path[1024];
        const char *slash = strrchr(model_path, '/');
        if (slash) {
            int n = slash - model_path + 1;
            snprintf(hf_path, sizeof(hf_path), "%.*s/tokenizer.json", n, model_path);
        } else {
            snprintf(hf_path, sizeof(hf_path), "tokenizer.json");
        }
        FILE *tf = fopen(hf_path, "rb");
        if (tf) { fclose(tf); hf_tok = wubu_tok_hf_load(hf_path); }
    }
    if (hf_tok) {
        wubu_print_stat("Tokenizer", "HF tokenizer.json (%d tokens)",
                        wubu_tok_hf_vocab_size(hf_tok));
        /* minimal wubu_tokenizer_t compatibility shim */
        tok.bos_id = wubu_tok_hf_bos_id(hf_tok);
        tok.eos_id = wubu_tok_hf_eos_id(hf_tok);
    } else if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to init tokenizer\n");
        wubu_model_free(&mdl);
        return 1;
    }

    // --- Repetition suppression (repeat_penalty + DRY) ---
    // Tuned for the Colonel models on RTX 5070 Ti (see wubuwizard STATUS.md):
    //   Q8:  repeat_penalty 1.05, DRY multiplier 0.5
    //   F16: repeat_penalty 1.1,  DRY multiplier 1.2 (Agents-A1-4B F16 exact)
        wubu_rep_state_t *rep = wubu_rep_create(mdl.vocab_size, 256, 2, -1);
        if (rep) {
            float rp = getenv("REPEAT_PENALTY") ? (float)atof(getenv("REPEAT_PENALTY")) : 1.1f;
            float dm = getenv("DRY_MULTIPLIER") ? (float)atof(getenv("DRY_MULTIPLIER")) : 1.2f;
            float db = getenv("DRY_BASE") ? (float)atof(getenv("DRY_BASE")) : 1.75f;
            wubu_rep_set_params(rep, rp, dm, db);
        }

    int D = D_MODEL;
    int vs = mdl.vocab_size;

    // Tokenize prompt
    int prompt_tokens[1024];
    int n_prompt;
    int chat_mode = getenv("CHAT") != NULL;
    if (chat_mode) {
        const int IM_START = 248045, IM_END = 248046, THINK = 248068, NL_TOKEN = 198;
        int pos = 0;
        prompt_tokens[pos++] = tok.bos_id;
        prompt_tokens[pos++] = IM_START;
        int n = hf_tok ? wubu_tok_hf_encode(hf_tok, "system\nYou are a helpful assistant.", prompt_tokens + pos, 1024 - pos)
                       : wubu_tokenizer_encode(&tok, "system\nYou are a helpful assistant.", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = IM_END; prompt_tokens[pos++] = NL_TOKEN;
        prompt_tokens[pos++] = IM_START;
        n = hf_tok ? wubu_tok_hf_encode(hf_tok, "user\n", prompt_tokens + pos, 1024 - pos)
                   : wubu_tokenizer_encode(&tok, "user\n", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        n = hf_tok ? wubu_tok_hf_encode(hf_tok, prompt, prompt_tokens + pos, 1024 - pos)
                   : wubu_tokenizer_encode(&tok, prompt, prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = IM_END; prompt_tokens[pos++] = NL_TOKEN;
        prompt_tokens[pos++] = IM_START;
        n = hf_tok ? wubu_tok_hf_encode(hf_tok, "assistant\n", prompt_tokens + pos, 1024 - pos)
                   : wubu_tokenizer_encode(&tok, "assistant\n", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = THINK; prompt_tokens[pos++] = NL_TOKEN;
        n_prompt = pos;
    } else {
        n_prompt = hf_tok ? wubu_tok_hf_encode(hf_tok, prompt, prompt_tokens, 1024)
                          : wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
        if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    }
    printf("wubu: prompt = %d tokens\n", n_prompt);

    // Embeddings — OOM guard: only allocate embed buffer if NOT using chunked prefill.
    // Chunked prefill (wubu_model_forward_chunked) takes raw token_ids and does
    // its own per-chunk embedding, so we skip the full n_prompt*D allocation.
    // At 512K tokens × D=2048, full embed = 4GB — would OOM on 13GB box.
    const char *_chunk_env = getenv("WUBU_CHUNK_PREFILL");
    int _chunk_check = _chunk_env ? atoi(_chunk_env) : 4096;
    bool _use_chunked = (_chunk_check > 0 && n_prompt > _chunk_check);
    float *embd = NULL;
    if (!_use_chunked) {
        size_t full_embd_bytes = (size_t)n_prompt * D * sizeof(float);
        size_t avail = wubu_mem_detect_available_ram();
        if (avail > 0 && full_embd_bytes > avail / 2) {
            /* Embedding too large — force chunked prefill path */
            _use_chunked = true;
            fprintf(stderr, "[oom-guard] embed %zuMB > %zuMB avail, using chunked prefill\n",
                    full_embd_bytes / (1024*1024), avail / (1024*1024));
        }
    }
    if (!_use_chunked) {
        embd = (float *)malloc(n_prompt * D * sizeof(float));
    }
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        /* The external embedding dump is OPTIONAL (data/qwen36_embeddings_c.bin.raw
         * was cold-archived off the SSD). If it is gone, fall back to the model's
         * own token_embd — read_embedding already handles mdl->token_embd. */
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) {
            fprintf(stderr, "[embed] external embedding dump missing; using model token_embd\n");
            mdl.use_embedding_file = false;
        }
    }
    for (int i = 0; i < n_prompt; i++)
        if (!read_embedding(&mdl, prompt_tokens[i], embd + i * D, emb_file))
            memset(embd + i * D, 0, D * sizeof(float));

    /* KB7 prefix cache (doc 010): before prefill, check whether the prompt
     * prefix matches a previously-cached KV prefix. On hit, we skip recompute
     * of the matched tokens (the engine's KV cache is still populated from
     * a prior call -- we only count the *unique* suffix).
     * A06: Also check the LMCache file-backed persistent KV cache. */
    static wubu_prefix_cache_t *g_prefix_cache = NULL;
    static wubu_lmcache_t *g_lmcache = NULL;
    int prefix_skip = 0;
    if (getenv("WUBU_PREFIX_CACHE") != NULL) {
        if (!g_prefix_cache) g_prefix_cache = wubu_prefix_cache_create();
        int blk_dummy[WUBU_PREFIX_MAX_LEN / 16];
        prefix_skip = wubu_prefix_cache_match(g_prefix_cache, prompt_tokens, n_prompt,
                                              blk_dummy, WUBU_PREFIX_MAX_LEN / 16);
        if (prefix_skip > 0) {
            fprintf(stderr, "[prefix-cache] HIT: %d tokens reused (out of %d)\n",
                    prefix_skip, n_prompt);
        } else {
            /* Register the new prefix for future hits */
            wubu_prefix_cache_register(g_prefix_cache, prompt_tokens, n_prompt,
                                       NULL, 16);
        }
    }
    /* A06: LMCache file-backed persistence */
    if (getenv("WUBU_LMCACHE") != NULL) {
        if (!g_lmcache) {
            const char *dir = getenv("WUBU_LMCACHE_DIR");
            if (!dir) dir = "/tmp/wubu_lmcache";
            g_lmcache = wubu_lmcache_create(dir, /*n_layers=*/2, 16, 128, 8);
        }
        if (g_lmcache) {
            /* Try to load KV from persistent cache — OOM guard:
             * n_prompt * 2 * 128 * 8 * 4 bytes = n_prompt * 8KB.
             * At 512K tokens = 4GB — cap to available / 4. */
            size_t avail = wubu_mem_detect_available_ram();
            int max_cached = n_prompt;
            if (avail > 0) {
                int cap = (int)(avail / 4 / (2 * 128 * 8 * sizeof(float)));
                if (cap < 64) cap = 64;
                if (max_cached > cap) {
                    max_cached = cap;
                    fprintf(stderr, "[oom-guard] LMCache capped to %d tokens\n", max_cached);
                }
            }
            float *cached_kv = (float *)malloc((size_t)max_cached * 2 * 128 * 8 * sizeof(float));
            if (cached_kv) {
                int n_cached = wubu_lmcache_load(g_lmcache, "model", prompt_tokens, n_prompt,
                                                  cached_kv, n_prompt / 16);
                if (n_cached > 0) {
                    fprintf(stderr, "[lmcache] HIT: %d blocks loaded from cache\n", n_cached);
                }
                free(cached_kv);
            }
        }
    }
    
    // DUMP_EMBEDDING_DIR: dump embedding output for 1:1 parity comparison
    {
        const char *dump_emb_dir = getenv("DUMP_EMBEDDING_DIR");
        if (dump_emb_dir && dump_emb_dir[0]) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/embedding.bin", dump_emb_dir);
            FILE *fp = fopen(fname, "wb");
            if (fp) {
                fwrite(embd, sizeof(float), n_prompt * D, fp);
                fclose(fp);
                fprintf(stderr, "DUMP_EMBEDDING_DIR: wrote %d floats to %s\n", n_prompt * D, fname);
            }
        }
    }

    // Prefill: logits or hidden states
    // OOM guard: n_prompt * vocab_size * sizeof(float) can be 500GB at 512K context.
    // For decode, we only need the LAST token's logits. For chunked prefill,
    // the function fills the caller's logits buffer with the last chunk only.
    // Strategy: allocate only chunk_sz * vocab_size (or 1 * vocab_size for decode).
    float *logits = NULL;
    {
        size_t avail = wubu_mem_detect_available_ram();
        size_t full_logits_bytes = (size_t)n_prompt * vs * sizeof(float);
        size_t safe_logits_bytes = full_logits_bytes;
        int chunk_sz_logits = getenv("WUBU_CHUNK_PREFILL")
                              ? atoi(getenv("WUBU_CHUNK_PREFILL")) : 4096;
        if (chunk_sz_logits <= 0) chunk_sz_logits = n_prompt;
        /* AirLLM layer streaming: force chunked prefill even for short prompts
         * when the budget system determined KV cache exceeds RAM budget. */
        if (mdl.use_layer_stream && chunk_sz_logits < n_prompt) {
            fprintf(stderr, "[airllm] layer streaming active, forcing chunked prefill\n");
        }
        if (chunk_sz_logits > 0 && n_prompt > chunk_sz_logits) {
            /* Already chunked — use chunk size for logits */
            safe_logits_bytes = (size_t)chunk_sz_logits * vs * sizeof(float);
        }
        size_t chunked_bytes = (size_t)chunk_sz_logits * vs * sizeof(float);
        if (avail > 0 && full_logits_bytes > avail / 4) {
            /* Full logits would OOM — use chunk-sized or single-token */
            if (n_prompt > chunk_sz_logits) {
                safe_logits_bytes = chunked_bytes;
                fprintf(stderr, "[oom-guard] logits %zuMB > %zuMB avail, capping to chunk %d (%zuMB)\n",
                        full_logits_bytes / (1024*1024), avail / 4 / (1024*1024),
                        chunk_sz_logits, safe_logits_bytes / (1024*1024));
            } else {
                /* Even single chunk too big — use just the last token */
                safe_logits_bytes = (size_t)vs * sizeof(float);
                fprintf(stderr, "[oom-guard] logits %zuMB > %zuMB avail, using last-token only\n",
                        full_logits_bytes / (1024*1024), avail / 4 / (1024*1024));
            }
        }
        /* The forward functions write to logits at offset (n_prompt-1)*vs for the
         * last token. For chunked prefill, the last chunk's logits go to logits[0..C*vs].
         * So we allocate chunk_sz*vs or vs (single token), NOT n_prompt*vs. */
        logits = (float *)malloc(safe_logits_bytes ? safe_logits_bytes : 16);
    }
    double t0 = clock_seconds();

    /* F02: Boot-time GEMV equivalence check (verified before first inference). */
    if (getenv("WUBU_SMT_CHECK") != NULL) {
        wubu_smt_result_t smt = wubu_smt_check_gemv(4, 0.1f);
        fprintf(stderr, "[smt] GEMV K=4: %s (%d checks, %d failures, max_err=%.2e)\n",
                wubu_smt_status_str(smt.status), smt.n_checks, smt.n_failures,
                (double)smt.max_error);
        if (smt.status != WUBU_SMT_OK) {
            fprintf(stderr, "[smt] WARNING: GEMV verification failed — results may be incorrect\n");
        }
    }

    if (use_gpu) {
        // GPU path: forward saves hidden states, GPU does output proj
        mdl.skip_output_proj = true;
        mdl.enable_moe = true;
        /* D04: Chunked prefill for 512K context — process prompt in
         * memory-bounded chunks to avoid 30-40 GB peak allocation.
         * Each chunk carries SSM/GQA state across calls. */
        int chunk_sz = getenv("WUBU_CHUNK_PREFILL")
                       ? atoi(getenv("WUBU_CHUNK_PREFILL")) : 4096;
        if (n_prompt > chunk_sz) {
            /* Only last chunk's logits needed; intermediate chunks free their buffers.
             * For GPU path, we need ALL hidden states — so use chunk_sz = n_prompt
             * with WUBU_CHUNK_PREFILL=0 to force single-pass. */
            wubu_model_forward_chunked(&mdl, prompt_tokens, 1, n_prompt, chunk_sz, logits);
        } else {
            wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
        }
        // skip_output_proj=true writes hidden states (D_MODEL per token) to logits buffer
        // Copy correctly: hidden states are at D_MODEL stride, not vocab_size stride
        float *hidden_batch = (float *)malloc(n_prompt * D * sizeof(float));
        for (int i = 0; i < n_prompt; i++)
            memcpy(hidden_batch + i * D, logits + i * D, D * sizeof(float));
        if (n_prompt > 0)
            gpu_output_project_batch(hidden_batch, logits, n_prompt);
        free(hidden_batch);
    } else {
        mdl.skip_output_proj = false;
        /* D04: Chunked prefill for large contexts (512K) to bound peak memory.
         * WUBU_CHUNK_PREFILL=0 disables (single-shot). */
        const char *chunk_env = getenv("WUBU_CHUNK_PREFILL");
        int chunk_sz = chunk_env ? atoi(chunk_env) : 4096;
        if (chunk_sz > 0 && n_prompt > chunk_sz) {
            wubu_model_forward_chunked(&mdl, prompt_tokens, 1, n_prompt, chunk_sz, logits);
        } else {
            wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
        }
    }

    // Dump logits if DUMP_LOGITS env var set
    const char *dump_logits_path = getenv("DUMP_LOGITS");
    if (dump_logits_path) {
        FILE *df = fopen(dump_logits_path, "wb");
        if (df) {
            // Last token's logits
            float *last_logits = logits + (n_prompt - 1) * vs;
            fwrite(last_logits, sizeof(float), vs, df);
            fclose(df);
            fprintf(stderr, "Dumped logits to %s\n", dump_logits_path);
        }
    }

    double t_prefill = clock_seconds() - t0;

    float *last_logits = logits + (n_prompt - 1) * vs;
    int generated = 0;

    { char *ibuf = hf_tok ? wubu_tok_hf_decode(hf_tok, prompt_tokens, n_prompt)
                                : NULL;
      if (ibuf) { printf("wubu: input = %s\n", ibuf); free(ibuf); } }

    // Decode loop
    xrng_t rng = { { 0x9E3779B97F4A7C15ULL, 0xD1B54A32D192ED03ULL } };
    if (getenv("SEED")) { uint64_t s = (uint64_t)atoll(getenv("SEED")); rng.s[0] ^= s; rng.s[1] ^= s * 0x9E3779B97F4A7C15ULL; }

    /* KB5: speculative decode (doc 018). When WUBU_SPEC_DECODE=1 is set, route
     * the decode loop through wubu_generate (n-gram drafter + target verify).
     * Output is provably identical to plain argmax decode; fewer forward calls. */
    if (getenv("WUBU_SPEC_DECODE") != NULL) {
        wubu_generate_cfg_t cfg = {0};
        cfg.max_tokens  = max_tokens;
        cfg.spec_k      = getenv("WUBU_SPEC_K") ? atoi(getenv("WUBU_SPEC_K")) : 4;
        cfg.ngram_order = getenv("WUBU_NGRAM_ORDER") ? atoi(getenv("WUBU_NGRAM_ORDER")) : 3;
        cfg.greedy      = (gen_temp <= 0.0f);
        cfg.temperature = gen_temp;
        cfg.seed        = (unsigned)rng.s[0];

        int *spec_out = (int *)malloc((size_t)max_tokens * sizeof(int));
        if (!spec_out) { fprintf(stderr, "spec alloc failed\n"); return 1; }
        t0 = clock_seconds();
        int spec_n = wubu_generate(&mdl, prompt_tokens, n_prompt, &cfg, spec_out);
        double t_spec = clock_seconds() - t0;
        generated = spec_n > max_tokens ? max_tokens : spec_n;
        for (int i = 0; i < generated; i++) {
            int tok_id = spec_out[i];
            char *piece = hf_tok ? wubu_tok_hf_decode(hf_tok, &tok_id, 1) : NULL;
            int n_chars = piece ? (int)strlen(piece) : 0;
            if (n_chars > 0) fwrite(piece, 1, n_chars, stdout);
            else printf("<%d>", tok_id);
            if (piece) free(piece);
        }
        fflush(stdout);
        free(spec_out);
        printf("\n");
        wubu_print_section("Stats");
        wubu_print_stat("Prefill", "%d tok in %.2fs (%.1f tok/s)", n_prompt, t_prefill, n_prompt / t_prefill);
        if (generated > 0 && t_spec > 0)
            wubu_print_stat("Decode", "%d tok in %.2fs (%.1f tok/s) [n-gram spec-k=%d]",
                   generated, t_spec, generated / t_spec, cfg.spec_k);
        free(logits); free(embd);
        if (emb_file) fclose(emb_file);
        if (hf_tok) wubu_tok_hf_free(hf_tok);
        else        wubu_tokenizer_free(&tok);
        if (rep) wubu_rep_free(rep);
        gpu_output_cleanup();
        wubu_model_free(&mdl);
        return 0;
    }

    /* G01: EAGLE self-draft speculative decode.
     * Uses a truncated model (draft_layers) as a draft model.
     * Draft model runs ~3x faster than target → if accuracy high,
     * most draft tokens accepted → up to 3x speedup. */
    if (getenv("WUBU_EAGLE") != NULL) {
        wubu_eagle_draft_t draft = {0};
        int draft_layers = getenv("WUBU_EAGLE_LAYERS")
                           ? atoi(getenv("WUBU_EAGLE_LAYERS"))
                           : (mdl.n_layers >= 32 ? mdl.n_layers / 3 : 1);
        if (wubu_eagle_draft_init(&draft, &mdl, draft_layers) == 0) {
            t0 = clock_seconds();
            int *eagle_out = (int *)malloc((size_t)max_tokens * sizeof(int));
            if (eagle_out) {
                int eagle_n = wubu_eagle_speculative_decode(&draft, &mdl,
                                                               prompt_tokens, n_prompt,
                                                               eagle_out, max_tokens);
                double t_eagle = clock_seconds() - t0;
                for (int i = 0; i < eagle_n; i++) {
                    int tok_id = eagle_out[i];
                    char *piece = hf_tok ? wubu_tok_hf_decode(hf_tok, &tok_id, 1) : NULL;
                    int n_chars = piece ? (int)strlen(piece) : 0;
                    if (n_chars > 0) fwrite(piece, 1, n_chars, stdout);
                    else printf("<%d>", tok_id);
                    if (piece) free(piece);
                }
                fflush(stdout);
                printf("\n");
                wubu_print_section("Stats");
                wubu_print_stat("Prefill", "%d tok in %.2fs (%.1f tok/s)", n_prompt, t_prefill, n_prompt / t_prefill);
                if (eagle_n > 0 && t_eagle > 0)
                    wubu_print_stat("Decode", "%d tok in %.2fs (%.1f tok/s) [eagle-%dlayers]",
                           eagle_n, t_eagle, eagle_n / t_eagle, draft_layers);
                free(eagle_out);
                free(logits); free(embd);
                if (emb_file) fclose(emb_file);
                if (hf_tok) wubu_tok_hf_free(hf_tok);
                else        wubu_tokenizer_free(&tok);
                if (rep) wubu_rep_free(rep);
                gpu_output_cleanup();
                wubu_model_free(&mdl);
                return 0;
            }
        } else {
            fprintf(stderr, "EAGLE init failed, falling back to plain decode\n");
        }
    }

    /* Plain decode loop */
    while (generated < max_tokens && !g_stop) {
        // Suppress repetitions (repeat_penalty + DRY) BEFORE sampling.
        if (rep) wubu_rep_apply(rep, last_logits);

        int next_token = sample_token(&rng, last_logits, vs,
                                       gen_temp, gen_top_p, gen_top_k);
        if (next_token == tok.eos_id || next_token == tok.bos_id) break;

        // Record the chosen token so future repetitions are penalized.
        if (rep) wubu_rep_observe(rep, next_token);

        char *piece = hf_tok ? wubu_tok_hf_decode(hf_tok, &next_token, 1) : NULL;
        int n_chars = piece ? (int)strlen(piece) : 0;
        if (n_chars > 0) fwrite(piece, 1, n_chars, stdout);
        else printf("<%d>", next_token);
        fflush(stdout);
        if (piece) free(piece);

        float x_next[D_MODEL];
        if (!read_embedding(&mdl, next_token, x_next, emb_file))
            memset(x_next, 0, D_MODEL * sizeof(float));

        /* HW-accel: game frame-budget. Begin a decode "frame"; if the step would
         * overrun the budget, skip optional extra work (none here, but Speculative
         * decode / extra drafting would check wubu_gamebud_can_spend). */
        int frame_id = mdl.gamebud ? wubu_gamebud_begin((wubu_gamebud_t *)mdl.gamebud) : 0;
        (void)frame_id;

        if (use_gpu) {
            mdl.skip_output_proj = true;
            wubu_model_forward_from_embd(&mdl, x_next, 1, 1, logits);
            // skip_output_proj=true writes hidden state (D_MODEL) to logits[0..D-1]
            // gpu_output_project reads from input buffer and writes to output buffer
            // Use separate buffer for hidden state
            float hidden_state[D_MODEL];
            memcpy(hidden_state, logits, D_MODEL * sizeof(float));
            gpu_output_project(hidden_state, logits);
        } else {
            mdl.skip_output_proj = false;
            wubu_model_forward_from_embd(&mdl, x_next, 1, 1, logits);
        }

        if (mdl.gamebud) {
            /* Bill the real wall-clock of this frame. */
            uint64_t us_used = wubu_gamebud_elapsed_us((wubu_gamebud_t *)mdl.gamebud);
            if (us_used == 0) us_used = 1;
            wubu_gamebud_end((wubu_gamebud_t *)mdl.gamebud, us_used);
        }
        last_logits = logits;
        generated++;
    }
    printf("\n");

    double t_total = clock_seconds() - t0;
    printf("\n");
    wubu_print_section("Stats");
    wubu_print_stat("Prefill", "%d tok in %.2fs (%.1f tok/s)", n_prompt, t_prefill, n_prompt / t_prefill);
    double t_decode = t_total - t_prefill;
    if (generated > 0 && t_decode > 0)
        wubu_print_stat("Decode", "%d tok in %.2fs (%.1f tok/s)", generated, t_decode, generated / t_decode);

    /* HW-accel stats */
    if (mdl.gamebud) {
        uint64_t gf = 0, go = 0, ga = 0, gp = 0, gt = 0;
        wubu_gamebud_stats((const wubu_gamebud_t *)mdl.gamebud,
                           &gf, &go, &ga, &gp, &gt);
        wubu_print_stat("HW-accel", "gamebud frames=%llu overruns=%llu avg=%.1fus peak=%.1fus",
               (unsigned long long)gf, (unsigned long long)go,
               (double)ga, (double)gp);
    }

    free(logits); free(embd);
    if (emb_file) fclose(emb_file);
    /* When using HF tokenizer, tok is only a shim — do NOT call
     * wubu_tokenizer_free() on it; that walks uninitialized pointers
     * and causes munmap_chunk() in decode. Free the real HF tokenizer
     * instead and skip the shim destructor entirely. */
    if (hf_tok) {
        wubu_tok_hf_free(hf_tok);
    } else {
        wubu_tokenizer_free(&tok);
    }
    if (rep) wubu_rep_free(rep);
    gpu_output_cleanup();
    wubu_model_free(&mdl);
    return 0;
}

#include <unistd.h>
#include <sys/stat.h>
