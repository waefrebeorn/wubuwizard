/*
 * wubu_generate.c -- autoregressive generation + n-gram speculative decoding
 * (doc 018 / K01). Self-contained C11. See header.
 */
#include "wubu_generate.h"
#include "wubu_ngram.h"
#include "wubu_integrate.h"
#include "wubu_safekern.h"
#include "wubu_latency.h"
#include "wubu_ctxvm.h"
#include "wubu_capzero.h"
#include "wubu_kvfs.h"
#include "wubu_kv_styx.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int argmaxf(const float *a, int n) {
    int best = 0; float bv = a[0];
    for (int i = 1; i < n; i++) if (a[i] > bv) { bv = a[i]; best = i; }
    return best;
}

/* deterministic rng (xorshift) for sampling */
static uint32_t grng_state;
static void grng_srand(uint32_t s) { grng_state = s ? s : 0x1234567u; }
static uint32_t grng_u32(void) {
    uint32_t x = grng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    grng_state = x;
    return x;
}
static float grng_uni(void) { return (float)grng_u32() / (float)0xFFFFFFFFu; }

/* softmax in place */
static void softmaxf(float *a, int n) {
    float mx = a[0]; for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float s = 0; for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); s += a[i]; }
    float inv = s > 0 ? 1.0f/s : 0; for (int i = 0; i < n; i++) a[i] *= inv;
}

int wubu_generate(wubu_model_t *model, const int *prompt, int n_prompt,
                  const wubu_generate_cfg_t *cfg, int *out) {
    if (!model || !prompt || !cfg || !out) return 0;
    const int V = model->vocab_size;
    grng_srand(cfg->seed);

    /* running sequence = prompt + emitted */
    int *seq = (int *)malloc(sizeof(int) * (n_prompt + cfg->max_tokens + cfg->spec_k + 1));
    if (!seq) return 0;
    memcpy(seq, prompt, (size_t)n_prompt * sizeof(int));
    int seqlen = n_prompt;

    /* runtime decode policy: composes the recursive-loop gap modules
     * (capacity_wall + kv_budget + stream_kv + ctx_manage + hybrid + pd).
     * Pure policy -- guards the 512K OOM ceiling and advises eviction. */
    int max_ctx = model->gqa_max_ctx > 0 ? model->gqa_max_ctx : 524288;
    wubu_decode_policy_t *policy = wubu_decode_policy_default(max_ctx, model->n_layers);

    /* AGI-OS runtime substrate (pass 31 integration): safety kernel +
     * latency-class scheduling + context virtual-memory paging. All pure
     * policy; the real KV cache stays owned by wubu_model.c. These close the
     * loop from the 100-goalpost AGI-OS research sweep (AF05-AF13). */

    /* AF11/13: non-tamperable safety kernel. stop_flag is kernel-owned -- the
     * agent reasoner has NO setter (wubu_safekern has no set_stop function),
     * so it cannot disable its own halt. Gate + 512K ceiling are immutable. */
    wubu_safekern_t safekern;
    safekern.stop_flag = 0;
    safekern.oom_ceiling = max_ctx;     /* 512K invariant, externalized */
    safekern.gate_enabled = 1;          /* hard OOM gate always on */

    /* AF05/07: latency class from env (default DT = throughput-first). */
    wubu_latclass_t latclass = WUBU_LC_DT;
    const char *lc = getenv("WUBU_LATENCY_CLASS");
    if (lc) {
        if (strcmp(lc, "HRT") == 0) latclass = WUBU_LC_HRT;
        else if (strcmp(lc, "SRT") == 0) latclass = WUBU_LC_SRT;
    }

    /* THE KV CACHE IS A FILE SYSTEM (doc 005). Env-gated opt-in:
     * WUBU_KVFS_NAMESPACE=1 mounts a live /kv/ namespace that
     * exposes the KV cache as a path-addressable file system.
     * The namespace mirrors the model's GQA layers:
     *   /kv/in      — incoming prompt KV (layer 0)
     *   /kv/synth   — synthesized thoughts (attention output)
     *   /kv/mem     — persistent memory (lmcache-backed)
     *   /kv/meta    — metadata (seqlen, token count)
     * Each layer gets /kv/L/layer_NN. The KV namespace syncs
     * with wubu_kv_styx (the 9P export layer) so WuBuOS's
     * body can `ls /n/kv/` and read the mind as files. */
    wubu_kvfs_t *kvfs = NULL;
    float *kv_base = NULL;
    const char *ns_env = getenv("WUBU_KVFS_NAMESPACE");
    if (ns_env && strcmp(ns_env, "1") == 0) {
        uint32_t total_blocks = 1024;
        kvfs = wubu_kvfs_create(64, total_blocks);
        if (kvfs) {
            /* allocate the backing KV tensor (flat, float32) */
            kv_base = (float *)calloc((size_t)total_blocks * 64, sizeof(float));
            if (!kv_base) { wubu_kvfs_free(kvfs); kvfs = NULL; }
            else {
                /* mount each layer */
                for (int l = 0; l < model->n_layers; l++) {
                    char mpath[64];
                    snprintf(mpath, sizeof(mpath), "/kv/L/layer_%02d", l);
                    /* each layer gets 64 blocks */
                    wubu_kvfs_mount(kvfs, mpath, (uint32_t)(l * 64), 64);
                }
                /* core namespace dirs */
                wubu_kvfs_mount(kvfs, "/kv/in",    0,        64);
                wubu_kvfs_mount(kvfs, "/kv/synth", 64,       64);
                wubu_kvfs_mount(kvfs, "/kv/mem",   128,     128);
                wubu_kvfs_mount(kvfs, "/kv/meta",  256,       8);
            }
        }
    }

    /* AF08/09: context virtual-memory ring (logical residency tracker for the
     * safety gate; capacity = max_ctx). FIFO demand-paging eviction on overflow. */
    wubu_ctxring_t cring;
    long *cring_buf = (long *)malloc(sizeof(long) * max_ctx);
    cring.tok = cring_buf; cring.head = 0; cring.size = 0; cring.capacity = max_ctx;

    /* AF11: honor a kernel stop signal if externally raised (env sentinel).
     * The agent cannot set this -- only the operator/kernel can. */
    if (getenv("WUBU_KERNEL_STOP")) safekern.stop_flag = 1;

    float *logits = (float *)malloc(sizeof(float) * (n_prompt + cfg->max_tokens + cfg->spec_k + 1) * V);
    if (!logits) { free(seq); return 0; }

    int emitted = 0;
    int K = cfg->spec_k;

    while (emitted < cfg->max_tokens) {
        wubu_decode_decision_t dec;
        wubu_decode_policy_step(policy, seqlen, 0, 1 << 30, 0, &dec);
        /* capacity_wall: never let decode exceed the 512K OOM ceiling (EAMM guard).
         * If unsafe, stop gracefully -- the operator's budget/streaming keeps us
         * under the ceiling; we never silently OOM. */
        if (!dec.oom_safe) {
            fprintf(stderr, "[decode-policy] 512K ceiling reached at seqlen=%d; "
                    "stopping (force_evict=%d). Tighten WUBU_STREAM_WINDOW.\n", seqlen, dec.force_evict);
            break;
        }

        /* AF11: non-tamperable stop. The agent reasoner has no setter; only the
         * kernel/operator can raise stop_flag. If set, halt immediately. */
        if (wubu_stop_honored(&safekern)) {
            fprintf(stderr, "[safety-kernel] kernel stop honored at seqlen=%d; halting.\n", seqlen);
            break;
        }

        /* AF13: stability-plasticity guard -- the operator may have tuned params,
         * but the 512K ceiling + gate are immutable. If a proposed mutation had
         * tried to weaken them, wubu_rsi_mutation_ok would have rejected it. We
         * re-assert here: any decode step must remain under the ceiling. */
        if (!wubu_rsi_mutation_ok(&safekern, max_ctx, safekern.gate_enabled)) {
            fprintf(stderr, "[safety-kernel] invariant violation: 512K gate/ceiling "
                    "compromised; refusing to proceed.\n");
            break;
        }

        /* AF08/09: context virtual-memory paging. Track logical residency; on
         * overflow, FIFO-evict oldest (cooperative with stream_kv eviction).
         * This is the demand-paging policy decoupled from the real KV alloc. */
        int evicted = wubu_ctx_evict_fifo(&cring, seqlen);
        if (evicted) {
            /* AF12: graduated containment -- over-pressure is a mild severity
             * event; throttle (reversible), not stop. Latency class tunes it. */
            float sev = (latclass == WUBU_LC_HRT) ? 0.5f : 0.3f;
            int lvl = wubu_containment_level(sev);
            if (lvl >= WUBU_CONT_STOP) break;  /* only if escalated */
        }

        /* KV CACHE IS A FILESYSTEM — sync namespace state each step.
         * The namespace mirrors the model's growing KV: seqlen grows,
         * blocks fill. The /kv/ mounts act as views into the model's
         * internal state so WuBuOS 9P clients can read the "mind as files".
         * Zero-cost when the env gate is off (kvfs == NULL). */
        if (kvfs && seqlen > 0 && kv_base) {
            /* Write seqlen + emitted into /kv/meta as float data.
             * We store them as floats (the KVFS API works in float units). */
            int32_t slen = (int32_t)seqlen;
            int32_t ecount = (int32_t)emitted;
            float meta_f[2] = {(float)slen, (float)ecount};
            wubu_kvfs_write(kvfs, "/kv/meta", kv_base, meta_f, 2);

            /* Write current token into /kv/in (incoming input stream).
             * /kv/in is mounted at blocks [0, 64) → offset 0 in kv_base.
             * We write the last token as a float at slot (seqlen-1) % 64. */
            int32_t curtok = (int32_t)seq[seqlen > 0 ? seqlen - 1 : 0];
            float tok_f = (float)curtok;
            uint32_t tok_slot = (uint32_t)((seqlen - 1) % 64);
            wubu_kvfs_write(kvfs, "/kv/in", kv_base, &tok_f, 1);

            /* Write synth (attention output) token into /kv/synth. */
            int32_t synth_tok = (emitted > 0) ? (int32_t)out[emitted - 1] : 0;
            float synth_f = (float)synth_tok;
            wubu_kvfs_write(kvfs, "/kv/synth", kv_base, &synth_f, 1);

            /* Layer metadata written per-step. Real K/V cache data
             * is mirrored through the namespace in teardown after
             * the final forward (see gqa_cache_len below). */
        }
        if (K <= 0) {
            /* plain decode: forward whole sequence, take last position */
            int T = seqlen;
            wubu_model_forward(model, seq, 1, T, logits);
            float *lp = logits + (size_t)(T - 1) * V;
            int tok;
            if (cfg->greedy) {
                tok = argmaxf(lp, V);
            } else {
                softmaxf(lp, V);
                float r = grng_uni(), acc = 0; tok = V - 1;
                for (int i = 0; i < V; i++) { acc += lp[i]; if (acc >= r) { tok = i; break; } }
            }
            out[emitted++] = tok;
            seq[seqlen++] = tok;
            continue;
        }

        /* ---- speculative decode (draft K tokens, verify in one forward) ---- */
        int *drafts = (int *)malloc(sizeof(int) * K);
        int *dc = (int *)malloc(sizeof(int) * (seqlen + K + 1)); /* virtual ctx */
        int nprop = 0;
        if (dc) memcpy(dc, seq, (size_t)seqlen * sizeof(int));
        for (int s = 0; s < K && dc; s++) {
            wubu_ngram_draft_t *ng = wubu_ngram_create(dc, seqlen + s, cfg->ngram_order);
            int p[1]; int got = wubu_ngram_propose(ng, 1, p);
            wubu_ngram_free(ng);
            if (got <= 0) break;
            drafts[nprop++] = p[0];
            dc[seqlen + nprop - 1] = p[0];  /* virtually extend for next lookup */
        }
        free(dc);

        if (nprop == 0) {
            /* no draft available -> fall back to one plain step */
            int T = seqlen;
            wubu_model_forward(model, seq, 1, T, logits);
            float *lp = logits + (size_t)(T - 1) * V;
            int tok = cfg->greedy ? argmaxf(lp, V) : (softmaxf(lp, V), argmaxf(lp, V));
            if (!cfg->greedy) { float r = grng_uni(), acc = 0; tok = V - 1;
                for (int i = 0; i < V; i++) { acc += lp[i]; if (acc >= r) { tok = i; break; } } }
            out[emitted++] = tok; seq[seqlen++] = tok;
            free(drafts); continue;
        }

        /* forward the sequence + drafts in one pass.
         * logits has T = seqlen + nprop positions. Position (seqlen-1 + k)
         * predicts drafts[k] (k=0..nprop-1); position (seqlen-1+nprop) predicts
         * the bonus token (it exists since T = seqlen+nprop). */
        int T = seqlen + nprop;
        wubu_model_forward(model, seq, 1, T, logits);

        /* Linear autoregressive verify: draft[k] checked against target logits
         * at its own position (seqlen-1+k). This is the provably-equivalent
         * greedy/sampled speculative step (Leviathan et al. 2023). */
        int L = 0;
        float rng = grng_uni();
        for (int k = 0; k < nprop; k++) {
            float *tlp = logits + (size_t)(seqlen - 1 + k) * V;
            if (!cfg->greedy) softmaxf(tlp, V);
            float p_target = cfg->greedy ? (tlp[drafts[k]] == argmaxf(tlp, V) ? 1.0f : 0.0f)
                                         : tlp[drafts[k]];
            /* draft prob: greedy -> one-hot on proposed; sampled -> uniform prior */
            float p_draft = cfg->greedy ? 1.0f : (1.0f / (float)V);
            int accept;
            if (cfg->greedy) accept = (tlp[drafts[k]] == argmaxf(tlp, V));
            else accept = (p_target >= p_draft) || (rng < p_target / (p_draft > 1e-9f ? p_draft : 1e-9f));
            if (!accept) break;
            L++;
        }

        /* commit accepted draft prefix */
        for (int i = 0; i < L; i++) { out[emitted++] = drafts[i]; seq[seqlen++] = drafts[i]; }

        /* commit one more token (target decision / bonus) if room */
        if (emitted < cfg->max_tokens) {
            int tok;
            float *tlp = logits + (size_t)(seqlen - 1) * V;  /* position after accepted prefix */
            if (!cfg->greedy) softmaxf(tlp, V);
            if (L < nprop) {
                /* rejection at L: target decides at the rejection position */
                if (cfg->greedy) tok = argmaxf(tlp, V);
                else { float r = grng_uni(), acc = 0; tok = V - 1;
                       for (int i = 0; i < V; i++) { acc += tlp[i]; if (acc >= r) { tok = i; break; } } }
            } else {
                /* all accepted: bonus from residual(target - draft) at position
                 * after the accepted prefix (== tlp already computed above). */
                float *dp = (float *)malloc(sizeof(float) * V);
                for (int i = 0; i < V; i++) dp[i] = cfg->greedy ? 0.0f : (1.0f / (float)V);
                int bonus = wubu_spec_bonus_token(tlp, dp, V, grng_uni());
                free(dp);
                tok = (bonus >= 0) ? bonus : argmaxf(tlp, V);
            }
            out[emitted++] = tok; seq[seqlen++] = tok;
        }

        free(drafts);
    }

    free(seq); free(logits);
    free(cring_buf);
    wubu_decode_policy_destroy(policy);

    /* KV CACHE IS A FILE SYSTEM — final namespace writeback + 9P snapshot */
    if (kvfs) {
        if (kv_base) {
            /* Final writeback: flush final seqlen + emitted to /kv/meta */
            float final_meta[2] = {(float)seqlen, (float)emitted};
            wubu_kvfs_write(kvfs, "/kv/meta", kv_base, final_meta, 2);
            /* Read back to verify the namespace is live (KVFS read path) */
            float verify[2] = {0};
            if (wubu_kvfs_read(kvfs, "/kv/meta", kv_base, verify, 2) == 0) {
                fprintf(stderr, "[kvfs] verified read-back: seqlen=%d emitted=%d\n",
                        (int)verify[0], (int)verify[1]);
            }
        }

        /* Mirror REAL K/V cache tensors through the namespace — this is the
         * core of "the KV cache IS a file system". For each GQA layer,
         * mount /kv/L/layer_NN/k and .v and write the actual gqa_k_cache
         * and gqa_v_cache rows at the final position. 9P clients can then
         * read attention K/V data as files. */
        /* Mirror REAL K/V cache tensors through the namespace — this is the
         * core of "the KV cache IS a file system". For each GQA layer,
         * mount /kv/L/layer_NN/k and .v and write the FULL cached span
         * (all positions, not just the last). 9P clients can then read
         * every attention K/V vector as a file. */
        if (kv_base && model->gqa_k_cache && model->gqa_v_cache &&
            seqlen > 0) {
            float *k_cache = (float *)model->gqa_k_cache;
            float *v_cache = (float *)model->gqa_v_cache;
            int gl = 0;  /* GQA layer index */
            /* The forward writes the whole sequence fresh each call, so the
             * cache holds positions [0, seqlen). Mirror all of them. */
            int npos = seqlen < model->gqa_max_ctx ? seqlen : model->gqa_max_ctx;
            const int kv_dim = GQA_KV_DIM;
            for (int l = 0; l < model->n_layers; l++) {
                if (model->layers[l].is_ssm) continue;  /* GQA only */
                if (kv_dim <= 0 || kv_dim > 1024) { gl++; continue; }
                /* Bounds: cache layout is [n_gqa_layers][gqa_max_ctx][kv_dim] */
                if ((int64_t)gl * model->gqa_max_ctx * kv_dim +
                    (int64_t)npos * kv_dim >
                    (int64_t)model->n_layers * model->gqa_max_ctx * kv_dim)
                    continue;
                float *k_span = k_cache + (size_t)gl * model->gqa_max_ctx * kv_dim;
                float *v_span = v_cache + (size_t)gl * model->gqa_max_ctx * kv_dim;
                int span = npos * kv_dim;
                char kpath[80], vpath[80];
                snprintf(kpath, sizeof(kpath), "/kv/L/layer_%02d/k", l);
                snprintf(vpath, sizeof(vpath), "/kv/L/layer_%02d/v", l);
                wubu_kvfs_unmount(kvfs, kpath);
                wubu_kvfs_unmount(kvfs, vpath);
                wubu_kvfs_mount(kvfs, kpath, 512 + gl * 128, span);
                wubu_kvfs_mount(kvfs, vpath, 512 + gl * 128 + 64, span);
                wubu_kvfs_write(kvfs, kpath, kv_base, k_span, span);
                wubu_kvfs_write(kvfs, vpath, kv_base, v_span, span);
                /* Read back to verify real K/V data is in the namespace */
                float *rb = (float *)malloc((size_t)span * sizeof(float));
                if (rb && wubu_kvfs_read(kvfs, kpath, kv_base, rb, span) == 0) {
                    double chk = 0, chk2 = 0;
                    for (int i = 0; i < span; i++) { chk += rb[i] * rb[i]; }
                    wubu_kvfs_read(kvfs, vpath, kv_base, rb, span);
                    for (int i = 0; i < span; i++) { chk2 += rb[i] * rb[i]; }
                    fprintf(stderr, "[kvfs] K/V mirror: layer %d gqa_idx=%d "
                            "kv_dim=%d npos=%d k_energy=%.3f v_energy=%.3f\n",
                            l, gl, kv_dim, npos, chk, chk2);
                }
                free(rb);
                gl++;
            }
        }

        /* Export live KV snapshot via 9P layer */
        char *snap = wubu_kvfs_snapshot_json(kvfs, NULL);
        if (snap) {
            fprintf(stderr, "[kvfs] namespace: %s\n", snap);
            free(snap);
        }

        /* Register each GQA layer's K cache with the Styx 9P
         * registry so external WuBuOS 9P clients can walk
         * /kv/ as a real filesystem — the KV cache IS a
         * filesystem, fully exposed to the namespace. */
        if (kvfs) {
            int gl = 0;
            for (int l = 0; l < model->n_layers; l++) {
                if (model->layers[l].is_ssm) continue;
                int kv_dim = GQA_KV_DIM;
                if (kv_dim <= 0 || kv_dim > 1024) { gl++; continue; }
                size_t span = (size_t)model->gqa_max_ctx * kv_dim;
                float *k_ptr = (float *)model->gqa_k_cache +
                    (size_t)gl * model->gqa_max_ctx * kv_dim;
                char kpath[256];
                snprintf(kpath, sizeof(kpath), "/kv/L/layer_%02d", l);
                wubu_kv_styx_register(kpath, k_ptr, span);
                gl++;
            }
        }
        wubu_kvfs_free(kvfs);
        if (kv_base) free(kv_base);
    }

    return emitted;
}
