/* wubu_gemma4_model.c — Gemma 4 12B dual-head-dim ISWA forward pass.
 *
 * Port of llama.cpp gemma4 model, adapted for wubuwizard quantized matmul.
 * Architecture: 48 layers, 40 sliding-window (HEAD_DIM=256) + 8 full-attention (HEAD_DIM=512).
 * Full attention at indices 5, 11, 17, 23, 29, 35, 41, 47 (every 6th from 5).
 * KV sharing: layers 40-47 reuse KV from layers 38-39.
 *
 * Uses quantized_matmul() from quantized_matmul.c for ALL quantized matmuls —
 * SSE2/AVX2-accelerated Q4_K vec_dot, OpenMP, and Q8_K-quantized activations.
 *
 * C11 — no VLA, no compound literals.
 */

#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include "wubu_residency.h"
#include "wubu_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* ============ Tensor name helper ============ */
static char tn_buf[256];
#define TN(fmt, i) (snprintf(tn_buf, sizeof(tn_buf), fmt, i), tn_buf)

/* ============ Layer initialization ============ */

static void g4_layer_init(g4_layer_t *l, int idx) {
    memset(l, 0, sizeof(*l));
    l->layer_idx = idx;
    l->is_full = g4_layer_is_full(idx);
    l->share_kv = g4_layer_shares_kv(idx);
    l->kv_src_idx = g4_layer_kv_src(idx);
    l->head_dim = g4_head_dim(idx);
    l->q_dim = g4_q_dim(idx);
    l->kv_dim = g4_kv_dim(idx);
    l->kv_heads = g4_kv_heads(idx);
    l->n_rot = g4_n_rot(idx);
    l->rope_base = g4_rope_base(idx);
    l->kv_eq = false;
}

/* ============ Quantized weight helpers ============ */

static void qw_load(g4_qweight_t *w, gguf_ctx *ctx, const char *name) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { memset(w, 0, sizeof(*w)); return; }
    w->ggml_type = t->ggml_type;
    w->n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) w->n_elems *= t->dims[d];
    w->raw_bytes = gguf_raw_size(t->ggml_type, w->n_elems);
    uint64_t offset = t->data_offset;
    if (ctx->data_blob) {
        w->data = (const uint8_t *)ctx->data_blob + offset;
        w->stream_ctx = NULL;
        w->stream_info = NULL;
    } else {
        w->data = NULL;
        w->stream_ctx = ctx;
        w->stream_info = t;
    }
    w->data_owned = 0;
    if (w->data && (w->data < (const uint8_t *)ctx->data_blob ||
        w->data + w->raw_bytes > (const uint8_t *)ctx->data_blob + ctx->data_blob_size)) {
        fprintf(stderr, "[G4] ERROR: tensor %s out of bounds\n", name);
    }
}

static void qw_load_aliased(g4_qweight_t *w, g4_qweight_t *src) {
    memcpy(w, src, sizeof(*w));
    w->data_owned = 0;  /* alias doesn't own the data */
    /* An alias must not inherit another tensor's LRU/raw-cache state. */
    w->raw_cache = NULL;
    w->last_use = 0;
    w->cache_prev = w->cache_next = NULL;
}

static float *g4_load_norm_f32(gguf_ctx *ctx, const char *name, int expected) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) return NULL;
    int64_t n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, buf, n)) { free(buf); return NULL; }
    if (n != expected)
        printf("[G4] Warning: %s has %lld elems (expected %d)\n", name, (long long)n, expected);
    return buf;
}

/* ============ Fast batched quantized matmul ============
 *
 * Uses quantized_matmul() from quantized_matmul.c which:
 *   - Quantizes F32 input to Q8_K once per row
 *   - Uses SSE2/AVX2 vec_dot for K-quant weights (Q4_K, Q5_K, Q6_K, IQ types)
 *   - OpenMP parallelized over columns with prefetching
 *   - Falls back to dequant+SGEMM for rare types
 *
 * c[M,N] = x[M,K] @ dequant(W[K,N])
 *
 * For Q4_0 (token_embd / output head): uses dequant-to-F32 then SGEMM
 * (not critical path — LM head called once per forward)
 */
static int g4_fseek_i64(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* ---- Bounded RAM tier for streamed weights (SSD -> RAM -> compute) ----
 * G4_WEIGHT_CACHE_MB caps the QUANTIZED bytes resident in RAM. Caching raw
 * bytes (not dequantized F32) keeps the tier ~4x smaller and avoids thrash. */
static g4_qweight_t *g_cache_head, *g_cache_tail;
static size_t g_cache_bytes, g_cache_cap;
static uint64_t g_cache_clock, g_cache_hits, g_cache_misses;

static void g4_cache_unlink(g4_qweight_t *w)
{
    if (w->cache_prev) ((g4_qweight_t *)w->cache_prev)->cache_next = w->cache_next;
    else if (g_cache_head == w) g_cache_head = (g4_qweight_t *)w->cache_next;
    if (w->cache_next) ((g4_qweight_t *)w->cache_next)->cache_prev = w->cache_prev;
    else if (g_cache_tail == w) g_cache_tail = (g4_qweight_t *)w->cache_prev;
    w->cache_prev = w->cache_next = NULL;
}

static void g4_cache_touch(g4_qweight_t *w)
{
    if (w->cache_prev || w->cache_next) g4_cache_unlink(w);
    w->cache_prev = NULL;
    w->cache_next = g_cache_head;
    if (g_cache_head) g_cache_head->cache_prev = w;
    g_cache_head = w;
    if (!g_cache_tail) g_cache_tail = w;
    w->last_use = ++g_cache_clock;
}

static void g4_cache_insert(g4_qweight_t *w, uint8_t *raw, size_t bytes)
{
    g4_cache_touch(w);
    w->raw_cache = raw;
    g_cache_bytes += bytes;
    while (g_cache_bytes > g_cache_cap) {
        /* Evict the least-recently-used unpinned entry; the residency hot set
         * stays resident by construction (EG-110). */
        g4_qweight_t *old = g_cache_tail;
        while (old && old->pinned) old = (g4_qweight_t *)old->cache_prev;
        if (!old) break;
        uint8_t *victim = old->raw_cache;
        size_t vb = (size_t)old->raw_bytes;
        g4_cache_unlink(old);
        old->raw_cache = NULL;
        g_cache_bytes -= vb;
        free(victim);
    }
}

static void g4_cache_clear(void)
{
    g4_qweight_t *w = g_cache_head;
    while (w) {
        g4_qweight_t *next = (g4_qweight_t *)w->cache_next;
        free(w->raw_cache);
        w->raw_cache = NULL;
        w->cache_prev = w->cache_next = NULL;
        w = next;
    }
    g_cache_head = g_cache_tail = NULL;
    g_cache_bytes = 0;
    if (g_cache_cap)
        fprintf(stderr, "[G4] RAM cache: %llu hits, %llu misses\n",
                (unsigned long long)g_cache_hits, (unsigned long long)g_cache_misses);
    g_cache_hits = g_cache_misses = 0;
}

/* ---- One-layer-lookahead weight prefetch (Windows) ----
 * A worker with its own FILE* reads layer N+1 while the main thread computes
 * layer N. It never touches LRU state: ready buffers are adopted by the main
 * thread at the layer boundary, so the cache lists stay single-threaded. */
#ifdef _WIN32
#define G4_PF_MAX_W 7
#define G4_PF_SLOTS 32

typedef struct {
    volatile LONG state; /* 0 idle, 1 requested, 2 ready */
    int layer;
    int n;
    g4_qweight_t *w[G4_PF_MAX_W];
    uint8_t *raw[G4_PF_MAX_W];
    int64_t bytes[G4_PF_MAX_W];
} g4_pf_slot_t;

typedef struct {
    FILE *file;
    uint64_t blob_off;
    HANDLE thread, req_evt, stop_evt;
    volatile LONG stop;
    g4_pf_slot_t slot[G4_PF_SLOTS];
    uint64_t requests, ready, stale;
} g4_prefetch_t;

static void g4_pf_layer_weights(g4_model_t *m, int il,
                                g4_qweight_t **out, int *n)
{
    g4_layer_t *l = &m->layers[il];
    g4_qweight_t *cand[G4_PF_MAX_W] = {
        &l->attn_q, &l->attn_k, &l->attn_v, &l->attn_out,
        &l->ffn_gate, &l->ffn_up, &l->ffn_down
    };
    int k = 0;
    for (int i = 0; i < G4_PF_MAX_W; i++) {
        g4_qweight_t *w = cand[i];
        if (!w || !w->stream_ctx || w->raw_bytes <= 0 || w->raw_cache) continue;
        int dup = 0;
        for (int j = 0; j < k; j++) if (out[j] == w) dup = 1;
        if (!dup) out[k++] = w;
    }
    *n = k;
}

static DWORD WINAPI g4_pf_thread(LPVOID arg)
{
    g4_prefetch_t *pf = (g4_prefetch_t *)arg;
    HANDLE evts[2] = { pf->req_evt, pf->stop_evt };
    for (;;) {
        DWORD r = WaitForMultipleObjects(2, evts, FALSE, INFINITE);
        if (r == WAIT_OBJECT_0 + 1 || pf->stop) break;
        if (r != WAIT_OBJECT_0) continue;
        for (int s = 0; s < G4_PF_SLOTS; s++) {
            g4_pf_slot_t *sl = &pf->slot[s];
            if (InterlockedCompareExchange(&sl->state, 1, 1) != 1) continue;
            for (int i = 0; i < sl->n; i++) {
                g4_qweight_t *w = sl->w[i];
                sl->raw[i] = NULL;
                if (!w || !w->stream_info || w->raw_bytes <= 0 || w->raw_cache) continue;
                uint8_t *buf = (uint8_t *)malloc((size_t)w->raw_bytes);
                if (!buf) continue;
                gguf_tensor_info *t = (gguf_tensor_info *)w->stream_info;
                uint64_t off = pf->blob_off + t->data_offset;
                if (_fseeki64(pf->file, (__int64)off, SEEK_SET) != 0 ||
                    fread(buf, 1, (size_t)w->raw_bytes, pf->file) != (size_t)w->raw_bytes) {
                    free(buf);
                    continue;
                }
                sl->raw[i] = buf;
            }
            InterlockedExchange(&sl->state, 2);
            pf->ready++;
        }
    }
    return 0;
}

static const uint8_t *g4_qweight_stream_raw(const g4_qweight_t *w, int *owned);

/* ---- GPU residency (optional, G4_GPU=1) ----
 * Uploads every quantized weight that the device can hold after a reserve, so
 * decode matvecs run device-resident (docs/GPU-LANE-PLAN.md). Weights that do
 * not fit keep dev == NULL and stay on the CPU path. */
static void g4_apply_gpu_plan(g4_model_t *m)
{
    const char *env = getenv("G4_GPU");
    if (!env || !strcmp(env, "0")) return;
    if (!wubu_gpu_available()) {
        fprintf(stderr, "[GPU] requested but no CUDA device/DLL; staying on CPU\n");
        return;
    }

    unsigned long long total = 0, freeb = 0;
    wubu_gpu_stats(&total, &freeb);
    const unsigned long long reserve = 768ULL * 1024ULL * 1024ULL;  /* KV + scratch + logits */
    fprintf(stderr, "[GPU] device %llu MiB free, reserve %llu MiB\n",
            freeb >> 20, reserve >> 20);

    unsigned long long used = 0;
    int uploaded = 0, skipped = 0;

    /* Tied head first: it is the largest single tensor and pure decode cost. */
    g4_qweight_t *head = &m->output;
    if (head->data && head->raw_bytes > 0) {
        if (freeb > reserve && (unsigned long long)head->raw_bytes <= freeb - reserve) {
            head->dev = wubu_gpu_upload(head->data, (size_t)head->raw_bytes);
            if (head->dev) { used += (unsigned long long)head->raw_bytes; ++uploaded; }
        }
    }

    for (int il = 0; il < m->n_layers; ++il) {
        g4_layer_t *l = &m->layers[il];
        g4_qweight_t *w[7] = { &l->attn_q, &l->attn_k, &l->attn_v, &l->attn_out,
                               &l->ffn_gate, &l->ffn_up, &l->ffn_down };
        for (int k = 0; k < 7; ++k) {
            g4_qweight_t *q = w[k];
            if (!q->data || q->raw_bytes <= 0) continue;
            const unsigned long long bytes = (unsigned long long)q->raw_bytes;
            if (freeb <= reserve || used + bytes > freeb - reserve) { ++skipped; continue; }
            q->dev = wubu_gpu_upload(q->data, (size_t)q->raw_bytes);
            if (q->dev) { used += bytes; ++uploaded; } else { ++skipped; }
        }
    }
    fprintf(stderr, "[GPU] uploaded %d tensors (%.0f MiB), %d left on CPU\n",
            uploaded, (double)(used >> 20), skipped);
}



/* ---- EG-110 residency plan -> RAM tier cap ----
 * WUBU_RESIDENCY_MB turns the ad-hoc G4_WEIGHT_CACHE_MB cap into a plan: attention
 * and the tied embedding/output stay resident, FFN tensors stream from the GGUF,
 * and the plan fails closed (everything streams) when the hot set plus one staging
 * buffer cannot fit the budget. Requires G4_STREAM_WEIGHTS=1 to have any effect. */
static void g4_apply_residency_plan(g4_model_t *m)
{
    const char *env = getenv("WUBU_RESIDENCY_MB");
    char auto_buf[32];
    if (!env) {
        /* No explicit budget but streamed weights: derive one from physical RAM,
         * leaving a quarter for the OS, KV cache and scratch. */
#ifdef _WIN32
        MEMORYSTATUSEX ms;
        memset(&ms, 0, sizeof(ms));
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys > 0) {
            unsigned long long mb = (ms.ullTotalPhys >> 20) * 3 / 4;
            snprintf(auto_buf, sizeof(auto_buf), "%llu", mb);
            env = auto_buf;
        }
#endif
        if (!env) return;
    }
    if (!getenv("G4_STREAM_WEIGHTS")) {
        fprintf(stderr, "[RESIDENCY] ignored: weights are mmap'd (set G4_STREAM_WEIGHTS=1)\n");
        return;
    }
    long long budget_mb = atoll(env);
    if (budget_mb <= 0) return;
    const uint64_t budget = (uint64_t)budget_mb * 1024ULL * 1024ULL;

    /* Two entries per layer (attention group resident, FFN group streamed) keeps
     * the plan inside wubu_residency's own limit of 256 entries for any model up
     * to 128 layers, and expresses the real policy: keep attention hot, stream
     * FFN. */
    enum { MAXT = 2 * WUBU_RESIDENCY_MAX_LAYERS };
    wubu_residency_layer_t *layers = (wubu_residency_layer_t *)calloc(MAXT, sizeof(*layers));
    int *plan = (int *)calloc(MAXT, sizeof(*plan));
    if (!layers || !plan) { free(layers); free(plan); return; }

    int *ent_layer = (int *)calloc(MAXT, sizeof(int));
    int *ent_attn  = (int *)calloc(MAXT, sizeof(int));
    if (!ent_layer || !ent_attn) { free(ent_layer); free(ent_attn); free(layers); free(plan); return; }

    int n = 0;
    for (int il = 0; il < m->n_layers && n + 2 <= MAXT; ++il) {
        g4_layer_t *l = &m->layers[il];
        g4_qweight_t *attn[4] = { &l->attn_q, &l->attn_k, &l->attn_v, &l->attn_out };
        g4_qweight_t *ffn[3]  = { &l->ffn_gate, &l->ffn_up, &l->ffn_down };
        uint64_t abytes = 0, fbytes = 0;
        for (int k = 0; k < 4; ++k)
            if (attn[k]->stream_ctx && attn[k]->raw_bytes > 0)
                abytes += (uint64_t)attn[k]->raw_bytes;
        for (int k = 0; k < 3; ++k)
            if (ffn[k]->stream_ctx && ffn[k]->raw_bytes > 0)
                fbytes += (uint64_t)ffn[k]->raw_bytes;
        if (abytes) {
            layers[n].bytes = abytes; layers[n].hot = 1;
            ent_layer[n] = il; ent_attn[n] = 1; ++n;
        }
        if (fbytes) {
            layers[n].bytes = fbytes; layers[n].hot = 0;
            ent_layer[n] = il; ent_attn[n] = 0; ++n;
        }
    }
    if (n == 0) { free(layers); free(plan); return; }

    int rc = wubu_residency_plan(layers, n, budget, plan);
    if (rc != 0) {
        for (int i = 0; i < n; ++i) layers[i].hot = 0;
        rc = wubu_residency_plan(layers, n, budget, plan);
        fprintf(stderr, "[RESIDENCY] hot set does not fit %lld MiB; streaming every tensor\n",
                budget_mb);
    }
    if (rc != 0) {
        fprintf(stderr, "[RESIDENCY] budget %lld MiB cannot stage the largest tensor; no cap applied\n",
                budget_mb);
        free(layers); free(plan);
        return;
    }

    /* Make the hot set real: read it once and pin it so the LRU can never
     * evict attention weights for streamed FFN traffic. */
    g_cache_cap = (size_t)budget;   /* allow the hot load before trimming to staging */
    int pinned_groups = 0;
    for (int i = 0; i < n && i < MAXT; ++i) {
        if (!plan[i] || !ent_attn[i]) continue;
        g4_layer_t *l = &m->layers[ent_layer[i]];
        g4_qweight_t *attn[4] = { &l->attn_q, &l->attn_k, &l->attn_v, &l->attn_out };
        for (int k = 0; k < 4; ++k) {
            if (!attn[k]->stream_ctx || attn[k]->raw_bytes <= 0 || attn[k]->raw_cache) continue;
            int owned = 0;
            const uint8_t *raw = g4_qweight_stream_raw(attn[k], &owned);
            if (raw && !owned) { attn[k]->pinned = 1; ++pinned_groups; }
        }
    }

    uint64_t dram = wubu_residency_dram_bytes(layers, plan, n);
    uint64_t flash = wubu_residency_flash_bytes(layers, plan, n);
    uint64_t staging = budget > dram ? budget - dram : 0;
    g_cache_cap = (size_t)staging;
    fprintf(stderr, "[RESIDENCY] budget=%lld MiB hot=%llu MiB streamed=%llu MiB cache_cap=%llu MiB\n",
            budget_mb, (unsigned long long)(dram >> 20), (unsigned long long)(flash >> 20),
            (unsigned long long)(staging >> 20));
    fprintf(stderr, "[RESIDENCY] pinned %d attention tensors (%.0f MiB) in the RAM tier\n",
            pinned_groups, (double)(dram >> 20));
    free(ent_layer);
    free(ent_attn);
    free(layers);
    free(plan);
}

static void g4_prefetch_request(g4_model_t *m, int il);
static void g4_prefetch_init(g4_model_t *m, const char *path)
{
    const char *s = getenv("G4_PREFETCH");
    if (!getenv("G4_STREAM_WEIGHTS") || !g_cache_cap || (s && !strcmp(s, "0"))) return;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    g4_prefetch_t *pf = (g4_prefetch_t *)calloc(1, sizeof(*pf));
    if (!pf) { fclose(f); return; }
    pf->file = f;
    pf->blob_off = m->gguf_data_offset;
    pf->req_evt = CreateEvent(NULL, FALSE, FALSE, NULL); /* auto-reset */
    pf->stop_evt = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!pf->req_evt || !pf->stop_evt) {
        if (pf->req_evt) CloseHandle(pf->req_evt);
        if (pf->stop_evt) CloseHandle(pf->stop_evt);
        fclose(f); free(pf); return;
    }
    pf->thread = CreateThread(NULL, 0, g4_pf_thread, pf, 0, NULL);
    if (!pf->thread) {
        CloseHandle(pf->req_evt); CloseHandle(pf->stop_evt);
        fclose(f); free(pf); return;
    }
    m->prefetch = pf;
    fprintf(stderr, "[G4] one-layer weight prefetch enabled\n");
    /* Layer weights are static: stage the first G4_PF_SLOTS layers at load
     * time so the worker has a real head start before the first forward. */
    for (int il = 0; il < G4_PF_SLOTS && il < m->n_layers; il++)
        g4_prefetch_request(m, il);
}

static void g4_prefetch_request(g4_model_t *m, int il)
{
    g4_prefetch_t *pf = (g4_prefetch_t *)m->prefetch;
    if (!pf || il < 0 || il >= m->n_layers) return;
    g4_pf_slot_t *sl = NULL;
    for (int s = 0; s < G4_PF_SLOTS; s++) {
        LONG st = InterlockedCompareExchange(&pf->slot[s].state, 0, 0);
        if (st == 0) { sl = &pf->slot[s]; break; }
        if (st == 2) { /* stale ready slot: release before reuse */
            g4_pf_slot_t *old = &pf->slot[s];
            for (int i = 0; i < old->n; i++) { free(old->raw[i]); old->raw[i] = NULL; }
            old->n = 0;
            InterlockedExchange(&old->state, 0);
            sl = old;
            break;
        }
    }
    if (!sl) return; /* both slots busy; compute path will fall back */
    sl->layer = il;
    g4_pf_layer_weights(m, il, sl->w, &sl->n);
    if (sl->n == 0) return;
    for (int i = 0; i < sl->n; i++) sl->bytes[i] = sl->w[i]->raw_bytes;
    InterlockedExchange(&sl->state, 1);
    pf->requests++;
    SetEvent(pf->req_evt);
}

static void g4_prefetch_adopt(g4_model_t *m, int il)
{
    g4_prefetch_t *pf = (g4_prefetch_t *)m->prefetch;
    if (!pf) return;
    for (int s = 0; s < G4_PF_SLOTS; s++) {
        g4_pf_slot_t *sl = &pf->slot[s];
        if (InterlockedCompareExchange(&sl->state, 2, 2) != 2 || sl->layer != il) continue;
        for (int i = 0; i < sl->n; i++) {
            uint8_t *raw = sl->raw[i];
            sl->raw[i] = NULL;
            if (!raw) continue;
            g4_qweight_t *w = sl->w[i];
            if (w && !w->raw_cache) g4_cache_insert(w, raw, (size_t)sl->bytes[i]);
            else free(raw);
        }
        InterlockedExchange(&sl->state, 0);
    }
}

static void g4_prefetch_shutdown(g4_model_t *m)
{
    g4_prefetch_t *pf = (g4_prefetch_t *)m->prefetch;
    if (!pf) return;
    InterlockedExchange(&pf->stop, 1);
    SetEvent(pf->stop_evt);
    if (pf->thread) {
        WaitForSingleObject(pf->thread, 10000);
        CloseHandle(pf->thread);
    }
    for (int s = 0; s < G4_PF_SLOTS; s++) {
        for (int i = 0; i < pf->slot[s].n; i++) free(pf->slot[s].raw[i]);
    }
    CloseHandle(pf->req_evt);
    CloseHandle(pf->stop_evt);
    if (pf->file) fclose(pf->file);
    if (pf->requests)
        fprintf(stderr, "[G4] prefetch: %llu requests, %llu ready\n",
                (unsigned long long)pf->requests, (unsigned long long)pf->ready);
    free(pf);
    m->prefetch = NULL;
}
#else
static void g4_prefetch_init(g4_model_t *m, const char *path) { (void)m; (void)path; }
static void g4_prefetch_request(g4_model_t *m, int il) { (void)m; (void)il; }
static void g4_prefetch_adopt(g4_model_t *m, int il) { (void)m; (void)il; }
static void g4_prefetch_shutdown(g4_model_t *m) { (void)m; }
#endif

/* Read one weight from disk and dequantize it to a temporary F32 buffer.
 * Resident cost is one weight (~236 MB for FFN) instead of the whole 4 GB blob. */
static float *g4_qweight_stream_buffer(const g4_qweight_t *w, int *owned)
{
    *owned = 0;
    if (!w || !w->stream_ctx || !w->stream_info || w->raw_bytes <= 0) return NULL;
    const uint8_t *src = NULL;
    int src_owned = 0;
    if (w->raw_cache) {
        g4_cache_touch((g4_qweight_t *)w);
        g_cache_hits++;
        src = w->raw_cache;
    } else {
        g_cache_misses++;
        uint8_t *raw = malloc((size_t)w->raw_bytes);
        if (!raw) return NULL;
        if (!gguf_read_raw_tensor((gguf_ctx *)w->stream_ctx,
                                  (gguf_tensor_info *)w->stream_info, raw)) {
            free(raw); return NULL;
        }
        if (g_cache_cap && g_cache_bytes + (size_t)w->raw_bytes <= g_cache_cap) {
            g4_cache_insert((g4_qweight_t *)w, raw, (size_t)w->raw_bytes);
            src = raw;
        } else {
            src = raw;
            src_owned = 1;
        }
    }
    float *f = (float *)malloc((size_t)w->n_elems * sizeof(float));
    if (!f) { if (src_owned) free((void *)src); return NULL; }
    gguf_dequantize(src, w->ggml_type, w->n_elems, f);
    if (src_owned) free((void *)src);
    *owned = 1; /* caller owns the F32 scratch buffer */
    return f;
}

static const uint8_t *g4_qweight_stream_raw(const g4_qweight_t *w, int *owned)
{
    *owned = 0;
    if (!w || !w->stream_ctx || !w->stream_info || w->raw_bytes <= 0) return NULL;
    if (w->raw_cache) {
        g4_cache_touch((g4_qweight_t *)w);
        g_cache_hits++;
        return w->raw_cache;
    }
    g_cache_misses++;
    uint8_t *raw = (uint8_t *)malloc((size_t)w->raw_bytes);
    if (!raw) return NULL;
    if (!gguf_read_raw_tensor((gguf_ctx *)w->stream_ctx,
                              (gguf_tensor_info *)w->stream_info, raw)) {
        free(raw); return NULL;
    }
    if (g_cache_cap && g_cache_bytes + (size_t)w->raw_bytes <= g_cache_cap) {
        g4_cache_insert((g4_qweight_t *)w, raw, (size_t)w->raw_bytes);
        return raw;
    }
    *owned = 1;
    return raw;
}

static int g4_type_native_supported(int t)
{
    switch (t) {
    case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:
        return 1;
    default:
        return 0;
    }
}

static int g4_qweight_stream_f32(const g4_qweight_t *w, float **out)
{
    int owned = 0;
    float *f = g4_qweight_stream_buffer(w, &owned);
    if (!f) return 0;
    *out = f;
    return 1;
}

static struct { double ms; long calls; } G4_CPU_STATS[64];
static struct { double ms; long calls; } G4_CPU_DEC_STATS[64];  /* M == 1 only */
static int G4_CPU_STATS_hooked;

#if defined(__AVX2__)
#include <immintrin.h>
/* Dot product for the attention score and value loops. These were the only remaining
 * scalar inner loops in decode: hd=256 per (query head, cached position), so a 1024-token
 * sliding window is 16 heads x 1024 positions x 256 MACs per layer, and it ran at scalar
 * speed while everything else had been vectorised. */
static inline float g4_dotf(const float *a, const float *b, int n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 16 <= n; d += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), s1);
    }
    for (; d + 8 <= n; d += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), s0);
    __m256 s = _mm256_add_ps(s0, s1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; d < n; d++) r += a[d] * b[d];
    return r;
}
#else
static inline float g4_dotf(const float *a, const float *b, int n)
{
    float r = 0.0f;
    for (int d = 0; d < n; d++) r += a[d] * b[d];
    return r;
}
#endif

/* out[d] += w * v[d]. Called with kp ascending from the softmax loop, so each output
 * element accumulates in exactly the order the scalar version did - bit-identical, just
 * eight lanes wide. */
static inline void g4_axpyf(float w, const float *v, float *out, int n)
{
#if defined(__AVX2__)
    const __m256 wv = _mm256_set1_ps(w);
    int d = 0;
    /* mul then add, not FMA: contracting this changed greedy token 22 (37475 -> 193634).
     * Two roundings match the scalar loop exactly; one rounding does not. */
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(out + d, _mm256_add_ps(_mm256_mul_ps(wv, _mm256_loadu_ps(v + d)),
                                                _mm256_loadu_ps(out + d)));
    for (; d < n; d++) out[d] += w * v[d];
#else
    for (int d = 0; d < n; d++) out[d] += w * v[d];
#endif
}

static double g4_now_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    static LARGE_INTEGER freq;
    static int init;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    return 0.0;
#endif
}

void wubu_gpu_stage_report(void);
extern int wubu_gpu_tag;   /* 1 qkv, 2 o-proj, 3 ffn, 4 head */

static void g4_cpu_stats_dump(void)
{
    if (!getenv("WUBU_GPU_STATS")) return;
    fprintf(stderr, "[CPU stats] type  calls   ms_total   decode_calls  decode_ms\n");
    for (int t = 0; t < 64; ++t)
        if (G4_CPU_STATS[t].calls)
            fprintf(stderr, "[CPU stats] %4d %7ld %9.1f   %7ld %9.1f\n", t,
                    G4_CPU_STATS[t].calls, G4_CPU_STATS[t].ms,
                    G4_CPU_DEC_STATS[t].calls, G4_CPU_DEC_STATS[t].ms);
}

static long G4_DEC_GPU_CALLS;
static long G4_DEC_TYPE_CALLS[64];   /* every decode matvec, by ggml_type */
static long G4_DEC_TYPE_GPU[64];     /* the subset that ran on the device    */

void g4_batched_qmatmul(const float *x, const g4_qweight_t *w,
                               int M, int N, int K, float *c) {
    if (!w) return;
    if (!G4_CPU_STATS_hooked) { G4_CPU_STATS_hooked = 1; atexit(g4_cpu_stats_dump); }
    /* Every decode matvec, by type. Calls that did not reach the device are the
     * difference from G4_DEC_TYPE_GPU below, so no separate counter is needed (an
     * earlier "cpu calls" counter incremented at entry and reported 202 for a token
     * that only had 5 CPU matvecs). */
    if (M == 1 && w->ggml_type >= 0 && w->ggml_type < 64) ++G4_DEC_TYPE_CALLS[w->ggml_type];
    const double t_cpu0 = g4_now_ms();
    /* Device-resident weights: decode (M == 1) runs entirely on the GPU. */
    if (w->dev && M == 1 && (K % 256) == 0 &&
        (w->ggml_type == GGML_TYPE_IQ2_S || w->ggml_type == GGML_TYPE_IQ3_S ||
         w->ggml_type == GGML_TYPE_Q3_K || w->ggml_type == GGML_TYPE_Q4_0 ||
         w->ggml_type == GGML_TYPE_IQ3_XXS ||     /* 282 MB of the 12B file */
         w->ggml_type == GGML_TYPE_IQ4_XS ||      /* 125 MB */
         w->ggml_type == GGML_TYPE_IQ2_XS)) {     /* 11 MB, last CPU matvec */
        const long long stride = (long long)(w->raw_bytes / N);
        if (wubu_gpu_matvec(w->dev, w->ggml_type, x, K, N, stride, c) == 0) {
            if (M == 1) {
                ++G4_DEC_GPU_CALLS;
                if (w->ggml_type >= 0 && w->ggml_type < 64) ++G4_DEC_TYPE_GPU[w->ggml_type];
            }
            return;
        }
    }
    /* The M-row device prefill path (wubu_gpu_matvec_mrow) is present but NOT used yet:
     * it measured prefill 6.8 -> 15.4 tok/s with identical greedy tokens on real model
     * data, but a synthetic gate (EG-122, random legal blocks, K=512) returns garbage
     * for 3-9 of 64 columns and the cause is not yet identified. An unexplained
     * data-dependent failure is not shippable, so prefill stays on the CPU path until
     * the mrow kernel has its own passing per-column gate. */
    if (!w->data && w->stream_ctx) {
        if (g4_type_native_supported(w->ggml_type) && N > 0) {
            /* Feed quantized bytes straight to the SIMD matmul: no 4x F32
             * scratch buffer and no separate dequantization pass. */
            int owned = 0;
            const uint8_t *raw = g4_qweight_stream_raw(w, &owned);
            if (!raw) {
                memset(c, 0, (size_t)M * N * sizeof(float));
                return;
            }
            quantized_matmul_batched(x, raw, w->ggml_type, (int64_t)K, (int64_t)N,
                                     w->raw_bytes / N, M, c);
            if (owned) free((void *)raw);
            return;
        }
        int owned = 0;
        float *wf = g4_qweight_stream_buffer(w, &owned);
        if (!wf) {
            memset(c, 0, (size_t)M * N * sizeof(float));
            return;
        }
        quantized_matmul_batched(x, wf, GGML_TYPE_F32, (int64_t)K, (int64_t)N,
                                 (int64_t)K * 4, M, c);
        if (owned) free(wf);
        return;
    }
    if (!w->data || w->raw_bytes <= 0) {
        memset(c, 0, (size_t)M * N * sizeof(float));
        return;
    }
    if (w->ggml_type == GGML_TYPE_Q4_0 && (K % 256) != 0) {
        /* Q4_0 with a K that is not a Q8_K multiple: scalar per-column fallback.
         * The normal path (K % 256 == 0) uses the fused AVX2 Q4_0 x Q8_K dot in
         * quantized_matmul_batched — that is the LM head, 262144 columns.
         * Here: 18 bytes/32 elems per column, dequant to register, dot with x. */
        int row_bytes = (int)(w->raw_bytes / N);
        #pragma omp parallel for if(N > 256)
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                const uint8_t *col_data = w->data + (size_t)n * row_bytes;
                int64_t n_blocks = (K + 31) / 32;
                float sum = 0.0f;
                for (int64_t b = 0; b < n_blocks; b++) {
                    uint16_t d_bits;
                    memcpy(&d_bits, col_data + b * 18, 2);
                    uint32_t sign = (d_bits >> 15) & 1;
                    uint32_t exp  = (d_bits >> 10) & 0x1F;
                    uint32_t mant = d_bits & 0x03FF;
                    uint32_t f32_bits;
                    if (exp == 0) {
                        /* subnormal fp16: normalize (mant<<shift until bit 10),
                         * drop implicit 1, exponent 113-shift. The old code
                         * assumed bit 10 set -> wrong scale for mant < 512. */
                        if (mant == 0) f32_bits = sign << 31;
                        else {
                            int shift = __builtin_clz(mant) - 21;
                            uint32_t m = ((uint32_t)mant << shift) & 0x3FFu;
                            f32_bits = (sign << 31) | ((113u - (uint32_t)shift) << 23) | (m << 13);
                        }
                    }
                    else if (exp == 31)
                        f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
                    else
                        f32_bits = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
                    float d;
                    memcpy(&d, &f32_bits, 4);
                    const uint8_t *qs = col_data + b * 18 + 2;
                    int64_t base = b * 32;
                    for (int j = 0; j < 32 && base + j < K; j++) {
                        /* ggml Q4_0 layout: elems 0-15 = LOW nibbles of
                         * qs[0..15], elems 16-31 = HIGH nibbles. */
                        int val = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
                        sum += x[(size_t)m * K + base + j] * d * (float)(val - 8);
                    }
                }
                c[(size_t)m * N + n] = sum;
            }
        }
        return;
    }

    /* Everything else: batched API shares the weight read/dequant across
     * all M tokens (IQ2_S/IQ3_S dequantize once per batch, K-quants share
     * the weight stream; was a per-token quantized_matmul loop). */
    int64_t col_stride = w->raw_bytes / N;
    quantized_matmul_batched(x, w->data, w->ggml_type,
                             (int64_t)K, (int64_t)N, col_stride, M, c);
    if (w->ggml_type >= 0 && w->ggml_type < 64) {
        const double dt = g4_now_ms() - t_cpu0;
        G4_CPU_STATS[w->ggml_type].calls++;
        G4_CPU_STATS[w->ggml_type].ms += dt;
        if (M == 1) {
            G4_CPU_DEC_STATS[w->ggml_type].calls++;
            G4_CPU_DEC_STATS[w->ggml_type].ms += dt;
        }
    }
}

/* ============ Math ============ */

void g4_rms_norm(const float *x, const float *weight, int n, float eps, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv * weight[i];
}

float g4_gelu_tanh(float x) {
    /* llama.cpp LLM_FFN_GELU uses ggml_gelu (erf form), not tanh approx. */
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

void g4_softcap(float *logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    #pragma omp parallel for if(n > 65536)
    for (int i = 0; i < n; i++) {
        logits[i] = tanhf(logits[i] * inv_cap) * cap;
    }
}

/* ============ RoPE (dual-mode) ============ */

/* Gemma 4 uses NEOX RoPE: each head is rotated as two halves
 * (j, j + n_rot/2), NOT adjacent pairs. For full-attention layers the
 * precomputed rope_freqs.weight acts as a divisor ('factors' in llama.cpp):
 *   theta_j = pos * rope_base^(-2j/n_rot) / freq_factors[j]
 * with factors 1.0 for the active 25% rotary dims and 1e30 for the rest
 * (division by 1e30 => identity rotation). SWA layers have no factors. */
static void g4_rope_apply(float *q, float *k, int n_tokens, int d_head,
                          int q_stride, int k_stride,
                          const int *positions, int n_rot, float rope_base,
                          const float *freqs, int is_full) {
    (void)is_full;
    int half = n_rot / 2;
    float theta_scale = powf(rope_base, -2.0f / (float)n_rot);
    for (int t = 0; t < n_tokens; t++) {
        float pos = (float)positions[t];
        /* Q has n_head heads, K has n_head_kv heads. Rotating K with the Q
         * head offset wrote past each K row into the next token's K and
         * re-rotated shared K heads multiple times. Keep the loops separate. */
        for (int h = 0; h < q_stride; h += d_head) {
            float *qv = q + (size_t)t * q_stride + h;
            for (int j = 0; j < half; j++) {
                float ff = freqs ? freqs[j] : 1.0f;
                if (ff > 1e20f) continue;              /* inactive rotary pair */
                float theta = pos * powf(theta_scale, (float)j) / ff;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float q0 = qv[j], q1 = qv[j + half];
                qv[j]       = q0 * cos_t - q1 * sin_t;
                qv[j+half]  = q0 * sin_t + q1 * cos_t;
            }
        }
        if (k) {
            for (int h = 0; h < k_stride; h += d_head) {
                float *kv = k + (size_t)t * k_stride + h;
                for (int j = 0; j < half; j++) {
                    float ff = freqs ? freqs[j] : 1.0f;
                    if (ff > 1e20f) continue;
                    float theta = pos * powf(theta_scale, (float)j) / ff;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float k0 = kv[j], k1 = kv[j + half];
                    kv[j]      = k0 * cos_t - k1 * sin_t;
                    kv[j+half] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }
}

/* ============ KV cache ============ */

static void g4_kv_init(g4_kv_cache_t *cache, int kv_heads, int head_dim) {
    int init = 2048;
    cache->max = init;
    cache->size = 0;
    cache->kv_heads = kv_heads;
    cache->head_dim = head_dim;
    int elem = kv_heads * head_dim;
    cache->k = (float *)calloc((size_t)init * elem, sizeof(float));
    cache->v = (float *)calloc((size_t)init * elem, sizeof(float));
}

static void g4_kv_grow(g4_kv_cache_t *cache, int needed) {
    if (needed <= cache->max) return;
    int new_max = cache->max * 2;
    if (new_max < needed) new_max = needed + 1024;
    int elem = cache->kv_heads * cache->head_dim;
    cache->k = (float *)realloc(cache->k, (size_t)new_max * elem * sizeof(float));
    cache->v = (float *)realloc(cache->v, (size_t)new_max * elem * sizeof(float));
    memset(cache->k + (size_t)cache->max * elem, 0,
           (size_t)(new_max - cache->max) * elem * sizeof(float));
    memset(cache->v + (size_t)cache->max * elem, 0,
           (size_t)(new_max - cache->max) * elem * sizeof(float));
    cache->max = new_max;
}

static void g4_kv_store(g4_kv_cache_t *cache, int pos,
                        const float *k, const float *v) {
    g4_kv_grow(cache, pos + 1);
    int elem = cache->kv_heads * cache->head_dim;
    memcpy(cache->k + (size_t)pos * elem, k, (size_t)elem * sizeof(float));
    memcpy(cache->v + (size_t)pos * elem, v, (size_t)elem * sizeof(float));
    if (pos >= cache->size) cache->size = pos + 1;
}

/* ============ Init ============ */

bool g4_model_init(g4_model_t *model, const char *gguf_path) {
    memset(model, 0, sizeof(*model));
    model->n_layers = G4_N_LAYERS;
    model->max_ctx = G4_MAX_CTX;
    model->current_pos = 0;

    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) return false;
    if (getenv("G4_STREAM_WEIGHTS")) {
        fprintf(stderr, "[G4] weight streaming: no blob buffer; reading tensors on demand\n");
        const char *cap = getenv("G4_WEIGHT_CACHE_MB");
        g_cache_cap = cap ? (size_t)atoll(cap) * 1048576 : 0;
        if (g_cache_cap)
            fprintf(stderr, "[G4] RAM weight cache cap: %zu MiB\n", g_cache_cap / 1048576);
    } else if (!gguf_buffer_data(ctx)) { gguf_close(ctx); return false; }

    printf("[G4] Loading model (%lld tensors, %zu MB data blob)\n",
           (long long)ctx->n_tensors, ctx->data_blob_size / 1048576);

    model->layers = (g4_layer_t *)calloc(model->n_layers, sizeof(g4_layer_t));
    if (!model->layers) { gguf_close(ctx); return false; }

    /* ---- Global tensors ---- */

    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "[G4] Missing token_embd.weight\n"); goto fail; }
    qw_load(&model->token_embd, ctx, "token_embd.weight");
    printf("  token_embd: %lld x %lld (type=%d, %lld MB)\n",
           (long long)t->dims[0], (long long)t->dims[1],
           t->ggml_type, (long long)model->token_embd.raw_bytes / 1048576);

    t = gguf_find_tensor(ctx, "output.weight");
    if (t) {
        qw_load(&model->output, ctx, "output.weight");
        model->tied_output = false;
        printf("  output: separate (%lld MB)\n", (long long)model->output.raw_bytes / 1048576);
    } else {
        qw_load_aliased(&model->output, &model->token_embd);
        model->tied_output = true;
        printf("  output: tied to token_embd\n");
    }

    t = gguf_find_tensor(ctx, "output_norm.weight");
    if (!t) { fprintf(stderr, "[G4] Missing output_norm.weight\n"); goto fail; }
    model->output_norm_weight = (float *)malloc(G4_HIDDEN * sizeof(float));
    gguf_read_tensor_f32(ctx, t, model->output_norm_weight, G4_HIDDEN);

    /* ---- Per-layer ---- */
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        g4_layer_init(l, i);

        int hd = l->head_dim;

        /* Norms */
        l->attn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_norm.weight", i), G4_HIDDEN);
        if (!l->attn_norm_weight) { fprintf(stderr,"[G4] attn_norm[%d] missing\n", i); goto fail; }

        l->attn_q_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_q_norm.weight", i), hd);
        if (!l->attn_q_norm_weight) { fprintf(stderr,"[G4] attn_q_norm[%d] missing\n", i); goto fail; }

        l->attn_k_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.attn_k_norm.weight", i), hd);
        if (!l->attn_k_norm_weight) l->attn_k_norm_weight = l->attn_q_norm_weight;
        if (i == 0 && getenv("G4_DUMP_TYPES")) {
            fprintf(stderr, "[NORM q]");
            for (int z = 0; z < 8; z++) fprintf(stderr, " %.5f", l->attn_q_norm_weight[z]);
            fprintf(stderr, "\n[NORM k]");
            for (int z = 0; z < 8; z++) fprintf(stderr, " %.5f", l->attn_k_norm_weight[z]);
            fprintf(stderr, "\n[NORM out]");
            for (int z = 0; z < 8; z++) fprintf(stderr, " %.5f", model->output_norm_weight[z]);
            fprintf(stderr, "\n");
        }

        l->post_attn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.post_attention_norm.weight", i), G4_HIDDEN);
        if (!l->post_attn_norm_weight) { fprintf(stderr,"[G4] post_attn_norm[%d] missing\n", i); goto fail; }

        l->ffn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.ffn_norm.weight", i), G4_HIDDEN);
        if (!l->ffn_norm_weight) { fprintf(stderr,"[G4] ffn_norm[%d] missing\n", i); goto fail; }

        l->post_ffn_norm_weight = g4_load_norm_f32(ctx, TN("blk.%d.post_ffw_norm.weight", i), G4_HIDDEN);
        if (!l->post_ffn_norm_weight) { fprintf(stderr,"[G4] post_ffw_norm[%d] missing\n", i); goto fail; }

        /* Quantized weights */
        qw_load(&l->attn_q, ctx, TN("blk.%d.attn_q.weight", i));
        qw_load(&l->attn_out, ctx, TN("blk.%d.attn_output.weight", i));

        if (!l->share_kv) {
            qw_load(&l->attn_k, ctx, TN("blk.%d.attn_k.weight", i));
            t = gguf_find_tensor(ctx, TN("blk.%d.attn_v.weight", i));
            if (t) {
                qw_load(&l->attn_v, ctx, TN("blk.%d.attn_v.weight", i));
            } else {
                qw_load_aliased(&l->attn_v, &l->attn_k);
                l->kv_eq = true;
            }
        }

        qw_load(&l->ffn_gate, ctx, TN("blk.%d.ffn_gate.weight", i));
        qw_load(&l->ffn_up,   ctx, TN("blk.%d.ffn_up.weight", i));
        qw_load(&l->ffn_down, ctx, TN("blk.%d.ffn_down.weight", i));

        t = gguf_find_tensor(ctx, TN("blk.%d.layer_output_scale.weight", i));
        if (t) {
            gguf_read_tensor_f32(ctx, t, &l->layer_out_scale, 1);
            l->has_out_scale = true;
        }

        if (l->is_full) {
            t = gguf_find_tensor(ctx, TN("blk.%d.rope_freqs.weight", i));
            if (!t) t = gguf_find_tensor(ctx, "rope_freqs.weight"); /* GGUF stores one global tensor */
            if (t) {
                l->rope_freqs = (float *)malloc((size_t)(hd / 2) * sizeof(float));
                gguf_read_tensor_f32(ctx, t, l->rope_freqs, hd / 2);
                l->has_rope_freqs = true;
            }
        }
    }

    model->data_blob = ctx->data_blob;
    model->data_blob_size = ctx->data_blob_size;
    model->data_blob_is_mmap = ctx->data_blob_is_mmap;
    model->data_blob_mmap_base = ctx->data_blob_mmap_base;
    model->data_blob_mmap_len = ctx->data_blob_mmap_len;
    ctx->data_blob = NULL;

    /* Save GGUF file handle and tensor info for GPU weight streaming.
     * We keep the file open so we can read weights directly from disk
     * during inference, bypassing the mmap'd data blob which causes
     * cudaMemcpy failures on WSL2 when file > physical RAM. */
    model->gguf_file = ctx->file;
    model->gguf_data_offset = ctx->data_blob_offset;
    model->gguf_n_tensors = (int)ctx->n_tensors;
    model->gguf_tensor_info = ctx->tensors;
    strncpy(model->gguf_path, gguf_path, sizeof(model->gguf_path) - 1);
    g4_apply_residency_plan(model);
    g4_apply_gpu_plan(model);
    g4_prefetch_init(model, model->gguf_path);
    if (getenv("G4_CACHE_LM_HEAD")) {
        int64_t head_elems = (int64_t)G4_VOCAB * G4_HIDDEN;
        model->lm_head_f32 = (float *)malloc((size_t)head_elems * sizeof(float));
        if (model->lm_head_f32) {
            if (model->output.data) {
                gguf_dequantize(model->output.data, model->output.ggml_type,
                                head_elems, model->lm_head_f32);
            } else {
                free(model->lm_head_f32);
                model->lm_head_f32 = NULL;
                if (g4_qweight_stream_f32(&model->output, &model->lm_head_f32)) {
                    fprintf(stderr, "[G4] LM head streamed from file\n");
                }
            }
        }
        if (model->lm_head_f32) {
            fprintf(stderr, "[G4] LM head cached: %.1f MiB F32\n",
                    (double)(head_elems * sizeof(float)) / 1048576.0);
        } else {
            fprintf(stderr, "[G4] LM head cache unavailable; using streaming path\n");
        }
    }

    if (getenv("G4_DUMP_TYPES")) {
        int hist[64] = {0};
        for (int64_t ti = 0; ti < ctx->n_tensors; ti++)
            hist[ctx->tensors[ti].ggml_type]++;
        for (int ty = 0; ty < 64; ty++)
            if (hist[ty]) fprintf(stderr, "[TYPE] %d = %d tensors\n", ty, hist[ty]);
        for (int64_t ti = 0; ti < ctx->n_tensors; ti++) {
            const char *nm = ctx->tensors[ti].name;
            if (strstr(nm, "blk.") != NULL) continue;
            fprintf(stderr, "[TENSOR] %s type=%d dim0=%lld dim1=%lld off=%llu\n",
                    nm, ctx->tensors[ti].ggml_type,
                    (long long)ctx->tensors[ti].dims[0],
                    ctx->tensors[ti].n_dims > 1 ? (long long)ctx->tensors[ti].dims[1] : -1,
                    (unsigned long long)ctx->tensors[ti].data_offset);
        }
    }
    if (getenv("G4_STREAM_WEIGHTS")) {
        model->stream_ctx = ctx; /* qweights hold pointers into this ctx */
    } else {
    ctx->file = NULL;       /* model now owns the file handle */
    ctx->tensors = NULL;    /* model now owns the tensor info */
    ctx->data_blob = NULL;  /* already transferred above */

    /* Close the gguf context but NOT the file (model owns it now) */
    free(ctx);
    }

    printf("[G4] %d layers loaded (~%d MB quantized + %.1f MB norms)\n",
           model->n_layers, (int)(model->data_blob_size / 1048576),
           (double)(model->n_layers * 6 * G4_HIDDEN * 4) / 1048576.0);

    model->kv_cache = (g4_kv_cache_t *)calloc(model->n_layers, sizeof(g4_kv_cache_t));
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        g4_kv_init(&model->kv_cache[i], l->kv_heads, l->head_dim);
    }

    return true;

fail:
    gguf_close(ctx);
    g4_model_destroy(model);
    return false;
}

/* ============ KV-cache helpers ============ */

static g4_kv_cache_t *g4_kv_get(g4_model_t *model, g4_layer_t *layer) {
    /* Only true KV-sharing layers read a different layer's cache. This GGUF
     * reports shared_kv_layers = 0, so last-8 layers must read their own. */
    return &model->kv_cache[layer->share_kv ? layer->kv_src_idx : layer->layer_idx];
}

/* ============ Sliding window attention (OpenMP-parallel) ============ */

static void g4_sliding_attn_cpu(g4_model_t *model, g4_layer_t *layer, g4_kv_cache_t *cache,
                                const float *q, float *attn_out,
                                int N, const int *positions,
                                float *scores_buf, float *row_buf) {
    (void)row_buf;
    (void)model;
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = cache->kv_heads * cache->head_dim;
    int n_heads = G4_HEADS;
    int kv_heads = cache->kv_heads;
    int window = G4_SLIDING_WINDOW;

    memset(attn_out, 0, (size_t)N * qd * sizeof(float));

    /* Flattened over (token, head): during decode N == 1, so paralleling over t left the
     * whole attention block on one core (8.4 ms/token for ~60k MACs per layer). Each
     * (t, hq) writes a disjoint slice of scores_buf and of attn_out. */
    #pragma omp parallel for if(N * n_heads > 1)
    for (int th = 0; th < N * n_heads; th++) {
        const int t = th / n_heads;
        const int hq = th % n_heads;
        int pos = positions[t];
        int kv_start = (pos >= window) ? (pos - window + 1) : 0; /* exactly `window` keys incl. current */
        int kv_len = pos - kv_start + 1;
        if (kv_len > cache->size) kv_len = cache->size;
        if (kv_len <= 0) continue;
        int kv_end = kv_start + kv_len - 1;

        {
            int kv_h = hq / (n_heads / kv_heads); /* GQA: consecutive Q heads share a KV head */
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;

            /* Scores in thread-local part of scores_buf */
            float *s = scores_buf + (size_t)th * window;

            float max_s = -FLT_MAX;
            for (int kp = kv_start; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                const float score = g4_dotf(q_vec, kv, hd);
                /* Gemma 4 uses kq_scale = 1.0 (no 1/sqrt(head_dim)); llama.cpp
                 * gemma4.cpp sets f_attention_scale = 1.0f. */
                s[kp - kv_start] = score;
                if (score > max_s) max_s = score;
            }

            float sum_e = 0.0f;
            for (int kp = kv_start; kp <= kv_end; kp++) {
                float e = expf(s[kp - kv_start] - max_s);
                s[kp - kv_start] = e;
                sum_e += e;
            }
            float inv_sum = 1.0f / (sum_e + 1e-10f);

            {
                float *out = attn_out + (size_t)t * qd + (size_t)hq * hd;
                memset(out, 0, (size_t)hd * sizeof(float));
                for (int kp = kv_start; kp <= kv_end; kp++)
                    g4_axpyf(s[kp - kv_start] * inv_sum,
                             cache->v + (size_t)kp * kvd + (size_t)kv_h * hd, out, hd);
            }
        }
    }
}

/* ============ Full attention (OpenMP-parallel) ============ */

static void g4_full_attn_cpu(g4_model_t *model, g4_layer_t *layer, g4_kv_cache_t *cache,
                             const float *q, float *attn_out,
                             int N, const int *positions,
                             float *scores_buf, float *row_buf) {
    (void)row_buf;
    (void)model;
    (void)scores_buf;
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = cache->kv_heads * cache->head_dim;
    int n_heads = G4_HEADS;
    int kv_heads = cache->kv_heads;
    int kv_size = cache->size;
    /* Caller's scores buffer is sized for SWA windows; full attention needs
     * one score per cached position. Allocate locally to avoid heap overflow. */
    float *scores_local = (float *)malloc((size_t)N * n_heads * (kv_size > 0 ? kv_size : 1) * sizeof(float));
    if (!scores_local) return;

    memset(attn_out, 0, (size_t)N * qd * sizeof(float));

    /* Flattened over (token, head) for the same reason as sliding attention above. */
    #pragma omp parallel for if(N * n_heads > 1)
    for (int th = 0; th < N * n_heads; th++) {
        const int t = th / n_heads;
        const int hq = th % n_heads;
        int pos = positions[t];
        int kv_end = kv_size - 1;
        if (kv_end > pos) kv_end = pos;
        int kv_len = kv_end + 1;
        if (kv_len <= 0) continue;

        {
            int kv_h = hq / (n_heads / kv_heads); /* GQA: consecutive Q heads share a KV head */
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;
            float *s = scores_local + (size_t)th * kv_size;

            float max_s = -FLT_MAX;
            for (int kp = 0; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                const float score = g4_dotf(q_vec, kv, hd);
                /* Gemma 4: kq_scale = 1.0f (see sliding-attention above). */
                s[kp] = score;
                if (score > max_s) max_s = score;
            }

            float sum_e = 0.0f;
            for (int kp = 0; kp <= kv_end; kp++) {
                float e = expf(s[kp] - max_s);
                s[kp] = e;
                sum_e += e;
            }
            float inv_sum = 1.0f / (sum_e + 1e-10f);

            /* Left in the original d-outer form on purpose: rewriting it kp-outer with
             * the same value loop changed greedy token 22 (37475 -> 193634) even though
             * the arithmetic is, on paper, identical. Not worth the token-identity bar
             * for the cost of the few non-sliding layers. */
            for (int d = 0; d < hd; d++) {
                float val = 0.0f;
                for (int kp = 0; kp <= kv_end; kp++) {
                    val += s[kp] * inv_sum *
                           (cache->v + (size_t)kp * kvd + (size_t)kv_h * hd)[d];
                }
                attn_out[(size_t)t * qd + (size_t)hq * hd + d] = val;
            }
        }
    }
    free(scores_local);
}

/* ============ Single layer forward (CPU, fused matmuls in one parallel region) ============ */

/* ---- section profiler (G4_PROF=1) -------------------------------------
 * The decode path spends ~97 ms/token while GPU matvec is only ~23 ms of it, so the
 * interesting number is what the CPU does between the matvecs. Accumulated over every
 * N == 1 forward and reported per token, because that is the case being optimised. */
enum { G4P_LAYER, G4P_NORM_QKV, G4P_QKNORM_ROPE, G4P_KVSTORE, G4P_ATTN,
       G4P_ATTNOUT, G4P_FFN, G4P_HEAD, G4P_FFN_PROJ, G4P_FFN_GELU, G4P_FFN_NORM,
       G4P_OPROJ, G4P_ATTN_NORM, G4P_COUNT };
static double G4_P[G4P_COUNT];
static long G4_P_TOKENS;
static int G4_P_hooked;

static void g4_prof_dump(void)
{
    if (!getenv("G4_PROF") || G4_P_TOKENS <= 0) return;
    static const char *names[G4P_COUNT] = {
        "layer total", "norm+qkv", "qk norm+rope", "kv store", "attention",
        "attn out+resid", "ffn all", "head", "  ffn proj", "  ffn gelu+mul", "  ffn norm+resid",
        "  o proj", "  attn norm+resid"
    };
    fprintf(stderr, "[PROF] per decode token over %ld tokens (ms):\n", G4_P_TOKENS);
    for (int i = 0; i < G4P_COUNT; ++i)
        fprintf(stderr, "[PROF] %-14s %8.2f\n", names[i], G4_P[i] / (double)G4_P_TOKENS);
    {
        long total = 0;
        for (int t = 0; t < 64; ++t) total += G4_DEC_TYPE_CALLS[t];
        fprintf(stderr, "[PROF] decode matvecs/token: %.1f total, %.1f on GPU, %.1f on CPU\n",
                (double)total / (double)G4_P_TOKENS,
                (double)G4_DEC_GPU_CALLS / (double)G4_P_TOKENS,
                (double)(total - G4_DEC_GPU_CALLS) / (double)G4_P_TOKENS);
    }
    for (int t = 0; t < 64; ++t)
        if (G4_DEC_TYPE_CALLS[t])
            fprintf(stderr, "[PROF]   type %2d: %.1f calls/token, %.1f on GPU\n", t,
                    (double)G4_DEC_TYPE_CALLS[t] / (double)G4_P_TOKENS,
                    (double)G4_DEC_TYPE_GPU[t] / (double)G4_P_TOKENS);
    wubu_gpu_stage_report();
}

void g4_layer_forward_cpu(g4_model_t *model, int il, float *x,
                                 float *normed, float *attn_out,
                                 float *q, float *k, float *v,
                                 float *ffn_gate, float *ffn_up, float *ffn_down_out,
                                 float *scores, float *row, int N, const int *positions) {
    if (!G4_P_hooked) { G4_P_hooked = 1; atexit(g4_prof_dump); }
    const double _pt0 = g4_now_ms();
    double _pk = _pt0;
    #define G4P_ADD(slot, from) do { if (N == 1) { const double _n = g4_now_ms(); G4_P[slot] += _n - (from); } } while (0)
    g4_layer_t *layer = &model->layers[il];
    int hd = layer->head_dim;
    int qd = layer->q_dim;
    int kvd = layer->kv_dim;
    int kvh = layer->kv_heads;
    (void)row;

    /* ===== Pre-attention norm ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, layer->attn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* ===== Device attention half-layer (decode) =====
     * q,k,v -> head norms -> RoPE -> KV append -> attention -> o-projection in one
     * submission, gated by EG-124 against a CPU sequence of the same steps. k and v come
     * back so the CPU cache stays current and a fallback past capacity stays correct. */
    int half_done = 0;
    {
        static int half_enabled = -1;
        static int half_mode = 0;
        if (half_enabled < 0) {
            /* Opt-in, and the reason is quality, not bookkeeping. It is the faster path -
             * 14.1-14.3 tok/s against 12.8-12.9 at 65 tokens of context and 13.5 against
             * 12.0 at 257 - and its arithmetic matches the CPU sequence closely (q/k/v/attn
             * to 4e-07, o-projection to 1.5e-05, EG-124). But the continuation it produces
             * on this model is not a paraphrase of the default one: measured over 40 tokens
             * it collapses into a loop of control tokens (237542, 236779, 236756 repeating)
             * and decodes to "_", where the CPU-attention path reaches
             * "This is a classic philosophical question". Same weights, same prompt, a
             * 1.5e-05 deviation per layer compounding into a degenerate loop - so this stays
             * opt-in for measurement, not default. */
            /* Default: full-attention layers only. Measured token-identical to the CPU path
             * over 64 tokens (the whole sequence, not just the first few), because on those
             * layers the composite reproduces q, k, v, the KV rows and the span exactly.
             * G4_ATTN_DEV=all also fuses the sliding layers, which is another 8.5% but still
             * parts from the reference around generated token 2 - that is the open bug.
             * G4_ATTN_DEV=slide fuses only the sliding layers, G4_ATTN_DEV=0 disables. */
            const char *e = getenv("G4_ATTN_DEV");
            half_enabled = !(e != NULL && *e == '0');
            if (e && strcmp(e, "all") == 0) half_mode = 0;
            else if (e && strcmp(e, "slide") == 0) half_mode = 2;
            else half_mode = 1;
        }
        const int mode_ok = (half_mode == 0) || (half_mode == 1 && layer->is_full) ||
                            (half_mode == 2 && !layer->is_full);
        static int warm = -1;
        if (warm < 0) { const char *w = getenv("G4_ATTN_WARM"); warm = w ? atoi(w) : 0; }
        const int warm_ok = positions[0] >= warm;
        if (half_enabled && mode_ok && warm_ok && N == 1 && !layer->share_kv && layer->attn_q.dev &&
            layer->attn_k.dev && layer->attn_v.dev && layer->attn_out.dev) {
            g4_kv_cache_t *cache_h = g4_kv_get(model, layer);
            const int pos_h = positions[0];
            int kv_start_h = 0, kv_len_h = 0;
            /* The span must be derived from the position, not from cache_h->size: size is
             * how many rows have been allocated so far, which at position p is p - so
             * clamping with it dropped the current token from the window (measured: at
             * position 1 the composite attended 1 row where the CPU sequence attends 2).
             * That is a different attention than the reference, which is why k, v and the
             * KV rows all matched while the attention output did not. */
            { static int shown2;
              if (!shown2 && pos_h == 21 && getenv("G4_ATTN_WARM")) { shown2 = 1;
                  fprintf(stderr, "[SPAN] layer %d full=%d size=%d pos=%d kv_len=%d kv_start=%d\n",
                          il, (int)layer->is_full, cache_h->size, pos_h, kv_len_h, kv_start_h); } }
            if (layer->is_full) {
                kv_len_h = pos_h + 1;
            } else {
                kv_start_h = (pos_h >= G4_SLIDING_WINDOW) ? pos_h - G4_SLIDING_WINDOW + 1 : 0;
                kv_len_h = pos_h - kv_start_h + 1;
            }
            static int primed[128];
            /* Compare the composite against the CPU sequence for layer 0 at position 0,
             * where the device KV is trivially consistent: run the composite into a scratch
             * buffer, let the normal CPU path run, then diff the two o-projection results.
             * Without this the difference is only visible as a wrong token, which says
             * nothing about which stage is responsible. */

            /* The device KV cache is only in sync once this layer has actually appended
             * through the composite. Marking it primed at position 0 was wrong: at position
             * 0 kv_len is 0 (the cache is not sized yet), the composite is skipped, row 0 is
             * written only to the CPU cache - and every later token then attended over an
             * uninitialised device row 0. That is a wrong continuation from the second
             * token, which is what the fused path was doing. */
            if (!primed[il] &&
                wubu_gpu_attn_half_prime(il, cache_h->k, cache_h->v, pos_h, kvd) == 0)
                primed[il] = 1;
            if (primed[il] && kv_len_h > 0) {
                const long long sq_h = (long long)(layer->attn_q.raw_bytes / qd);
                const long long sk_h = (long long)(layer->attn_k.raw_bytes / kvd);
                const long long sv_h = (long long)(layer->attn_v.raw_bytes / kvd);
                const long long so_h = (long long)(layer->attn_out.raw_bytes / G4_HIDDEN);
                /* the composite writes its o-projection into the same buffer it reads, so
                 * the input has to be copied out before the call for any comparison */
                static float *qdump_x;
                const int qdump_now = getenv("G4_QDUMP") && il == 0 && pos_h == 23;
                if (qdump_now) {
                    if (!qdump_x) qdump_x = (float *)malloc((size_t)G4_HIDDEN * sizeof(float));
                    if (qdump_x) memcpy(qdump_x, normed, (size_t)G4_HIDDEN * sizeof(float));
                }
                const int rc_h = wubu_gpu_attn_half(il, layer->attn_q.dev, layer->attn_k.dev,
                                       layer->attn_v.dev, layer->attn_out.dev,
                                       layer->attn_q.ggml_type, layer->attn_k.ggml_type,
                                       layer->attn_v.ggml_type, layer->attn_out.ggml_type,
                                       normed, G4_HIDDEN, qd, kvd, hd, G4_HEADS, kvh,
                                       pos_h, kv_start_h, kv_len_h,
                                       sq_h, sk_h, sv_h, so_h,
                                       layer->attn_q_norm_weight, layer->attn_k_norm_weight,
                                       layer->rope_freqs, layer->n_rot, layer->rope_base,
                                       normed, k, v);
                if (rc_h == 0) {
                    g4_kv_store(cache_h, pos_h, k, v);
                    /* G4_QDUMP: compare the composite's q against the same matvec + q-norm +
                     * rope done on the CPU from the same input. k, v and the KV rows already
                     * match exactly, so q is the last stage that can differ. */
                    if (qdump_now && qdump_x) {
                        static float *qref, *qdev;
                        if (!qref) {
                            qref = (float *)malloc((size_t)G4_MAX_Q_DIM * sizeof(float));
                            qdev = (float *)malloc((size_t)G4_MAX_Q_DIM * sizeof(float));
                        }
                        memcpy(qref, qdump_x, (size_t)G4_HIDDEN * sizeof(float));
                        g4_batched_qmatmul(qref, &layer->attn_q, 1, qd, G4_HIDDEN, qref);
                        for (int hh = 0; hh < G4_HEADS; ++hh)
                            g4_rms_norm(qref + (size_t)hh * hd, layer->attn_q_norm_weight, hd,
                                        G4_RMS_EPS, qref + (size_t)hh * hd);
                        g4_rope_apply(qref, NULL, 1, hd, qd, 0, positions,
                                      layer->n_rot, layer->rope_base, layer->rope_freqs, layer->is_full);
                        { extern int wubu_gpu_get_qdump(float *, int);
                          wubu_gpu_get_qdump(qdev, qd); }
                        double wq4 = 0.0; int wi4 = -1;
                        for (int i = 0; i < qd; ++i) {
                            const double d = fabs((double)qdev[i] - (double)qref[i]);
                            if (d > wq4) { wq4 = d; wi4 = i; }
                        }
                        fprintf(stderr, "[QDUMP] layer0 pos21 q dev-vs-cpu=%.3e at %d: dev %.6f cpu %.6f\n",
                                wq4, wi4, wi4 >= 0 ? qdev[wi4] : 0.0f, wi4 >= 0 ? qref[wi4] : 0.0f);
                        for (int hh = 0; hh < G4_HEADS; ++hh) {
                            double wh = 0.0;
                            for (int d = 0; d < hd; ++d) {
                                const double dd = fabs((double)qdev[hh * hd + d] - (double)qref[hh * hd + d]);
                                if (dd > wh) wh = dd;
                            }
                            if (wh > 1e-2) fprintf(stderr, "[QDUMP] head %d worst %.3e\n", hh, wh);
                        }
                    }
                    primed[il] = 1;
                    half_done = 1;
                }
            }
        }
    }

    if (!half_done) {
    /* ===== Q, K, V projections =====
     * All three read the same normalised input, so stage it on the device once and run
     * the three kernels against the staged Q8_K row (decode only). */
    int qkv_done = 0;
    /* Eligibility first: a partial attempt would run the projections twice, which is
     * exactly what the first version of this did (qkv 22.4 -> 28.4 ms). */
    /* The 12B model mixes quant types across the three projections (attn_q IQ2_S,
     * attn_k IQ3_S, ...), so requiring one type for all three left most layers on three
     * separate round trips - 2.5 device calls per layer, measured. wubu_gpu_qkv3 takes a
     * type per projection: one upload, one Q8_K quantisation, three kernels, one call. */
    const int qkv_all_dev = layer->attn_q.dev && layer->attn_k.dev &&
                            !layer->share_kv && !layer->kv_eq && layer->attn_v.dev;
    const int qkv_kv_eq = layer->attn_q.dev && layer->attn_k.dev && layer->kv_eq;
    if (N == 1 && (G4_HIDDEN % 256) == 0 && (qkv_all_dev || qkv_kv_eq) && !layer->share_kv) {
        const int tq = layer->attn_q.ggml_type;
        const int tk = layer->attn_k.ggml_type;
        const int tv = layer->attn_v.ggml_type;
        const long long sq = (long long)(layer->attn_q.raw_bytes / qd);
        const long long sk = (long long)(layer->attn_k.raw_bytes / kvd);
        const long long sv = (long long)(layer->attn_v.raw_bytes / kvd);
        if (layer->kv_eq) {
            /* No separate V projection: q and k only, k copied into v on the host. */
            if (wubu_gpu_stage_input(normed, G4_HIDDEN) == 0 &&
                wubu_gpu_matvec_cached(layer->attn_q.dev, tq, G4_HIDDEN, qd, sq, q) == 0 &&
                wubu_gpu_matvec_cached(layer->attn_k.dev, tk, G4_HIDDEN, kvd, sk, k) == 0) {
                memcpy(v, k, (size_t)N * kvd * sizeof(float));
                qkv_done = 1;
            }
        } else if (wubu_gpu_qkv3(layer->attn_q.dev, layer->attn_k.dev, layer->attn_v.dev,
                                 tq, tk, tv, normed, G4_HIDDEN, qd, kvd,
                                 sq, sk, sv, q, k, v) == 0) {
            qkv_done = 1;
        }
    }
    if (!qkv_done) {
        wubu_gpu_tag = 1;
        g4_batched_qmatmul(normed, &layer->attn_q, N, qd, G4_HIDDEN, q);
        if (!layer->share_kv) {
            g4_batched_qmatmul(normed, &layer->attn_k, N, kvd, G4_HIDDEN, k);
            if (layer->kv_eq) {
                memcpy(v, k, (size_t)N * kvd * sizeof(float));
            } else {
                g4_batched_qmatmul(normed, &layer->attn_v, N, kvd, G4_HIDDEN, v);
            }
        }
    }

    G4P_ADD(G4P_NORM_QKV, _pk); _pk = g4_now_ms();

    /* ===== Q/K head norms =====
     * Kept single-threaded for decode on purpose: flattening this over (token, head)
     * measured 3.5 -> 9.5 ms/token, because 2 OpenMP fork/join pairs per layer
     * (~60 us each, 96 per token) cost more than the 16 x 128-element norms they
     * distribute. Parallelism here only pays for prefill (N > 1). */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < G4_HEADS; h++)
            g4_rms_norm(q + (size_t)t * qd + (size_t)h * hd,
                        layer->attn_q_norm_weight, hd, G4_RMS_EPS,
                        q + (size_t)t * qd + (size_t)h * hd);
        if (!layer->share_kv) {
            /* Gemma 4 RMS-normalizes V with NO learned weight (llama.cpp
             * gemma4.cpp: Vcur = ggml_rms_norm(ctx0, Vcur, eps)). Gemma 3
             * does not. Do not remove this for the Gemma 4 engine. */
            for (int h = 0; h < kvh; h++) {
                g4_rms_norm(k + (size_t)t * kvd + (size_t)h * hd,
                            layer->attn_k_norm_weight, hd, G4_RMS_EPS,
                            k + (size_t)t * kvd + (size_t)h * hd);
                float ss = 0.0f;
                for (int d = 0; d < hd; d++)
                    ss += v[(size_t)t * kvd + (size_t)h * hd + d]
                        * v[(size_t)t * kvd + (size_t)h * hd + d];
                const float inv_rms = 1.0f / sqrtf(ss / (float)hd + G4_RMS_EPS);
                for (int d = 0; d < hd; d++)
                    v[(size_t)t * kvd + (size_t)h * hd + d] *= inv_rms;
            }
        }
    }

    /* ===== RoPE ===== */
    if (!layer->share_kv) {
        g4_rope_apply(q, k, N, hd, qd, kvd, positions,
                      layer->n_rot, layer->rope_base, layer->rope_freqs, layer->is_full);
    } else {
        g4_rope_apply(q, NULL, N, hd, qd, 0, positions,
                      layer->n_rot, layer->rope_base, layer->rope_freqs, layer->is_full);
    }

    /* ===== Store K,V to KV cache ===== */
    if (!layer->share_kv) {
        for (int t = 0; t < N; t++) {
            int pos = positions[t];
            g4_kv_store(&model->kv_cache[il], pos,
                        k + (size_t)t * kvd, v + (size_t)t * kvd);
        }
    }

    G4P_ADD(G4P_QKNORM_ROPE, _pk); _pk = g4_now_ms();

    g4_kv_cache_t *cache = g4_kv_get(model, layer);

    /* ===== Attention ===== */
    if (layer->is_full) {
        g4_full_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    } else {
        g4_sliding_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    }

    G4P_ADD(G4P_ATTN, _pk); _pk = g4_now_ms();

    /* Sensitivity probe: how much relative noise in the attention output does this model
     * tolerate before the greedy continuation changes? The fused device path differs from
     * this sequence by ~5e-06 relative per layer, so this says whether that magnitude alone
     * is enough to explain its degenerate output. */
    {
        static double jit = -1.0;
        static int jit_seeded;
        if (jit < 0.0) {
            const char *e = getenv("G4_ATTN_JITTER");
            jit = e ? atof(e) : 0.0;
        }
        if (jit > 0.0) {
            unsigned int s = jit_seeded ? 12345u : 12345u;
            jit_seeded = 1;
            const int n = N * qd;
            for (int i = 0; i < n; ++i) {
                s = s * 1103515245u + 12345u;
                const float u = ((float)((s >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
                attn_out[i] *= (1.0f + (float)jit * u);
            }
        }
    }

    /* ===== Attention output projection ===== */
    const double _po0 = g4_now_ms();
    wubu_gpu_tag = 2;
    g4_batched_qmatmul(attn_out, &layer->attn_out, N, G4_HIDDEN, qd, normed);
    wubu_gpu_tag = 0;
    if (N == 1) { const double _t = g4_now_ms(); G4_P[G4P_OPROJ] += _t - _po0; _pk = _t; }
    }   /* end: !half_done */

    /* ===== Post-attention norm + residual ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(normed + (size_t)t * G4_HIDDEN, layer->post_attn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += normed[(size_t)t * G4_HIDDEN + d];
    }

    if (N == 1) G4_P[G4P_ATTN_NORM] += g4_now_ms() - _pk;
    G4P_ADD(G4P_ATTNOUT, _pk); _pk = g4_now_ms();

    /* ===== FFN ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, layer->ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* Quantize input to Q8_K once, reuse across gate/up/down */
    const double _pf0 = g4_now_ms();
    /* Decode: if all three FFN tensors are device-resident and share a type, run the
     * whole FFN on the device (one upload + one download instead of three round trips,
     * and the tanh GELU off the CPU). Falls through to the per-tensor path otherwise. */
    int ffn_dev_done = 0;
    /* A type per tensor, not one for all three: requiring a shared type left about a third
     * of the layers on three separate round trips. */
    if (N == 1 && layer->ffn_gate.dev && layer->ffn_up.dev && layer->ffn_down.dev &&
        (G4_HIDDEN % 256) == 0 && (G4_FFN % 256) == 0) {
        const long long sg = (long long)(layer->ffn_gate.raw_bytes / G4_FFN);
        const long long su = (long long)(layer->ffn_up.raw_bytes / G4_FFN);
        const long long sd = (long long)(layer->ffn_down.raw_bytes / G4_HIDDEN);
        ffn_dev_done = (wubu_gpu_ffn3(normed, layer->ffn_gate.dev, layer->ffn_up.dev,
                                      layer->ffn_down.dev,
                                      layer->ffn_gate.ggml_type, layer->ffn_up.ggml_type,
                                      layer->ffn_down.ggml_type,
                                      G4_HIDDEN, G4_FFN, sg, su, sd, ffn_down_out) == 0);
    }
    if (!ffn_dev_done) {
        wubu_gpu_tag = 3;
        g4_batched_qmatmul(normed, &layer->ffn_gate, N, G4_FFN, G4_HIDDEN, ffn_gate);
        g4_batched_qmatmul(normed, &layer->ffn_up,   N, G4_FFN, G4_HIDDEN, ffn_up);
    }
    if (N == 1) { const double _t = g4_now_ms(); G4_P[G4P_FFN_PROJ] += _t - _pf0; _pk = _t; }

    /* GELU + element-wise multiply (OpenMP parallel) */
    if (!ffn_dev_done) {
        int ffn_elems = N * G4_FFN;
        #pragma omp parallel for if(ffn_elems > 4096)
        for (int i = 0; i < ffn_elems; i++)
            ffn_gate[i] = g4_gelu_tanh(ffn_gate[i]);
        #pragma omp parallel for if(ffn_elems > 4096)
        for (int i = 0; i < ffn_elems; i++)
            ffn_up[i] *= ffn_gate[i];
    }

    if (N == 1) { const double _t = g4_now_ms(); G4_P[G4P_FFN_GELU] += _t - _pk; _pk = _t; }
    if (!ffn_dev_done)
        g4_batched_qmatmul(ffn_up, &layer->ffn_down, N, G4_HIDDEN, G4_FFN, ffn_down_out);
        wubu_gpu_tag = 0;
    if (N == 1) { const double _t = g4_now_ms(); G4_P[G4P_FFN_PROJ] += _t - _pk; _pk = _t; }

    /* Post-FFN norm + residual */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(ffn_down_out + (size_t)t * G4_HIDDEN, layer->post_ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, ffn_down_out + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += ffn_down_out[(size_t)t * G4_HIDDEN + d];
    }

    if (N == 1) { G4_P[G4P_FFN_NORM] += g4_now_ms() - _pk; G4_P[G4P_FFN] += g4_now_ms() - _pf0; }
    if (N == 1) G4_P[G4P_LAYER] += g4_now_ms() - _pt0;
    #undef G4P_ADD

    if (layer->has_out_scale) {
        float s = layer->layer_out_scale;
        #pragma omp parallel for if(N * G4_HIDDEN > 4096)
        for (int i = 0; i < N * G4_HIDDEN; i++) x[i] *= s;
    }
}

/* ============ Forward pass ============ */

extern int omp_get_max_threads(void);

void g4_model_forward(g4_model_t *model, const float *embeddings, int B, int T, float *logits) {
    const int N = B * T;

    const int max_q_dim = G4_MAX_Q_DIM;
    const int max_kv_dim = G4_MAX_KV_DIM;
    const int max_dim2 = (G4_FFN > G4_HIDDEN ? G4_FFN : G4_HIDDEN) > max_q_dim
                         ? (G4_FFN > G4_HIDDEN ? G4_FFN : G4_HIDDEN)
                         : max_q_dim;

    /* Grow buffers */
    if (N > model->buf_size) {
        model->buf_size = N + 1024;
        int bs = model->buf_size;
        model->buf_q = (float *)realloc(model->buf_q, (size_t)bs * max_q_dim * sizeof(float));
        model->buf_k = (float *)realloc(model->buf_k, (size_t)bs * max_kv_dim * sizeof(float));
        model->buf_v = (float *)realloc(model->buf_v, (size_t)bs * max_kv_dim * sizeof(float));
        model->buf_attn_out = (float *)realloc(model->buf_attn_out, (size_t)bs * max_q_dim * sizeof(float));
        model->buf_normed = (float *)realloc(model->buf_normed, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_embd = (float *)realloc(model->buf_embd, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_ffn_gate = (float *)realloc(model->buf_ffn_gate, (size_t)bs * G4_FFN * sizeof(float));
        model->buf_ffn_up = (float *)realloc(model->buf_ffn_up, (size_t)bs * G4_FFN * sizeof(float));
        model->buf_ffn_out = (float *)realloc(model->buf_ffn_out, (size_t)bs * G4_HIDDEN * sizeof(float));
        model->buf_scores = (float *)realloc(model->buf_scores,
                                             (size_t)bs * G4_HEADS * G4_SLIDING_WINDOW * sizeof(float));
        model->buf_row = (float *)realloc(model->buf_row, (size_t)max_dim2 * sizeof(float));
    }

    float *x = model->buf_embd;
    float *normed = model->buf_normed;
    float *attn_out = model->buf_attn_out;
    float *q = model->buf_q;
    float *k = model->buf_k;
    float *v = model->buf_v;
    float *ffn_gate = model->buf_ffn_gate;
    float *ffn_up = model->buf_ffn_up;
    float *ffn_down_out = model->buf_ffn_out;
    float *row = model->buf_row;
    float *scores = model->buf_scores;

    memcpy(x, embeddings, (size_t)N * G4_HIDDEN * sizeof(float));

    int *positions = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) positions[i] = model->current_pos + i;

    for (int il = 0; il < model->n_layers; il++) {
        /* Queue more work G4_PF_SLOTS layers ahead; the worker started at
         * model load, so most first-pass weights are already staged. */
        if (il + G4_PF_SLOTS < model->n_layers)
            g4_prefetch_request(model, il + G4_PF_SLOTS);
        g4_prefetch_adopt(model, il);
        g4_layer_forward_cpu(model, il, x, normed, attn_out, q, k, v,
                            ffn_gate, ffn_up, ffn_down_out, scores, row, N, positions);
    }

    /* ===== Final norm ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, model->output_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    if (N == 1) { ++G4_P_TOKENS; G4_P[G4P_HEAD] -= g4_now_ms(); }

    /* ===== LM head (Q3_K tied embedding, or cached F32 matvec) ===== */
    if (model->lm_head_f32) {
        #pragma omp parallel for collapse(2) if(N * G4_VOCAB > 65536)
        for (int t = 0; t < N; t++) {
            for (int j = 0; j < G4_VOCAB; j++) {
                const float *xm = normed + (size_t)t * G4_HIDDEN;
                const float *w = model->lm_head_f32 + (size_t)j * G4_HIDDEN;
                float sum = 0.0f;
                for (int k = 0; k < G4_HIDDEN; k++) sum += xm[k] * w[k];
                logits[(size_t)t * G4_VOCAB + j] = sum;
            }
        }
    } else {
        wubu_gpu_tag = 4;
        g4_batched_qmatmul(normed, &model->output, N, G4_VOCAB, G4_HIDDEN, logits);
        wubu_gpu_tag = 0;
    }

    if (N == 1) G4_P[G4P_HEAD] += g4_now_ms();

    /* ===== Softcap ===== */
    g4_softcap(logits, N * G4_VOCAB, G4_SOFTCAP);

    model->current_pos += N;
    free(positions);
}

/* ============ Token-based forward ============ */

void g4_model_forward_from_tokens(g4_model_t *model, const int *tokens, int B, int T, float *logits) {
    const int N = B * T;
    float *embd = (float *)malloc((size_t)N * G4_HIDDEN * sizeof(float));

    int row_bytes = (int)(model->token_embd.raw_bytes / G4_VOCAB);
    for (int i = 0; i < N; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= G4_VOCAB) tok = 0;
        const uint8_t *row = NULL;
        uint8_t *tmp = NULL;
        if (model->token_embd.data) {
            row = model->token_embd.data + (size_t)tok * row_bytes;
        } else if (model->gguf_file && row_bytes > 0) {
            tmp = (uint8_t *)malloc((size_t)row_bytes);
            uint64_t tensor_off = model->token_embd.stream_info
                ? ((gguf_tensor_info *)model->token_embd.stream_info)->data_offset : 0;
            if (tmp && g4_fseek_i64(model->gguf_file,
                    model->gguf_data_offset + tensor_off +
                    (uint64_t)tok * (uint64_t)row_bytes) == 0 &&
                fread(tmp, 1, (size_t)row_bytes, model->gguf_file) == (size_t)row_bytes) {
                row = tmp;
            }
        }
        if (row) {
            gguf_dequantize(row, model->token_embd.ggml_type, G4_HIDDEN,
                            embd + (size_t)i * G4_HIDDEN);
        } else {
            memset(embd + (size_t)i * G4_HIDDEN, 0, (size_t)G4_HIDDEN * sizeof(float));
        }
        free(tmp);
    }

    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < N * G4_HIDDEN; i++) embd[i] *= scale;
    if (getenv("G4_DEBUG_EMBD")) {
        for (int i = 0; i < N; i++) {
            int bad = 0; double l2 = 0;
            const float *r = embd + (size_t)i * G4_HIDDEN;
            for (int d = 0; d < G4_HIDDEN; d++) {
                if (!isfinite(r[d])) bad++;
                else l2 += (double)r[d] * r[d];
            }
            fprintf(stderr, "[EMBDROW %d] tok=%d bad=%d l2=%.3f\n",
                    i, tokens[i], bad, sqrt(l2));
        }
    }

    g4_model_forward(model, embd, B, T, logits);
    free(embd);
}

/* ============ Single-token decode ============ */

void g4_model_decode(g4_model_t *model, int token, float *logits) {
    g4_model_forward_from_tokens(model, &token, 1, 1, logits);
}

/* ============ Reset ============ */

void g4_model_reset(g4_model_t *model) {
    model->current_pos = 0;
    for (int i = 0; i < model->n_layers; i++)
        model->kv_cache[i].size = 0;
}

/* ============ Destroy ============ */

void g4_model_destroy(g4_model_t *model) {
    g4_prefetch_shutdown(model); /* join worker before stream_ctx/file closes */
    g4_cache_clear();
    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        free(l->attn_norm_weight);
        free(l->attn_q_norm_weight);
        if (l->attn_k_norm_weight && l->attn_k_norm_weight != l->attn_q_norm_weight)
            free(l->attn_k_norm_weight);
        free(l->post_attn_norm_weight);
        free(l->ffn_norm_weight);
        free(l->post_ffn_norm_weight);
        free(l->rope_freqs);
    }
    free(model->layers);
    free(model->output_norm_weight);
    free(model->lm_head_f32);
    if (model->data_blob) {
        if (model->data_blob_is_mmap) {
            void *base = model->data_blob_mmap_base
                         ? model->data_blob_mmap_base : model->data_blob;
            size_t len = model->data_blob_mmap_len
                         ? model->data_blob_mmap_len : model->data_blob_size;
#ifdef _WIN32
            wubu_munmap(base, len);
#else
            munmap(base, len);
#endif
        } else {
            free(model->data_blob); /* posix_memalign'd blob */
        }
    }

    if (model->kv_cache) {
        for (int i = 0; i < model->n_layers; i++) {
            free(model->kv_cache[i].k);
            free(model->kv_cache[i].v);
        }
        free(model->kv_cache);
    }

    free(model->buf_q);
    free(model->buf_k);
    free(model->buf_v);
    free(model->buf_attn_out);
    free(model->buf_normed);
    free(model->buf_embd);
    free(model->buf_ffn_gate);
    free(model->buf_ffn_up);
    free(model->buf_ffn_out);
    free(model->buf_scores);
    free(model->buf_row);

    /* Close GGUF file and free tensor info */
    if (model->stream_ctx) {
        /* gguf_close owns the file, tensors and ctx in streaming mode */
        gguf_close((gguf_ctx *)model->stream_ctx);
        model->gguf_file = NULL;
        model->gguf_tensor_info = NULL;
    } else {
        if (model->gguf_file) fclose(model->gguf_file);
        free(model->gguf_tensor_info);
    }
}
