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

void g4_batched_qmatmul(const float *x, const g4_qweight_t *w,
                               int M, int N, int K, float *c) {
    if (!w) return;
    if (!G4_CPU_STATS_hooked) { G4_CPU_STATS_hooked = 1; atexit(g4_cpu_stats_dump); }
    const double t_cpu0 = g4_now_ms();
    /* Device-resident weights: decode (M == 1) runs entirely on the GPU. */
    if (w->dev && M == 1 && (K % 256) == 0 &&
        (w->ggml_type == GGML_TYPE_IQ2_S || w->ggml_type == GGML_TYPE_IQ3_S ||
         w->ggml_type == GGML_TYPE_Q3_K || w->ggml_type == GGML_TYPE_Q4_0)) {
        const long long stride = (long long)(w->raw_bytes / N);
        if (wubu_gpu_matvec(w->dev, w->ggml_type, x, K, N, stride, c) == 0) return;
    }
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

    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        int pos = positions[t];
        int kv_start = (pos >= window) ? (pos - window + 1) : 0; /* exactly `window` keys incl. current */
        int kv_len = pos - kv_start + 1;
        if (kv_len > cache->size) kv_len = cache->size;
        if (kv_len <= 0) continue;
        int kv_end = kv_start + kv_len - 1;

        for (int hq = 0; hq < n_heads; hq++) {
            int kv_h = hq / (n_heads / kv_heads); /* GQA: consecutive Q heads share a KV head */
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;

            /* Scores in thread-local part of scores_buf */
            float *s = scores_buf + (size_t)(t * n_heads + hq) * window;

            float max_s = -FLT_MAX;
            for (int kp = kv_start; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++)
                    score += q_vec[d] * kv[d];
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

            for (int d = 0; d < hd; d++) {
                float val = 0.0f;
                for (int kp = kv_start; kp <= kv_end; kp++) {
                    val += s[kp - kv_start] * inv_sum *
                           (cache->v + (size_t)kp * kvd + (size_t)kv_h * hd)[d];
                }
                attn_out[(size_t)t * qd + (size_t)hq * hd + d] = val;
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

    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        int pos = positions[t];
        int kv_end = kv_size - 1;
        if (kv_end > pos) kv_end = pos;
        int kv_len = kv_end + 1;
        if (kv_len <= 0) continue;

        for (int hq = 0; hq < n_heads; hq++) {
            int kv_h = hq / (n_heads / kv_heads); /* GQA: consecutive Q heads share a KV head */
            const float *q_vec = q + (size_t)t * qd + (size_t)hq * hd;
            float *s = scores_local + (size_t)(t * n_heads + hq) * kv_size;

            float max_s = -FLT_MAX;
            for (int kp = 0; kp <= kv_end; kp++) {
                const float *kv = cache->k + (size_t)kp * kvd + (size_t)kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++)
                    score += q_vec[d] * kv[d];
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

void g4_layer_forward_cpu(g4_model_t *model, int il, float *x,
                                 float *normed, float *attn_out,
                                 float *q, float *k, float *v,
                                 float *ffn_gate, float *ffn_up, float *ffn_down_out,
                                 float *scores, float *row, int N, const int *positions) {
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

    /* ===== Q projection (batched quantized matmul) ===== */
    g4_batched_qmatmul(normed, &layer->attn_q, N, qd, G4_HIDDEN, q);

    /* ===== K, V projections ===== */
    if (!layer->share_kv) {
        g4_batched_qmatmul(normed, &layer->attn_k, N, kvd, G4_HIDDEN, k);
        if (layer->kv_eq) {
            memcpy(v, k, (size_t)N * kvd * sizeof(float));
        } else {
            g4_batched_qmatmul(normed, &layer->attn_v, N, kvd, G4_HIDDEN, v);
        }
    }

    /* ===== Q/K head norms ===== */
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
                float inv_rms = 1.0f / sqrtf(ss / (float)hd + G4_RMS_EPS);
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

    g4_kv_cache_t *cache = g4_kv_get(model, layer);

    /* ===== Attention ===== */
    if (layer->is_full) {
        g4_full_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    } else {
        g4_sliding_attn_cpu(model, layer, cache, q, attn_out, N, positions, scores, row);
    }

    /* ===== Attention output projection ===== */
    g4_batched_qmatmul(attn_out, &layer->attn_out, N, G4_HIDDEN, qd, normed);

    /* ===== Post-attention norm + residual ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(normed + (size_t)t * G4_HIDDEN, layer->post_attn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += normed[(size_t)t * G4_HIDDEN + d];
    }

    /* ===== FFN ===== */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++)
        g4_rms_norm(x + (size_t)t * G4_HIDDEN, layer->ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, normed + (size_t)t * G4_HIDDEN);

    /* Quantize input to Q8_K once, reuse across gate/up/down */
    g4_batched_qmatmul(normed, &layer->ffn_gate, N, G4_FFN, G4_HIDDEN, ffn_gate);
    g4_batched_qmatmul(normed, &layer->ffn_up,   N, G4_FFN, G4_HIDDEN, ffn_up);

    /* GELU + element-wise multiply (OpenMP parallel) */
    int ffn_elems = N * G4_FFN;
    #pragma omp parallel for if(ffn_elems > 4096)
    for (int i = 0; i < ffn_elems; i++)
        ffn_gate[i] = g4_gelu_tanh(ffn_gate[i]);
    #pragma omp parallel for if(ffn_elems > 4096)
    for (int i = 0; i < ffn_elems; i++)
        ffn_up[i] *= ffn_gate[i];

    g4_batched_qmatmul(ffn_up, &layer->ffn_down, N, G4_HIDDEN, G4_FFN, ffn_down_out);

    /* Post-FFN norm + residual */
    #pragma omp parallel for if(N > 1)
    for (int t = 0; t < N; t++) {
        g4_rms_norm(ffn_down_out + (size_t)t * G4_HIDDEN, layer->post_ffn_norm_weight,
                    G4_HIDDEN, G4_RMS_EPS, ffn_down_out + (size_t)t * G4_HIDDEN);
        for (int d = 0; d < G4_HIDDEN; d++)
            x[(size_t)t * G4_HIDDEN + d] += ffn_down_out[(size_t)t * G4_HIDDEN + d];
    }

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
        g4_batched_qmatmul(normed, &model->output, N, G4_VOCAB, G4_HIDDEN, logits);
    }

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
