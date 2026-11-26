/* wubu_inkling2_fwd.c -- EG-118 step 3: numeric primitives and the embedding path.
 *
 * Deliberately unoptimised: every weight is dequantised to f32 and multiplied with a plain dot product.
 * That is far too slow to generate text with, but it makes the graph checkable, and each primitive here
 * is written to be validated independently before anything is built on top of it.
 *
 * Tensor layout follows ggml: dims[0] is the fastest-varying axis, so a linear weight with
 * dims (n_in, n_out) is row-major as W[o * n_in + i].
 */
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_inkling2.h"

/* per-type cost accounting, read out by IN2_TIMETY. Defined at file scope because both matvec paths feed it. */
double in2_ty_sec[256];
long   in2_ty_calls[256];
double in2_ty_macs[256];
double in2_lay_sec, in2_moe_sec;

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);

#ifdef _WIN32
#include <windows.h>
/* Row-parallel kernel dispatch.
 *
 * Every row of a projection is independent and writes only its own output element, and wubu_iq_rows already takes
 * an r0/r1 range, so the split needs no accumulation changes and no new arithmetic: N threads, N disjoint row
 * ranges, same kernel. The engine is otherwise single-threaded on an 8-core box.
 *
 * Gated by IN2_THREADS (default 1 = off) until it is measured against the harness. */
typedef struct {
    int ty; const void *w; size_t rb; int n;
    const void *const *act; int nt; float *const *out; int r0, r1;
} in2_rows_job;

static DWORD WINAPI in2_rows_worker(LPVOID p)
{
    in2_rows_job *j = (in2_rows_job *)p;
    wubu_iq_rows(j->ty, j->w, j->rb, j->n, j->act, j->nt, j->out, j->r0, j->r1);
    return 0;
}

static int in2_threads(void)
{
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("IN2_THREADS");
#ifdef _OPENMP
        /* Parallel by default: a decode step makes ~270 matvec calls, so the split has to be cheap. Spawning a
         * fresh Windows thread per call (what this used to do) costs more than the split saves - measured 1.15x
         * from 1 to 4 threads against ~3.5x for the same kernels under a persistent pool. */
        n = e ? atoi(e) : omp_get_max_threads();
#else
        n = e ? atoi(e) : 1;
#endif
        if (n < 1) n = 1;
        if (n > 8) n = 8;
    }
    return n;
}

/* returns 1 if the work was done by threads */
static int in2_rows_parallel(int ty, const void *w, size_t rb, int n,
                             const void *const *act, int nt, float *const *out, int n_out)
{
    const int NT = in2_threads();
    if (NT < 2 || n_out < 256) return 0;
    in2_rows_job jobs[8];
    int chunk = (n_out + NT - 1) / NT;
    for (int i = 0; i < NT; ++i) {
        jobs[i].ty = ty; jobs[i].w = w; jobs[i].rb = rb; jobs[i].n = n;
        jobs[i].act = act; jobs[i].nt = nt; jobs[i].out = out;
        jobs[i].r0 = i * chunk;
        jobs[i].r1 = (i + 1) * chunk;
        if (jobs[i].r1 > n_out) jobs[i].r1 = n_out;
    }
    /* Persistent pool, not CreateThread per call: the range work is identical, so the split cannot change results,
     * but 270 calls per decode step x a thread spawn each was pure overhead. */
#ifdef _OPENMP
    #pragma omp parallel for num_threads(NT) schedule(static)
    for (int i = 0; i < NT; ++i)
        if (jobs[i].r0 < jobs[i].r1) in2_rows_worker(&jobs[i]);
#else
    for (int i = 0; i < NT; ++i)
        if (jobs[i].r0 < jobs[i].r1) in2_rows_worker(&jobs[i]);
#endif
    return 1;
}
#endif

/* Cache on by default; IN2_NO_KV forces the full-recompute path. Keeping this in one place matters because the
 * tool and the engine each used to test IN2_KV separately, and a caller asking for an incremental step while the
 * engine declined to attach a cache failed at the second token with rc=-6. */
int in2_kv_enabled(void);
/* On by default: the incremental path is bit-identical to full recompute (max|diff| = 0.0 over the hidden state of
 * every token tested) once in2_kv_conv_row summed its taps in the batch conv's order. IN2_NO_KV forces the O(n^2)
 * full recompute, which is still the honest way to check a cache change. */
int in2_kv_enabled(void) { return getenv("IN2_NO_KV") == NULL; }
static int g_force_cache;
int  in2_dq_force_begin(void) { const int s = g_force_cache; g_force_cache = 1; return s; }
void in2_dq_force_end(int s) { g_force_cache = s; }

double in2_arq_sec, in2_mixfn_sec, in2_tail_sec; long in2_arq_calls;
double in2_alloc_sec;
double in2_full_sec, in2_slide_sec;
double in2_phase[8];   /* mix, q, kv, rel/sink, attn, gate/wo, resconv, mlp+moe */
static clock_t _tk;    /* file scope so the phase macro works in every layer body */

/* High-resolution clock. clock() on Windows/MSYS resolves at ~15.6 ms, so every region shorter than that reads as
 * exactly 0.000 s - which is why in2_mix, the arq load, the post-mlp conv and the callocs all measured 0.000 while
 * their sum over 40 layer visits is real. That aliasing produced three contradictory answers from the same span. */
static double in2_now(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}
static double _tkh;    /* high-res companion to _tk */
#define TKPH(i) do { const double _n = in2_now(); in2_phase[i] += _n - _tkh; _tkh = _n; } while (0)
double in2_wall_now(void) { return in2_now(); }
/* full-attention layer sections: state/mix, q, kv+conv, attention, gate+wo, mlp+moe */
double in2_fph[8];
double in2_kvproj_sec, in2_kvconv_sec;
#include "wubu_inkling2_kv.h"
const unsigned char *wubu_inkling2_map_ptr(gguf_ctx *, const wubu_in2_tensor_t *);
#include "wubu_dequant_fp4.h"
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"

/* block_q8_K lives in wubu_iq2q8.h now, together with its quantiser, so the layout is defined once and can be
 * tested against ggml's own ref quantiser in isolation. */

/* in2_q8k_quantise() now lives in wubu_act_quant.c: one definition, testable against ggml's ref. */
void dot_row_iq2_xs(const uint8_t *data, const float *x, int64_t n_elems, float *out);

/* Types proven bit-identical by the local sweep (scratch iq_verify.c): K-quants and the IQ2/IQ3/IQ1_M families.
 * New types are added here only after their sweep passes, never merely because wubu_iq_supported() returns 1. */
static int in2_ty_verified(int ty)
{
    const char *only = getenv("IN2_TY_ONLY");   /* isolate one type when a new kernel disagrees end to end */
    if (only) return ty == atoi(only);
    switch (ty) {
        case 8: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 22:
        case 23: case 29: case 40: case 42: return 1;
        default: return 0;
    }
}

/* ---- prototypes (definitions follow in dependency order) ---- */
void in2_mix(const float *blocks, int n_blocks, int block_stride, int pos_off,
             const float *partial, const float *queries, int n_q, int d, float eps, float *out);
void in2_shortconv_dynamic(const float *x, int n_tok, int channels, const float *stat,
                           const float *dyn_proj, const float *dyn_input, int hidden,
                           int n_head, int k, float *out);
int  in2_layer_full(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                    int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                    const float *msgs);
void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out);
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y);
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out);
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold);
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias);
void in2_expert_act(const float *gate, const float *up, int n, float *out);
int  in2_dequant_slice(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off, int64_t n_elems, float *dst);
int  in2_matvec_quant(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off,
                      const float *x, int n_out, int n_in, float *y);
int  in2_router(const float *x, const float *W, const float *corr_bias, int n_expert, int d_model,
                int top_k, float scale, int *idx, float *w);
int  in2_moe(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
             int il, const float *x, int n_tok, float *out);
/* KV cache hook. The forward pass still computes every position, so this only exercises the store path: each
 * visit writes its final (post-clip / post-norm) K and V rows plus the raw rows the 4-tap conv will need, then
 * reads them back and compares. Nothing consumes the cache yet, which is why it is gated and why a wrong index
 * cannot affect a single number the engine produces. */
static in2_kv_t *g_kv;
static int g_visit;
static int g_kv_pos;

/* shared with the full-attention layers, which live in another translation unit but must use the same visit
 * numbering or their rows land in a different slot every pass */
in2_kv_t *in2_kv_current(int *visit, int *pos0)
{
    if (visit) *visit = g_visit;
    if (pos0) *pos0 = g_kv_pos;
    return g_kv;
}
void in2_kv_advance(void) { ++g_visit; }

/* Called once per forward: visits always start at zero for a step, and the step's first token sits at pos0.
 * Without this a second step would number its visits 45..90 and every lookup past the allocation would return
 * NULL, which the attention loops skip - a silently truncated context rather than a loud failure. */
void in2_kv_begin(int pos0) { g_visit = 0; g_kv_pos = pos0; }

/* Second history record for the two convs that are not K/V: the residual conv after attention and the MoE conv.
 * They carry three positions of history each, at d_model rather than kdim, so they get their own cache instance
 * with the same visit numbering (kraw = residual stream, vraw = moe stream). Missing these is why a 1-token step
 * produced different tokens from a full recompute: K and V were cached, these two were still pad-with-zeros. */
static in2_kv_t *g_kv2;
int g_kv_window = 0;   /* ring size of the K/V cache, 0 = linear slots */

static void in2_conv_stream(in2_kv_t *kvc, int vvis, int which, const float *x, int n_tok,
                            int ch, const float *w, int k, float *out)
{
    if (!kvc) { in2_shortconv(x, n_tok, ch, w, k, out); return; }
    if (n_tok == 1) {
        const float *hist = which ? in2_kv_hist_v(kvc, vvis, 3) : in2_kv_hist_k(kvc, vvis, 3);
        in2_kv_conv_row(hist, x, ch, w, k, out);
        if (which) in2_kv_store_row_v(kvc, vvis, x);
        else       in2_kv_store_row_k(kvc, vvis, x);
    } else {
        /* a real batch: the batch conv is already correct for it, and the history it must leave behind is its
         * own last rows, pushed one per position so the record ends up in order */
        in2_shortconv(x, n_tok, ch, w, k, out);
        const int from = n_tok > 3 ? n_tok - 3 : 0;
        for (int t = from; t < n_tok; ++t) {
            if (which) in2_kv_store_row_v(kvc, vvis, x + (size_t)t * ch);
            else       in2_kv_store_row_k(kvc, vvis, x + (size_t)t * ch);
        }
    }
}

/* same two streams, reachable from the full-attention translation unit */
void in2_conv_stream_res(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out)
{ in2_conv_stream(g_kv2, vvis, 0, x, n_tok, ch, w, k, out); }
void in2_conv_stream_moe(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out)
{ in2_conv_stream(g_kv2, vvis, 1, x, n_tok, ch, w, k, out); }

void in2_kv_attach(in2_kv_t *kv, int pos0)
{
    g_kv = kv;
    g_visit = 0;
    g_kv_pos = pos0;
}

int  in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                       int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                       const float *msgs);
void in2_messages(const float *srcs, int n_src, int n_tok,
                  const float *qres, const float *qkv, int li, int d, float eps, float *out);
void in2_message_gate(const float *h, const float *message, const float *down, const float *forget,
                      const float *msgp, int d, int rank, float eps, float *out);
float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out);
/* Read-only view of the dequant cache's own buffer. Defined below; declared here because the layer paths above use it
 * and an implicit declaration silently truncates the returned pointer to int. */
const float *in2_tensor_borrow(gguf_ctx *ctx, const wubu_in2_tensor_t *t);

/* Unweighted RMS norm: x * rsqrt(mean(x^2) + eps). Used by the loop message gates. */
void in2_rmsnorm_unweighted(const float *x, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale;
}

/* Weighted RMS norm, the form used by attn_norm / ffn_norm / the full-layer input norms. */
void in2_rmsnorm(const float *x, const float *w, int n, float eps, float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

/* y[o] = dot(dequant(tensor row o), x), dequantising ONE ROW AT A TIME into a small buffer that stays in
 * L1 instead of materialising the whole expert as floats.
 *
 * The materialising path was the dominant memory cost of this port: an expert's gate_up is 2*1152*2304
 * floats = 21 MB and its down is 10.6 MB, i.e. 32 MB written and then read back per expert, while the
 * quantised expert itself is only ~1.5 MB (IQ2_XS) + ~0.8 MB (Q2_0). That is a 14x amplification, times 8
 * experts times 20 sliding layers - 5 GB of transient traffic per token that carries no information.
 * Fused, the traffic is the quant bytes plus a 9 KB row that is reused. */
/* ---------------- optional device weight tier (LP-22) ----------------
 *
 * gen_fast.exe must not link the CUDA library: the engine has to run on a box with no card, and a hard import would
 * turn that into a missing-DLL dialog instead of a slower token. So the entry points are resolved once, lazily, and
 * every call site falls back to the CPU kernel whenever they are missing.
 */
static void *in2_gpu_res_fn;
static int   in2_gpu_state;      /* 0 = untried, 1 = ready, -1 = not available */
static double in2_gpu_sec;
static int64_t in2_gpu_calls;

#ifdef _WIN32
typedef int (*in2_gpu_res_sig)(int, const void *, size_t, const void *, int, float *, int, int, int64_t);
typedef int (*in2_gpu_up_sig)(const void *, const void *);
extern const void *wubu_iq2xxs_grid_bytes(void);
extern const void *wubu_iq2xxs_ksigns_bytes(void);

static void in2_gpu_init(void)
{
    if (in2_gpu_state) return;
    in2_gpu_state = -1;
    const char *nm = getenv("IN2_CUDA_DLL");
    HMODULE h = LoadLibraryA(nm && nm[0] ? nm : "wubu_cuda.dll");
    if (!h) return;
    in2_gpu_res_fn = (void *) (size_t) GetProcAddress(h, "wubu_cuda_resident_matvec");
    if (!in2_gpu_res_fn) return;
    /* The IQ2_XXS kernel reads a grid and a sign table that the library cannot supply itself without linking this
     * translation unit; the engine has both, so it hands them over once. */
    in2_gpu_up_sig up = (in2_gpu_up_sig) (void *) (size_t) GetProcAddress(h, "wubu_cuda_upload_iq2xxs");
    if (up) up(wubu_iq2xxs_grid_bytes(), wubu_iq2xxs_ksigns_bytes());
    /* IQ4_XS is the output head's format and its kernel reads a 16-entry non-linear table. Without this the head
     * returns zeros while every call still reports success, which is how the first attempt produced id=0 logit=0. */
    {
        typedef int (*iq4nl_sig)(const void *);
        iq4nl_sig t4 = (iq4nl_sig) (void *) (size_t) GetProcAddress(h, "wubu_cuda_upload_table_iq4nl");
        extern const void *wubu_iq4nl_bytes(void);
        if (t4) t4(wubu_iq4nl_bytes());
    }
    {
        typedef int (*iq2xs_sig)(const void *);
        iq2xs_sig t2 = (iq2xs_sig) (void *) (size_t) GetProcAddress(h, "wubu_cuda_upload_tables_iq2xs");
        (void) t2;
    }
    in2_gpu_state = 1;
}

static int in2_gpu_resident(int type, const void *w, size_t wbytes, const void *act, int act_bytes,
                            float *y, int K, int n_cols, int64_t stride)
{
    return ((in2_gpu_res_sig) in2_gpu_res_fn)(type, w, wbytes, act, act_bytes, y, K, n_cols, stride);
}
#else
static void in2_gpu_init(void) { in2_gpu_state = -1; }
static int in2_gpu_resident(int t, const void *w, size_t wb, const void *a, int ab, float *y,
                            int K, int nc, int64_t st)
{ (void) t; (void) w; (void) wb; (void) a; (void) ab; (void) y; (void) K; (void) nc; (void) st; return -1; }
#endif

double in2_gpu_seconds(void) { return in2_gpu_sec; }
long long in2_gpu_call_count(void) { return (long long) in2_gpu_calls; }

int in2_matvec_quant(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off,
                     const float *x, int n_out, int n_in, float *y)
{
    if (n_in <= 0 || n_out <= 0) return WUBU_INKLING2_ERR_IO;
    float *row = (float *)malloc((size_t)n_in * sizeof(float));
    if (!row) return WUBU_INKLING2_ERR_IO;
    /* Q2_0 (`down` experts) never becomes floats. dot_row_q2_0 unpacks the 2-bit fields straight into an AVX2
     * multiply-accumulate: measured 12.64x against dequantise-then-dot on a 2304-element row, worst relative
     * error 5.9e-08 over 64 random block sets (tools/dot_test.c). The other expert types still materialise. */
    const int ty_mv = (t && t->info) ? t->info->ggml_type : -1;
    /* Generic kernel path: any type wubu_iq_supported() knows goes through the translated AVX2 row kernels with
     * ONE call for every row and the activation quantised once. Which activation format a type needs is a property
     * of the ggml block layout, not a choice: Q8_0/Q2_0/NVFP4 use 32-element block_q8_0, the K-quants and the
     * IQ2/IQ3/IQ1 families use 256-element block_q8_K. Sending the wrong one gives plausible numbers silently, so
     * it is keyed off the type. */
    if (getenv("IN2_TRACE_TY") && t && t->info)
        fprintf(stderr, "matvec ty=%d n_in=%d n_out=%d supported=%d\n", ty_mv, n_in, n_out, wubu_iq_supported(ty_mv));
    /* Does this function even see the dense projections, and does it see them with a name? The resident tier's gate
     * requires info->name[0] != 0; if the descriptor arrives with an empty name the gate fails silently and the only
     * visible symptom is that the tensor is never resident. Print the name itself, not a verdict about it. */
    if (getenv("IN2_Q_TRACE") && ty_mv >= 0 && ty_mv < 256) {
        static unsigned char seen[256][2];
        const int has = (t && t->info && t->info->name[0]) ? 1 : 0;
        if (!seen[ty_mv][has]) {
            seen[ty_mv][has] = 1;
            fprintf(stderr, "Q ty=%d n_in=%d n_out=%d name='%s' hasnamed=%d\n", ty_mv, n_in, n_out,
                    (t && t->info) ? t->info->name : "(no info)", has);
        }
    }
    /* Only types whose bit-identical sweep I have personally re-run go through the kernels. wubu_iq_supported()
     * reports what exists in the translation unit, which is not the same thing as what has been proven: a
     * half-finished type returning 1 silently changed the logit from 29.67 to 22.52, and a 7-logit drift with the
     * top token unchanged is exactly the failure a "supported=1" flag cannot catch. */
    if (getenv("IN2_NOFAST") == NULL && t && t->info && wubu_iq_supported(ty_mv) && in2_ty_verified(ty_mv)) {
        const int q8_0_act = (ty_mv == 8 || ty_mv == 40 || ty_mv == 42);
        const int be = q8_0_act ? 32 : 256;
        const int64_t rb = (n_in % be == 0) ? gguf_raw_size(ty_mv, n_in) : -1;
        if (rb > 0) {
            unsigned char *whole = (unsigned char *)malloc((size_t)rb * (size_t)n_out);
            void *actbuf = q8_0_act
                ? malloc(((size_t)n_in / 32) * sizeof(in2_q8_0_blk))
                : malloc(((size_t)n_in / 256) * sizeof(in2_q8k_blk));
            const int64_t boff = (elem_off == 0) ? 0 : gguf_raw_size(ty_mv, elem_off);
            if (whole && actbuf && boff >= 0) {
                wubu_in2_tensor_t sh = *t;
                sh.file_offset = t->file_offset + boff;
                /* read straight from the mapping when it exists: this path runs ~1150 times per decode token over
                 * ~1.5 MB expert matrices, so the copy was ~1.7 GB of memcpy per token on immutable mapped data */
                const unsigned char *msrc = wubu_inkling2_map_ptr(ctx, &sh);
                if (msrc || wubu_inkling2_read(ctx, &sh, whole, rb * n_out) == WUBU_INKLING2_OK) {
                    const unsigned char *src = msrc ? msrc : whole;
                    if (q8_0_act) in2_q8_0_quantise(x, n_in / 32, (in2_q8_0_blk *)actbuf);
                    else          in2_q8k_quantise(x, n_in / 256, (in2_q8k_blk *)actbuf);
                    const void *ap[1] = { actbuf };
                    float *op[1] = { y };
                    int gpu_done = 0;
                    /* Same weights, different address space. Per call the card receives the quantised activation
                     * (2.4 KB at n_in=2304 for q8_0) instead of the matrix (5.6 MB), so the dense tier stops paying
                     * the read that dominates a token. Experts are excluded by name: they are 48x128 distinct slabs,
                     * far more than the resident tier holds, and they are the thing the cache policy is for. */
                    if (getenv("IN2_GPU_DENSE") && t->info && t->info->name[0] &&
                        strstr(t->info->name, "exps") == NULL) {
                        in2_gpu_init();
                        if (in2_gpu_state == 1) {
                            const int act_bytes = q8_0_act
                                ? (n_in / 32) * (int) sizeof(in2_q8_0_blk)
                                : (n_in / 256) * (int) sizeof(in2_q8k_blk);
                            const clock_t _g0 = clock();
                            const int g_rc = in2_gpu_resident(ty_mv, src, (size_t) rb * (size_t) n_out, actbuf,
                                                              act_bytes, y, n_in, n_out, (int64_t) rb);
                            /* A refusal must never be silent: the first attempt at this path reported success
                             * while the head returned zeros, and the only visible symptom was id=0 logit=0. */
                            if (g_rc != 0 && getenv("IN2_GPU_TRACE")) {
                                static int warned[256];
                                if (ty_mv >= 0 && ty_mv < 256 && !warned[ty_mv]) {
                                    warned[ty_mv] = 1;
                                    fprintf(stderr, "GPU refused ty=%d rc=%d n_in=%d n_out=%d (%s)\n",
                                            ty_mv, g_rc, n_in, n_out, t->info->name);
                                }
                            }
                            if (g_rc == 0) {
                                gpu_done = 1;
                                in2_gpu_sec += (double) (clock() - _g0) / (double) CLOCKS_PER_SEC;
                                if (in2_gpu_calls == 0)
                                    fprintf(stderr, "GPU tier: resident dense matvec engaged (%s)\n",
                                            t->info->name);
                                in2_gpu_calls += 1;
                            }
                        }
                    }
                    if (!gpu_done) { const clock_t _q0 = clock();
                    wubu_iq_rows(ty_mv, src, (size_t)rb, n_in, ap, 1, op, 0, n_out);
                    if (ty_mv >= 0 && ty_mv < 256) {
                        in2_ty_sec[ty_mv] += (double)(clock() - _q0) / (double)CLOCKS_PER_SEC;
                        in2_ty_calls[ty_mv] += 1;
                        in2_ty_macs[ty_mv] += (double)n_out * (double)n_in;
                    } }
                    /* Does the kernel agree with THIS engine's own dequantiser on REAL bytes from THIS file?
                     * A kernel can be bit-identical to ggml on synthetic blocks and still disagree here, which
                     * would mean the file's encoding of that type is not ggml's canonical struct. */
                    if (getenv("IN2_CMP")) {
                        float *r0 = (float *)malloc((size_t)n_in * sizeof(float));
                        if (r0) {
                            if (in2_dequant_slice(ctx, t, elem_off, n_in, r0) == WUBU_INKLING2_OK) {
                                double a0 = 0.0;
                                for (int i = 0; i < n_in; ++i) a0 += (double)r0[i] * (double)x[i];
                                fprintf(stderr, "CMP ty=%d row0 kernel=%.9g dequant=%.9g rel=%.3e\n",
                                        ty_mv, (double)y[0], a0, fabs((double)y[0] - a0) / (fabs(a0) > 1e-30 ? fabs(a0) : 1e-30));
                            }
                            free(r0);
                        }
                    }
                    free(actbuf); free(whole);
                    return WUBU_INKLING2_OK;
                }
            }
            free(actbuf); free(whole);
        }
    }
    /* Translated maya kernels (bit-identical to ggml's AVX2 dot, verified per type) cover the gate/up formats.
     * One call handles every row of the expert, so the activation is quantised once and the row loops stay inside
     * the kernel. */
    if (getenv("IN2_NOFAST") == NULL && (n_in % 256) == 0 &&
        (ty_mv == 16 || ty_mv == 17 || ty_mv == 22) && wubu_iq_supported(ty_mv)) {
        const int64_t rb = gguf_raw_size(ty_mv, n_in);
        unsigned char *whole = (rb > 0) ? (unsigned char *)malloc((size_t)rb * (size_t)n_out) : NULL;
        in2_q8k_blk *act = (in2_q8k_blk *)malloc(((size_t)n_in / 256) * sizeof(in2_q8k_blk));
        if (whole && act) {
            const int64_t eo   = elem_off;
            const int64_t boff = (eo == 0) ? 0 : gguf_raw_size(ty_mv, eo);
            wubu_in2_tensor_t sh = *t;
            sh.file_offset = t->file_offset + (boff < 0 ? 0 : boff);
            if (boff >= 0 && wubu_inkling2_read(ctx, &sh, whole, rb * n_out) == WUBU_INKLING2_OK) {
                in2_q8k_quantise(x, n_in / 256, act);   /* groups, not elements: 9 blocks for a 2304 row */
                const void *ap[1] = { (const void *)act };
                float *op[1] = { y };
                wubu_iq_rows(ty_mv, whole, (size_t)rb, n_in, ap, 1, op, 0, n_out);
                free(act);
                return WUBU_INKLING2_OK;
            }
        }
        free(act);
    }
    /* Types with a fused row kernel: Q2_0 (`down`) and IQ2_XS (gate/up). Everything else still materialises. */
    /* Both types now have a vectorised path (Q2_0 float AVX2, IQ2_XS int8 AVX2 against a once-per-matvec
     * activation). The IQ2_XS SCALAR fused form measured 14% SLOWER
     * than materialise-then-dot (48,014/48,121 ms against 42,230/42,227 ms, twice each, logits identical), because
     * the per-group double accumulator serialises 8 dependent adds while the materialised dot interleaves two.
     * So it stays behind IN2_FAST until an AVX2 IQ2_XS kernel exists - a quantised load path is not a speedup
     * just because it skips the buffer. */
    const int fast_ty = (getenv("IN2_NOFAST") == NULL && (ty_mv == 42 || ty_mv == 17)) ? ty_mv : -1;
    const int64_t row_bytes = (fast_ty > 0) ? gguf_raw_size(fast_ty, n_in) : -1;
    unsigned char *rawrow = (row_bytes > 0) ? (unsigned char *)malloc((size_t)row_bytes) : NULL;
    /* int8 activation for the IQ2_XS vector path: quantised ONCE here and reused by every row of the expert, which
     * is the amortisation that makes the kernel pay. Only when the type and the ISA both allow it. */
    in2_act_q8 *actq = NULL;
    if (fast_ty == 17 && (n_in % 32) == 0 && __builtin_cpu_supports("avx2")) {
        actq = (in2_act_q8 *)malloc(((size_t)n_in / 32) * sizeof(in2_act_q8));
        if (actq) in2_act_quantise(x, n_in / 32, actq);
    }
    for (int o = 0; o < n_out; ++o) {
        if (rawrow) {
            const int64_t eo   = elem_off + (int64_t)o * n_in;
            const int64_t boff = (eo == 0) ? 0 : gguf_raw_size(fast_ty, eo);
            if (boff >= 0) {
                wubu_in2_tensor_t sh = *t;
                sh.file_offset = t->file_offset + boff;
                if (wubu_inkling2_read(ctx, &sh, rawrow, row_bytes) == WUBU_INKLING2_OK) {
                    if (fast_ty == 42)     dot_row_q2_0(rawrow, x, n_in, &y[o]);
                    else if (actq)         y[o] = dot_row_iq2_xs_q8(rawrow, actq, n_in / 32);
                    else                   dot_row_iq2_xs(rawrow, x, n_in, &y[o]);
                    continue;
                }
            }
        }
        if (in2_dequant_slice(ctx, t, elem_off + (int64_t)o * n_in, n_in, row) != WUBU_INKLING2_OK) {
            free(actq);
            free(rawrow);
            free(row);
            return WUBU_INKLING2_ERR_IO;
        }
        double a0 = 0.0, a1 = 0.0;
        for (int i = 0; i + 1 < n_in; i += 2) { a0 += (double)row[i] * x[i]; a1 += (double)row[i + 1] * x[i + 1]; }
        for (int i = n_in & ~1; i < n_in; ++i) a0 += (double)row[i] * x[i];
        y[o] = (float)(a0 + a1);
    }
    free(actq);
    free(rawrow);
    free(row);
    return WUBU_INKLING2_OK;
}


/* Batched dense projection: Y[i] = W . X[i] for i in [0, nt).
 *
 * This replaces "materialise the weight matrix as floats, then dot it once per position". Two costs disappear:
 * the f32 copy (wq is 3072 x 2304 = 28 MB, rebuilt on every layer visit, so ~2 GB of dequantisation per token
 * across 24 layers x 3 passes) and the per-position scalar dot. The kernel decodes each weight row ONCE and
 * applies it to all nt tokens, which is the whole reason batching matters: per-position calls would re-read the
 * matrix and lose the reuse.
 *
 * nt is grouped in eights because the kernels take an array of activation pointers. Falls back to the exact
 * per-position float path, materialising only when the kernels cannot take the type. */
/* per-type cost accounting, filled when IN2_TIMETY is set */
double in2_ty_sec[256];
long   in2_ty_calls[256];
double in2_ty_macs[256];

int in2_matvec_nt(gguf_ctx *ctx, const wubu_in2_tensor_t *t,
                  const float *X, int nt, int n_in, int n_out, float *Y)
{
    const int ty = (t && t->info) ? t->info->ggml_type : -1;
    const int q8_0_act = (ty == 8 || ty == 40 || ty == 42);
    const int be = q8_0_act ? 32 : 256;
    const int64_t rb = (ty >= 0 && (n_in % be) == 0) ? gguf_raw_size(ty, n_in) : -1;

    const int rb_ok = (ty >= 0 && (n_in % be) == 0);
    if (getenv("IN2_TRACE_NT"))
        fprintf(stderr, "NT ty=%d n_in=%d n_out=%d raw=%d supported=%d verified=%d -> %s\n",
                ty, n_in, n_out, rb_ok, wubu_iq_supported(ty), in2_ty_verified(ty),
                (getenv("IN2_NOFAST") == NULL && rb_ok && wubu_iq_supported(ty) && in2_ty_verified(ty))
                    ? "kernel" : "FLOAT-FALLBACK");
    if (getenv("IN2_NOFAST") == NULL && rb > 0 && t && t->info &&
        wubu_iq_supported(ty) && in2_ty_verified(ty)) {
        const size_t ablk = q8_0_act ? sizeof(in2_q8_0_blk) : sizeof(in2_q8k_blk);
        const int nab = n_in / be;
        /* Reused across calls. A decode token makes ~1150 of these calls (48 visits x 8 experts x 3 matrices), and
         * each was a fresh malloc+free of ~1.5 MB - the allocation traffic cost more than the arithmetic. Grown
         * once to the largest matrix seen; the buffer is only ever read from the mapping into itself. */
        /* Thread-local: the MoE runs one expert per thread, and eight of them sharing this fallback buffer meant
         * concurrent reads/writes of the same bytes whenever the mapping was unavailable (or still being set up) -
         * which showed up as a greedy token flipping between runs, not as a crash. */
        static _Thread_local unsigned char *g_whole;
        static _Thread_local size_t g_whole_cap;
        const size_t need = (size_t)rb * (size_t)n_out;
        unsigned char *whole = g_whole;
        if (need > g_whole_cap) {
            unsigned char *nb = (unsigned char *)realloc(g_whole, need);
            if (nb) { g_whole = nb; g_whole_cap = need; whole = nb; }
            else whole = NULL;
        }
        void *act = malloc(ablk * (size_t)nab * (size_t)nt);
        const unsigned char *mapped = wubu_inkling2_map_ptr(ctx, t);
        if (act && (mapped || whole)) {
            const int have = mapped ? 1 : (wubu_inkling2_read(ctx, t, whole, rb * n_out) == WUBU_INKLING2_OK);
            if (have) {
                const unsigned char *base = mapped ? mapped : whole;
                for (int i = 0; i < nt; ++i) {
                    const float *x = X + (size_t)i * n_in;
                    void *a = (unsigned char *)act + ablk * (size_t)nab * (size_t)i;
                    if (q8_0_act) in2_q8_0_quantise(x, nab, (in2_q8_0_blk *)a);
                    else          in2_q8k_quantise(x, nab, (in2_q8k_blk *)a);
                }
                int done = 1;
                for (int g = 0; g < nt && done; g += 8) {
                    const int nn = (nt - g < 8) ? (nt - g) : 8;
                    const void *ap[8];
                    float *op[8];
                    for (int i = 0; i < nn; ++i) {
                        ap[i] = (unsigned char *)act + ablk * (size_t)nab * (size_t)(g + i);
                        op[i] = Y + (size_t)(g + i) * n_out;
                    }
                    const clock_t _k0 = clock();
                    /* LP-22: the resident tier gets first refusal. Same weights, different address space - the card
                     * receives the quantised activation (2.4 KB at n_in=2304 for q8_0) instead of the matrix (5.6 MB),
                     * which is where a decode step actually spends its time. Only nn==1 is taken because the resident
                     * entry runs the single-token kernel; a prefill batch falls through to the CPU kernels unchanged.
                     * Experts are excluded by name: 48x128 slabs is far more than the resident tier can hold. */
                    int gpu_took = 0;
                    if (nn == 1 && getenv("IN2_GPU_DENSE") && t && t->info && t->info->name[0] &&
                        strstr(t->info->name, "exps") == NULL) {
                        in2_gpu_init();
                        const int g_rc2 = (in2_gpu_state == 1)
                            ? in2_gpu_resident(ty, base, (size_t) rb * (size_t) n_out, ap[0],
                                               (int) (ablk * (size_t) nab), op[0], n_in, n_out, (int64_t) rb)
                            : -99;
                        if (g_rc2 != 0 && getenv("IN2_GPU_TRACE")) {
                            static int w2[256];
                            if (ty >= 0 && ty < 256 && !w2[ty]) {
                                w2[ty] = 1;
                                fprintf(stderr, "NT-GPU refused ty=%d rc=%d actb=%d wb=%zu (%s)\n",
                                        ty, g_rc2, (int) (ablk * (size_t) nab), (size_t) rb * (size_t) n_out,
                                        t->info->name);
                            }
                        }
                        if (g_rc2 == 0 && getenv("IN2_GPU_TRACE")) {
                            static int ok2[256];
                            if (ty >= 0 && ty < 256 && !ok2[ty]) {
                                ok2[ty] = 1;
                                fprintf(stderr, "NT-GPU ok ty=%d (%s)\n", ty, t->info->name);
                            }
                        }
                        if (g_rc2 == 0) {
                            gpu_took = 1;
                            if (in2_gpu_calls++ == 0)
                                fprintf(stderr, "GPU tier: engaged (%s)\n", t->info->name);
                        }
                    }
                    if (!gpu_took) {
#ifdef _WIN32
                    if (!in2_rows_parallel(ty, base, (size_t)rb, n_in, ap, nn, op, n_out))
#endif
                        wubu_iq_rows(ty, base, (size_t)rb, n_in, ap, nn, op, 0, n_out);
                    }
                    if (ty >= 0 && ty < 256) {
                        in2_ty_sec[ty] += (double)(clock() - _k0) / (double)CLOCKS_PER_SEC;
                        in2_ty_calls[ty] += 1;
                        in2_ty_macs[ty] += (double)nn * (double)n_out * (double)n_in;
                    }
                }
                free(act);
                return WUBU_INKLING2_OK;
            }
        }
        free(act);
    }

    /* Exact fallback: materialise once, dot per position. */
    float *W = in2_tensor_to_f32(ctx, t, NULL);
    if (!W) return WUBU_INKLING2_ERR_IO;
    for (int i = 0; i < nt; ++i)
        in2_matvec(W, X + (size_t)i * n_in, n_in, n_out, Y + (size_t)i * n_out);
    free(W);
    return WUBU_INKLING2_OK;
}


#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static float in2_dot_f32_avx2(const float *a, const float *b, int n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    __m256 s = _mm256_add_ps(s0, s1);
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float r = _mm_cvtss_f32(lo);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}
__attribute__((target("avx2,fma")))
static void in2_saxpy_avx2(float *y, const float *x, float a, int n)
{
    const __m256 av = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}
static int in2_avx2(void) { static int c = -1; if (c < 0) c = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); return c; }
#endif

/* attention core: QK dot and the weighted value sum, both 8-wide when AVX2 is present */
/* Double accumulation is deliberate, not conservatism. QK products here reach ~1e5 before the 1/sqrt(HD) scale,
 * so a 128-term float dot loses ~1 order more than the double sum, and the error is carried through every one of
 * the 48 layer visits. Measured: the same edit in the full-attention layer moved the final logit from 27.1610 to
 * 25.0769. The AVX2 float version was reverted because it bought nothing - 8,021 vs 8,041 ms at a 5-token prefix -
 * and this dot is not where the time goes. */
static float in2_attn_dot(const float *a, const float *b, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)a[i] * (double)b[i];
    return (float)acc;
}
static void in2_attn_accum(float *y, const float *x, float a, int n)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (in2_avx2()) { in2_saxpy_avx2(y, x, a, n); return; }
#endif
    for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

/* y[o] = dot(W + o*n_in, x) for o in [0, n_out). */
#if defined(__x86_64__) || defined(_M_X64)
/* Four independent double accumulators, eight floats per iteration. Double, not float: the QK experiment showed
 * float accumulation drifting 2 logits over 48 layer visits, and this function feeds the message gates that every
 * pass is conditioned on. Splitting into 4 partial sums changes the summation order but each partial is still
 * double, so the result differs from the scalar loop only at ~1e-16 relative - unlike a float accumulator. */
__attribute__((target("avx2,fma")))
static void in2_matvec_avx2(const float *W, const float *x, int n_in, int n_out, float *y)
{
    for (int o = 0; o < n_out; ++o) {
        const float *row = W + (size_t)o * n_in;
        __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
        int i = 0;
        for (; i + 16 <= n_in; i += 16) {
            s0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i)),      _mm256_cvtps_pd(_mm_loadu_ps(x + i)),      s0);
            s1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 4)),  _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)),  s1);
            s2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 8)),  _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)),  s2);
            s3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 12)), _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), s3);
        }
        for (; i + 4 <= n_in; i += 4)
            s0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i)), _mm256_cvtps_pd(_mm_loadu_ps(x + i)), s0);
        __m256d s = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
        __m128d lo = _mm256_castpd256_pd128(s), hi = _mm256_extractf128_pd(s, 1);
        lo = _mm_add_pd(lo, hi);
        double acc = _mm_cvtsd_f64(lo) + _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
        for (; i < n_in; ++i) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}
static int in2_have_avx2(void) { static int c = -1; if (c < 0) c = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); return c; }
#endif

void in2_matvec(const float *W, const float *x, int n_in, int n_out, float *y)
{
    /* Shape trace, because the float path carries no tensor identity and the shapes are unambiguous:
     * 2304->3072 is attn_q, 2304->1024 is attn_k/attn_v, 3072->2304 is attn_output, 2304->201024 is the head.
     * If this fires ~144x per token for q/k/v, then the dense projections never enter a typed kernel at all and the
     * quantised path in every in2_layer_* implementation is dead code for this checkpoint. */
    if (getenv("IN2_MV_TRACE")) {
        static unsigned long long calls;
        static int seen[64];
        int slot = -1;
        for (int i = 0; i < 64; ++i) if (!seen[i]) { slot = i; break; }
        int dup = 0;
        for (int i = 0; i < 64; ++i) {
            if (seen[i] == 1000000 + n_in * 100000 + n_out) { dup = 1; break; }
        }
        if (!dup && slot >= 0) {
            seen[slot] = 1000000 + n_in * 100000 + n_out;
            fprintf(stderr, "MV n_in=%d n_out=%d\n", n_in, n_out);
        }
        if ((calls++ % 500) == 0) fprintf(stderr, "MV call %llu\n", calls);
    }
#if defined(__x86_64__) || defined(_M_X64)
    if (in2_have_avx2()) { in2_matvec_avx2(W, x, n_in, n_out, y); return; }
#endif
    for (int o = 0; o < n_out; ++o) {
        const float *row = W + (size_t)o * n_in;
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}

/* Static depthwise causal short convolution over a sequence.
 *
 * Reference semantics (Inkling2LoopShortConvolution): nn.Conv1d(kernel=k, groups=channels,
 * padding=k-1, bias=False), i.e. output at step t sees x[t-k+1..t], followed by
 *     out = (conv(x) + x) / sqrt(k + 1)
 * The residual add and the 1/sqrt(k+1) scale are part of the block, not an optimisation detail:
 * dropping them changes the magnitude by ~2.2x.
 *
 * w is ggml (k, channels): ne0 = k is the FASTEST axis, so the weight for channel c, tap j lives at
 * w[c*k + j]. That matches nn.Conv1d's [out_channels, 1, kernel] layout. Indexing it as
 * w[j*channels + c] (tap-major) reads a plausible-looking but wrong filter: the per-tap magnitudes come
 * out flat, which is how the mistake was caught. Tap index == backward distance, so tap 0 is the
 * current token.
 */
void in2_shortconv(const float *x, int n_tok, int channels, const float *w, int k, float *out)
{
    const float scale = 1.0f / sqrtf((float)(k + 1));
    for (int t = 0; t < n_tok; ++t) {
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (int tap = 0; tap < k; ++tap) {
                /* Reference (llama.cpp SSM_CONV) builds in = [3 state rows, x0..xN] and computes
                 * out[t] = sum_i w[i]*in[t+i], so the LAST tap sees the current token and tap 0 is the
                 * oldest. The earlier `t - tap` put the current token on tap 0, which reverses the whole
                 * filter (proven against node_35 of the reference dump). */
                const int src = t - (k - 1 - tap);
                if (src < 0) continue;               /* left padding with zeros */
                acc += w[(size_t)c * k + tap] * x[(size_t)src * channels + c];
            }
            const float res = x[(size_t)t * channels + c];
            out[(size_t)t * channels + c] = (acc + res) * scale;
        }
    }
}

/* L2 clip, optionally preceded by a learned per-(head,dim) scale.
 *
 *   x *= scale;  row = x[h*dim .. h*dim+dim];  x *= min(1, threshold / (|row| + 1e-6))
 *
 * q uses this with a learned scale (attn_q_clip), k uses it bare, and the relative-position states use it
 * with rel_clip. The scale is stored in ggml order (dim, heads), i.e. scale[h*dim + d].
 */
void in2_l2_clip(float *x, const float *scale, int n_head, int dim, float threshold)
{
    for (int h = 0; h < n_head; ++h) {
        float *row = x + (size_t)h * dim;
        if (scale) for (int d = 0; d < dim; ++d) row[d] *= scale[(size_t)h * dim + d];
        double ss = 0.0;
        for (int d = 0; d < dim; ++d) ss += (double)row[d] * row[d];
        const float norm = sqrtf((float)ss);
        float f = threshold / (norm + 1e-6f);
        if (f > 1.0f) f = 1.0f;
        for (int d = 0; d < dim; ++d) row[d] *= f;
    }
}

/* Relative-position bias table: bias[h][dist] = sum_r rel_states[h][r] * P[dist][r].
 *
 * P is attn_rel_bias, stored in ggml order (d_rel, extent) so that r is the fastest axis:
 * P[dist * d_rel + r]. Computing the whole table once and indexing it by distance is what the reference
 * does, and it is cheaper than a gemm per (query,key) pair. Distances outside [0, extent) get 0, which
 * the caller handles by skipping them.
 */
void in2_rel_bias_table(const float *rel_states, const float *P, int n_head, int d_rel, int extent, float *bias)
{
    for (int h = 0; h < n_head; ++h) {
        const float *rs = rel_states + (size_t)h * d_rel;
        float *out = bias + (size_t)h * extent;
        for (int d = 0; d < extent; ++d) {
            const float *pr = P + (size_t)d * d_rel;
            double acc = 0.0;
            for (int r = 0; r < d_rel; ++r) acc += (double)rs[r] * pr[r];
            out[d] = (float)acc;
        }
    }
}

/* Read a block-aligned element range of a tensor and dequantise it. The MoE expert blocks are
 * (2*d_ff, d_model, n_expert) with expert slices contiguous and 256-aligned, so one expert can be read
 * without touching the other 127 (2.7 GB if dequantised whole). */
int in2_dequant_slice(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off, int64_t n_elems, float *dst)
{
    if (!ctx || !t || !t->info || !dst) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    /* Alignment must be a multiple of the TYPE's block size, not a flat 256: Q2_0's block is 64 elements,
     * so a row stride of DFF = 1152 is legal (18 blocks) even though 1152 % 256 = 128. A flat 256 check
     * rejected every expert's `down` row and made the fused path return ERR_SHAPE for the whole forward.
     * 64 is the finest block in use (Q2_0); Q8_0's 32 divides it, and the 256-block K/IQ types only ever
     * arrive here at offsets that are multiples of d = 2304. */
    if (elem_off % 64 != 0) return WUBU_INKLING2_ERR_SHAPE;
    const int ty = t->info->ggml_type;
    /* wubu_fp4_raw_size returns -1 for n_elems <= 0, so the byte offset of element 0 has to be
     * spelled out rather than asked for. Expert 0 of every tensor hit this. */
    const int64_t byte_off = (elem_off == 0) ? 0 : gguf_raw_size(ty, elem_off);
    const int64_t nbytes   = gguf_raw_size(ty, n_elems);
    if (byte_off < 0 || nbytes <= 0) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;
    unsigned char *raw = (unsigned char *)malloc((size_t)nbytes);
    if (!raw) return WUBU_INKLING2_ERR_IO;
    wubu_in2_tensor_t shifted = *t;
    shifted.file_offset = t->file_offset + byte_off;
    int rc = wubu_inkling2_read(ctx, &shifted, raw, nbytes);
    if (rc == WUBU_INKLING2_OK) gguf_dequantize(raw, ty, n_elems, dst);
    free(raw);
    return rc;
}

/* Softcap: tanh(x / c) * c. Bounds a value into (-c, c) without a hard clip. */
static float in2_softcap(float x, float c) { return tanhf(x / c) * c; }

/* Softcapped SwiGLU: the sigmoid sees the RAW gate; only the linear factor and `up` are softcapped.
 *   act = softcap(gate, 4.0) * sigmoid(gate) * softcap(up, 25.0)
 * Applying the softcap inside the sigmoid as well is a common misreading and changes the gate
 * non-linearly, so it is spelled out here. */
void in2_expert_act(const float *gate, const float *up, int n, float *out)
{
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + expf(-g));
        out[i] = in2_softcap(g, 4.0f) * sig * in2_softcap(up[i], 25.0f);
    }
}

/* Router: sigmoid scores, selection by score + correction bias, weights from the UNBIASED scores
 * renormalised to sum 1 and then multiplied by the routed scaling factor (8.0).
 *
 * The distinction matters: the correction bias only changes which experts are picked, never how much
 * they contribute. Returns the number of experts selected. */
int in2_router(const float *x, const float *W, const float *corr_bias, int n_expert, int d_model,
               int top_k, float scale, int *idx, float *w)
{
    float *scores = (float *)malloc((size_t)n_expert * sizeof(float));
    float *adj    = (float *)malloc((size_t)n_expert * sizeof(float));
    if (!scores || !adj) { free(scores); free(adj); return 0; }
    for (int e = 0; e < n_expert; ++e) {
        const float *row = W + (size_t)e * d_model;
        double acc = 0.0;
        for (int i = 0; i < d_model; ++i) acc += (double)row[i] * x[i];
        const float s = 1.0f / (1.0f + expf(-(float)acc));
        scores[e] = s;
        adj[e] = s + (corr_bias ? corr_bias[e] : 0.0f);
    }
    /* partial selection sort for top_k */
    char *taken = (char *)calloc((size_t)n_expert, 1);
    for (int k = 0; k < top_k; ++k) {
        int best = -1; float bestv = -1e30f;
        for (int e = 0; e < n_expert; ++e) if (!taken[e] && adj[e] > bestv) { bestv = adj[e]; best = e; }
        if (best < 0) break;
        taken[best] = 1; idx[k] = best;
    }
    double sum = 0.0;
    for (int k = 0; k < top_k; ++k) sum += scores[idx[k]];
    const float inv = (float)(1.0 / (sum + 1e-20));
    for (int k = 0; k < top_k; ++k) w[k] = scores[idx[k]] * inv * scale;
    free(scores); free(adj); free(taken);
    return top_k;
}

/* ------------------------------------------------------------------------------------------- */
/* A complete sliding-layer forward over n_tok tokens starting at absolute position pos0.       */
/*                                                                                              */
/* Order is taken from Inkling2LoopSlidingDecoderLayer.forward, which is not the conventional    */
/* pre-norm block:                                                                              */
/*                                                                                              */
/*   block_residual = h                                                                         */
/*   h = attention_input(block_states, h)        # depth mix, layer>0 only; empty -> bare rmsnorm */
/*   h = attn(input_layernorm(h))                                                               */
/*   h = post_attention_conv(h)                  # (conv(x)+x)/sqrt(k+1)                        */
/*   boundary = (il % block_size == 0)                                                          */
/*   if boundary: block_states += block_residual      # residual banked, NOT added               */
/*   else:        residual = h = block_residual + h                                            */
/*   h = mlp_input(block_states, h)             # bare rmsnorm when block_states is empty        */
/*   h = moe(post_attention_layernorm(h))                                                       */
/*   h = post_mlp_conv(h)                                                                       */
/*   return residual + h                                                                        */
/*                                                                                              */
/* For a single token the attention reduces to one key plus the sink, so the output is exactly  */
/*   v * sigmoid(score - sink_logit)                                                            */
/* which is used as the instrument: the sink must dampen the output and the weights must not     */
/* silently renormalise to 1.                                                                   */
/* ------------------------------------------------------------------------------------------- */

void in2_matvec_row_major(const float *W, const float *x, int n_in, int n_out, float *y);

/* softmax weights over keys [0, n_key) plus a sink logit; writes alpha[n_key]. */
static void in2_softmax_sink(const float *logits, int n_key, float sink, float *alpha)
{
    float mx = sink;
    for (int i = 0; i < n_key; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = exp((double)(sink - mx));
    for (int i = 0; i < n_key; ++i) { alpha[i] = expf(logits[i] - mx); sum += alpha[i]; }
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n_key; ++i) alpha[i] *= inv;   /* note: sum(alpha) < 1 by the sink share */
}

int in2_layer_sliding(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                      int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                      const float *msgs)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NH = cfg->n_head, NKV = cfg->n_kv_head, HD = cfg->head_dim;
    const float eps = cfg->rms_eps;

    const int DBG = getenv("IN2_DEBUG") != NULL;
    /* Timer origin for the phase split. It MUST be set at function entry: it was previously set further down,
     * after the first TKPH had already fired, so the "mix" phase was timing the PREVIOUS layer's uninstrumented
     * tail and reporting 81% of layer time for code that measured 0.000 s when instrumented directly. */
    _tk = clock(); _tkh = in2_now();
    float *an = in2_tensor_to_f32(ctx, &L->attn_norm, NULL);
    if (!an) { if (DBG) fprintf(stderr, "[L%d] attn_norm failed\n", il); return WUBU_INKLING2_ERR_IO; }
    if (DBG) fprintf(stderr, "[L%d] step1 ok\n", il);

    /* 0. attention_input: the depth mix that runs on every layer above 0 (blk.0 has no attn_res_query).
     *    The layer was previously norming the raw hidden state and skipping this entirely, which is why
     *    the sliding and full layers disagreed about what they were looking at. */
    const clock_t _al0 = clock();
    float *mixh = (float *)calloc((size_t)n_tok * d, sizeof(float));
    in2_alloc_sec += (double)(clock() - _al0) / (double)CLOCKS_PER_SEC;
    if (!mixh) { free(an); return WUBU_INKLING2_ERR_IO; }
    const clock_t _a0 = clock();
    /* Borrowed rather than copied: in2_mix takes const float*, so the dequant cache's own buffer is usable
     * directly and the pointer must not be freed here. MEASURED SIZE: this tensor is 2304 floats (9 KB), not
     * the 2304x2304 matrix an earlier comment claimed - reading the GGUF header shows dims=[2304] type=0 for
     * blk.N.attn_res_query.weight. The change is therefore neutral in time (~3.00 vs 3.03 tok/s); it is kept
     * only because it removes the per-visit allocation. */
    const float *arq = in2_tensor_borrow(ctx, &L->attn_res_q);
    in2_arq_sec += (double)(clock() - _a0) / (double)CLOCKS_PER_SEC; in2_arq_calls++;
    if (arq && il > 0) {
        /* At a boundary the reference banks this layer's input between the two mixes, so the
         * attention_input mix must exclude the entry the caller has already appended, while the
         * mlp_input mix below must include it. */
        const int n_attn = (cfg->attn_res_block > 0 && il % cfg->attn_res_block == 0)
                         ? n_blocks - 1 : n_blocks;
        for (int t = 0; t < n_tok; ++t) {
            const clock_t _m0 = clock();
            in2_mix(blocks, n_attn, n_tok * d, t * d, h + (size_t)t * d, arq, 1, d, eps,
                    mixh + (size_t)t * d);
            in2_mixfn_sec += (double)(clock() - _m0) / (double)CLOCKS_PER_SEC;
        }
    } else {
        memcpy(mixh, h, sizeof(float) * (size_t)n_tok * d);
    }
    if ((il == 1 || il == 5) && getenv("IN2_DUMP_FULLMIX")) {
        /* blk.5 is the first full layer: its mix takes THREE queries, a path blk.1 never exercises */
        char mp[512]; snprintf(mp, sizeof mp, "%s/L%02d_mix.bin", getenv("IN2_DUMP_FULLMIX"), il);
        FILE *mf = fopen(mp, "wb");
        if (mf) { fwrite(mixh, sizeof(float), (size_t)n_tok * d * 3, mf); fclose(mf); }
    }
    if (il == 1 && getenv("IN2_DUMP_MIX")) {
        FILE *mf = fopen(getenv("IN2_DUMP_MIX"), "wb");
        if (mf) { fwrite(mixh, sizeof(float), (size_t)n_tok * d, mf); fclose(mf); }
    }
    if (il == cfg->loop_core_start && getenv("IN2_DUMP_CTRL")) {
        /* blk.6's four ctrl triples, so the gate formula can be solved against the reference numerically */
        static const char *_nm[4] = { "attn", "k", "v", "mlp" };
        (void)_nm;
        char _p[512];
        for (int s = 0; s < 4; ++s) {
            static const char *part[3] = { "down", "forget", "gain" };
            for (int q = 0; q < 3; ++q) {
                wubu_in2_tensor_t *tt = (q == 0) ? &((wubu_in2_layer_t *)L)->loop_down[s]
                                      : (q == 1) ? &((wubu_in2_layer_t *)L)->loop_forget[s]
                                                 : &((wubu_in2_layer_t *)L)->loop_gain[s];
                int64_t nn = 0;
                float *vv = in2_tensor_to_f32(ctx, tt, &nn);
                if (!vv) continue;
                snprintf(_p, sizeof _p, "%s/ctrl_%d_%s.bin", getenv("IN2_DUMP_CTRL"), s, part[q]);
                FILE *_f = fopen(_p, "wb");
                if (_f) { fwrite(vv, sizeof(float), (size_t)nn, _f); fclose(_f); }
                free(vv);
            }
        }
        /* the k/v short-conv kernels, to test whether the conv state is zeroed per pass */
        {
            float *kc = in2_tensor_to_f32(ctx, &((wubu_in2_layer_t *)L)->k_conv, NULL);
            float *vc = in2_tensor_to_f32(ctx, &((wubu_in2_layer_t *)L)->v_conv, NULL);
            if (kc) { FILE *_f = fopen("C:/Users/eman5/AppData/Local/Temp/ctrl_kconv.bin", "wb");
                      if (_f) { fwrite(kc, sizeof(float), 4u * 1024u, _f); fclose(_f); } free(kc); }
            if (vc) { FILE *_f = fopen("C:/Users/eman5/AppData/Local/Temp/ctrl_vconv.bin", "wb");
                      if (_f) { fwrite(vc, sizeof(float), 4u * 1024u, _f); fclose(_f); } free(vc); }
        }
    }

    if (il == cfg->loop_core_start && getenv("IN2_DUMP_MSG2")) {
        /* the true depth-mix output, before any gate overwrites mixh (slot 3 reuses the buffer) */
        static int _vm = 0;
        char _p[512];
        snprintf(_p, sizeof _p, "%s/mix_v%d.bin", getenv("IN2_DUMP_MSG2"), _vm++);
        FILE *_f = fopen(_p, "wb");
        if (_f) { fwrite(mixh, sizeof(float), (size_t)n_tok * d, _f); fclose(_f); }
    }

    TKPH(0);
    /* 1. attn_norm applied to the mixed state. In passes after the first, q/k/v each get their own
     *    message-gated input and the same norm is applied to all three (the reference uses one
     *    input_layernorm for sliding layers, unlike the full layers' three). */
    float *xn = (float *)calloc((size_t)n_tok * d, sizeof(float));
    float *xkk = (float *)calloc((size_t)n_tok * d, sizeof(float));
    float *xvv = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (msgs) {
        float *g_k = getenv("IN2_DUMP_MSG2") ? (float *)calloc((size_t)n_tok * d, sizeof(float)) : NULL;
        for (int s = 0; s < 3; ++s) {   /* slot 3 (mlp) is gated later, on the MLP depth mix, not on mixh */
            /* Borrowed: read-only message-gate weights, previously copied and freed per visit. */
            const float *dw = in2_tensor_borrow(ctx, &L->loop_down[s]);
            const float *fw = in2_tensor_borrow(ctx, &L->loop_forget[s]);
            const float *gw = in2_tensor_borrow(ctx, &L->loop_gain[s]);
            float *dst = (s == 0) ? xn : (s == 1 ? xkk : xvv);
            for (int t = 0; t < n_tok; ++t) {
                float *gated = (float *)calloc((size_t)d, sizeof(float));
                if (dw && fw && gw) in2_message_gate(mixh + (size_t)t * d, msgs + ((size_t)t * 4 + s) * d, dw, fw, gw,
                                                     d, cfg->loop_gate_rank, eps, gated);
                else                memcpy(gated, mixh + (size_t)t * d, sizeof(float) * d);
                if (g_k && s == 1) memcpy(g_k + (size_t)t * d, gated, sizeof(float) * d);
                in2_rmsnorm(gated, an, d, eps, dst + (size_t)t * d);
                free(gated);
            }
            
        }
        if (g_k && il == cfg->loop_core_start) {
            static int _vg = 0;
            char _p[512];
            snprintf(_p, sizeof _p, "%s/gate_k_v%d.bin", getenv("IN2_DUMP_MSG2"), _vg++);
            FILE *_f = fopen(_p, "wb");
            if (_f) { fwrite(g_k, sizeof(float), (size_t)n_tok * d, _f); fclose(_f); }
        }
        free(g_k);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            in2_rmsnorm(mixh + (size_t)t * d, an, d, eps, xn + (size_t)t * d);
            memcpy(xkk + (size_t)t * d, xn + (size_t)t * d, sizeof(float) * d);
            memcpy(xvv + (size_t)t * d, xn + (size_t)t * d, sizeof(float) * d);
        }
    }

    /* 2. q: project, scale+clip per head. */
    const int qdim = NH * HD, kdim = NKV * HD;
    if (il == cfg->loop_core_start && getenv("IN2_DUMP_MSG2")) {
        /* one file per visit of the first core layer: visit 2 is loop pass 2 (the first pass with messages) */
        static int _v6 = 0;
        char _p[512];
        snprintf(_p, sizeof _p, "%s/il6_v%d.bin", getenv("IN2_DUMP_MSG2"), _v6++);
        FILE *_f = fopen(_p, "wb");
        if (_f) {
            fwrite(mixh, sizeof(float), (size_t)n_tok * d, _f);                 /* the 1-query depth mix */
            if (msgs) fwrite(msgs, sizeof(float), (size_t)n_tok * 4 * d, _f);   /* 4 role messages per token */
            fwrite(xn, sizeof(float), (size_t)n_tok * d, _f);                   /* gated+normed q input */
            fwrite(xkk, sizeof(float), (size_t)n_tok * d, _f);                  /* gated+normed k input */
            fwrite(xvv, sizeof(float), (size_t)n_tok * d, _f);                  /* gated+normed v input */
            fclose(_f);
        }
    }
    if (!g_kv && in2_kv_enabled()) {
        /* Real sizing. Sliding layers never look past EXT keys, so the ring holds EXT rows per visit:
         * 72 x 1024 x 1024 x 4 B x 2 = 604 MB, which is the honest cost of a 1024-token window on this model.
         * IN2_KV_POS lowers it for short-context work. */
        static in2_kv_t boot;
        const char *pz = getenv("IN2_KV_POS");
        const int ctx  = pz ? atoi(pz) : 1024;
        /* PER-VISIT CAPACITY. IN2_KV_TRACE puts full attention at visits 5,11,17,23,29,35,41,47 only - every other
         * visit is sliding and can never read past cfg->swa_window (1024 in this model). Allocating `ctx` rows for all
         * 72 visits, as this line used to, is what made a long context unaffordable: 72 x 1M x 1024 x 4 B x 2 = 603 GB
         * for a 1M context. Sized per visit the same context is 8 x 1M x 1024 x 8 B = 68.7 GB in f32, and the 64
         * sliding visits stay at 1024 rows forever (0.5 GB). The layer rule is the same one is_full_attn_layer uses,
         * restated here because the cache is built before the model is loaded. */
        const int swa = cfg->swa_window > 0 ? cfg->swa_window : 1024;
        const int rb  = cfg->attn_res_block > 1 ? cfg->attn_res_block : 6;
        int *rows = (int *)malloc(72 * sizeof(int));
        if (rows) {
            for (int v = 0; v < 72; ++v)
                rows[v] = (v < 48 && (v % rb) == (rb - 1)) ? ctx : (swa < ctx ? swa : ctx);
            if (in2_kv_init_pervisit(&boot, 72, kdim, swa, rows) == 0) {
                g_kv = &boot; g_kv_window = swa;
                fprintf(stderr, "KV    ctx=%d rows/visit: %d full x %d + %d sliding x %d = %.1f MB\n",
                        ctx, 8, ctx, 64, swa, (double) in2_kv_bytes(&boot) / 1048576.0);
            }
            free(rows);
        }
    }
    if (!g_kv2 && in2_kv_enabled()) {
        /* The residual and MoE convs only ever need the last three rows, so this record is a 3-row window per
         * visit rather than a position ring: 72 x 3 x 2304 x 4 B x 2 = 4 MB, against 1.36 GB if it were sized
         * like the K/V cache. */
        static in2_kv_t boot2;
        if (in2_kv_init(&boot2, 72, d, 3, 3) == 0) g_kv2 = &boot2;
    }
    /* pinned for the whole layer: the store advances g_visit at the end, so attention must not re-read it or it
     * would attend to the next visit's empty rows */
    const int vvis = g_visit;
    if (il == 6 && getenv("IN2_VTRACE")) fprintf(stderr, "VISIT il=6 vvis=%d pos0=%d ntok=%d\n", vvis, pos0, n_tok);
    float *qcl = in2_tensor_to_f32(ctx, &L->q_clip, NULL);
    float *q = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    if (!q) { if (DBG) fprintf(stderr, "[L%d] q alloc failed\n", il);
        free(an); free(xn); free(qcl); free(q); return WUBU_INKLING2_ERR_IO; }
    if (DBG) fprintf(stderr, "[L%d] step2 ok (qcl=%p)\n", il, (void*)qcl);
    if (in2_matvec_nt(ctx, &L->wq, xn, n_tok, d, qdim, q) != WUBU_INKLING2_OK) {
        free(an); free(xn); free(qcl); free(q); return WUBU_INKLING2_ERR_IO;
    }
    for (int t = 0; t < n_tok; ++t)
        in2_l2_clip(q + (size_t)t * qdim, qcl, NH, HD, 22.6274166f);
    if (il == 1 && getenv("IN2_DUMP_Q")) {
        FILE *qf = fopen(getenv("IN2_DUMP_Q"), "wb");
        if (qf) {
            fwrite(xn, sizeof(float), (size_t)n_tok * d, qf);        /* attn-normed input */
            fwrite(q,  sizeof(float), (size_t)n_tok * qdim, qf);     /* post-clip q */
            fclose(qf);
        }
    }

    TKPH(1);
    /* 3. k, v: project, static short conv, then bare L2 clip (k) / unweighted rms (v). */
    /* Wk/Wv used to be materialised here as full float matrices and then never read - the projections call
     * in2_matvec_nt on the tensors directly. At 1024x2304 that is 9.4 MB x 2 per layer visit, ~750 MB of dead
     * memcpy per decode token (and, since neither was in the free list, a leak of the same size). */
    float *kc = in2_tensor_to_f32(ctx, &L->k_conv, NULL);
    float *vc = in2_tensor_to_f32(ctx, &L->v_conv, NULL);
    float *kraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vraw = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *kt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *vt = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *k = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    float *v = (float *)calloc((size_t)n_tok * kdim, sizeof(float));
    int rc = WUBU_INKLING2_OK;
    if (!kc || !vc || !kraw || !vraw || !kt || !vt || !k || !v) rc = WUBU_INKLING2_ERR_IO;
    extern double in2_wall_now(void);
    const double _kp0 = in2_wall_now();
    if (rc == WUBU_INKLING2_OK) {
        if (in2_matvec_nt(ctx, &L->wk, xkk, n_tok, d, kdim, kraw) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
        else if (in2_matvec_nt(ctx, &L->wv, xvv, n_tok, d, kdim, vraw) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
    }
    in2_kvproj_sec += in2_wall_now() - _kp0;
    const double _kc0 = in2_wall_now();
    if (rc == WUBU_INKLING2_OK) {
        if (g_kv) {
            /* Cache mode: conv each row against the cached raw neighbours and push the raw row immediately, so
             * the next row in this same batch sees it. The batch conv cannot do this - it only knows the rows it
             * was handed, so a single new position would pad the first taps with zeros. */
            for (int t = 0; t < n_tok; ++t) {
                in2_kv_conv_row(in2_kv_hist_k(g_kv, vvis, 3), kraw + (size_t)t * kdim, kdim, kc, 4, kt + (size_t)t * kdim);
                in2_kv_conv_row(in2_kv_hist_v(g_kv, vvis, 3), vraw + (size_t)t * kdim, kdim, vc, 4, vt + (size_t)t * kdim);
                in2_kv_store_raw(g_kv, vvis, kraw + (size_t)t * kdim, vraw + (size_t)t * kdim);
            }
        } else {
            in2_shortconv(kraw, n_tok, kdim, kc, 4, kt);
        }
        if (il == 1 && getenv("IN2_DUMP_KV")) {
            /* kraw then post-conv k, so the conv formula can be checked in isolation from the matvec */
            FILE *kf = fopen(getenv("IN2_DUMP_KV"), "wb");
            if (kf) {
                fwrite(kraw, sizeof(float), (size_t)n_tok * kdim, kf);
                fwrite(kt,   sizeof(float), (size_t)n_tok * kdim, kf);
                fwrite(kc,   sizeof(float), 4 * (size_t)kdim, kf);
                fclose(kf);
            }
        }
        if (!g_kv) in2_shortconv(vraw, n_tok, kdim, vc, 4, vt);
        if (il == 1 && getenv("IN2_DUMP_V")) {
            FILE *vf = fopen(getenv("IN2_DUMP_V"), "wb");
            if (vf) { fwrite(vraw, sizeof(float), (size_t)n_tok*kdim, vf);
                      fwrite(vt,   sizeof(float), (size_t)n_tok*kdim, vf);
                      fwrite(vc,   sizeof(float), 4*(size_t)kdim, vf);
                      fclose(vf); }
        }
        for (int t = 0; t < n_tok; ++t) {
            memcpy(k + (size_t)t * kdim, kt + (size_t)t * kdim, sizeof(float) * kdim);
            in2_l2_clip(k + (size_t)t * kdim, NULL, NKV, HD, 22.6274166f);
            if (g_kv) {
                const int gp = g_kv_pos + t, vis = vvis;
                float *ck = in2_kv_k(g_kv, vis, gp);
                if (ck) memcpy(ck, k + (size_t)t * kdim, sizeof(float) * (size_t)kdim);
            }
            memcpy(v + (size_t)t * kdim, vt + (size_t)t * kdim, sizeof(float) * kdim);
            float *row = v + (size_t)t * kdim;
            for (int hh = 0; hh < NKV; ++hh) in2_rmsnorm_unweighted(row + hh * HD, HD, eps, row + hh * HD);
            if (g_kv) {
                float *cv = in2_kv_v(g_kv, vvis, g_kv_pos + t);
                if (cv) memcpy(cv, row, sizeof(float) * (size_t)kdim);
            }
        }
        if (g_kv && getenv("IN2_KV_VERIFY")) {
            /* read back what was just stored and compare with the live rows: this is an index test, not a numerics
             * test - the cache is written from these same bytes, so anything nonzero here means the slot mapping
             * or the visit identity is wrong, which is the failure a later read path would silently inherit. */
            float worst = 0.0f;
            for (int t = 0; t < n_tok; ++t) {
                const float *ck = in2_kv_k(g_kv, vvis, g_kv_pos + t);
                const float *cv = in2_kv_v(g_kv, vvis, g_kv_pos + t);
                if (!ck || !cv) { worst = 1e30f; break; }
                for (int e = 0; e < kdim; ++e) {
                    float e1 = fabsf(ck[e] - k[(size_t)t * kdim + e]);
                    float e2 = fabsf(cv[e] - v[(size_t)t * kdim + e]);
                    if (e1 > worst) worst = e1;
                    if (e2 > worst) worst = e2;
                }
            }
            if (worst > 1e-6f) fprintf(stderr, "KV STORE MISMATCH L%d visit=%d worst=%.3e\n", il, vvis, worst);
            else if (il == 0 && n_tok > 0) fprintf(stderr, "KV store ok L%d visit=%d n_tok=%d worst=%.3e\n", il, vvis, n_tok, worst);
        }
        if (g_kv) {
            if (g_kv->n_stored < g_kv_pos + n_tok) g_kv->n_stored = g_kv_pos + n_tok;
            ++g_visit;
        }
    }

    in2_kvconv_sec += in2_wall_now() - _kc0;
    TKPH(2);
    /* 4. relative bias table from the normed input, plus sinks. */
    float *rcl = in2_tensor_to_f32(ctx, &L->rel_clip, NULL);
    float *P = in2_tensor_to_f32(ctx, &L->rel_bias, NULL);
    const int DREL = 16, EXT = cfg->swa_window;
    /* Relative-position vectors are kept for every token instead of being recomputed inside the query loop:
     * the old code called this same projection once in the pre-pass and AGAIN for each query, n_tok times over,
     * on weights that do not depend on the query. */
    float *rs_all = (float *)calloc((size_t)n_tok * NH * DREL, sizeof(float));
    float *bias = (float *)calloc((size_t)NH * EXT, sizeof(float));
    float *sink = (float *)calloc((size_t)n_tok * NH, sizeof(float));
    if (rcl && P && rs_all && bias && sink && rc == WUBU_INKLING2_OK) {
        if (in2_matvec_nt(ctx, &L->wrel, xn, n_tok, d, NH * DREL, rs_all) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
        else if (in2_matvec_nt(ctx, &L->sink, xn, n_tok, d, NH, sink) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
        for (int t = 0; t < n_tok; ++t) in2_l2_clip(rs_all + (size_t)t * NH * DREL, rcl, NH, DREL, 8.0f);
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] rel/sink failed rcl=%p P=%p\n", il, (void*)rcl, (void*)P); }
    if (DBG) fprintf(stderr, "[L%d] step4 rc=%d\n", il, rc);

    TKPH(3);
    /* 5. attention per head. The bias table is token-dependent, so it is recomputed per token; the
     *    loop below therefore walks queries and rebuilds only the distances it needs. */
    float *attn_out = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    /* sized for the visible window, not the batch: with a cache n_key = min(pos0+t+1, EXT) and a 1-token step at
     * pos0 = 3 writes four entries into a one-element buffer (that is the silent death on step 1). */
    float *alpha = (float *)calloc((size_t)n_tok + (size_t)pos0 + 1u, sizeof(float));
    float *logits = (float *)calloc((size_t)n_tok + (size_t)pos0 + 1u, sizeof(float));
    if (attn_out && alpha && logits && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            const int kvh_stride = NH / NKV;
            /* bias table from the already-clipped per-token relative vector */
            in2_rel_bias_table(rs_all + (size_t)t * NH * DREL, P, NH, DREL, EXT, bias);
            for (int hh = 0; hh < NH; ++hh) {
                const int kvh = hh / kvh_stride;
                /* Positions are addressed globally so the same loop serves a full recompute (pos0 = 0, every key
                 * local) and an incremental step (pos0 = already-cached prefix, keys read from the cache). With no
                 * cache attached the lookup is the local array and the arithmetic is bit-identical to before. */
                const int gp = pos0 + t;
                if (il == 6 && getenv("IN2_KROW") && t == 0 && hh == 0) {
                    const float *sk = g_kv ? in2_kv_k(g_kv, vvis, 0) : k;
                    const float *sv = g_kv ? in2_kv_v(g_kv, vvis, 0) : v;
                    fprintf(stderr, "ATT row0 mode=%s pos0=%d nkey=%d k=[%.8g %.8g] v=[%.8g %.8g]\n",
                            g_kv ? "cache" : "local", pos0, gp + 1,
                            sk ? sk[0] : 0.f, sk ? sk[1] : 0.f, sv ? sv[0] : 0.f, sv ? sv[1] : 0.f);
                }
                const int j0 = (gp + 1 > EXT) ? (gp + 1 - EXT) : 0;
                int n_key = 0;
                for (int j = j0; j <= gp; ++j) {
                    const int dist = gp - j;
                    const float *qa = q + (size_t)t * qdim + hh * HD;
                    const float *kb;
                    if (g_kv) {
                        kb = in2_kv_k(g_kv, vvis, j);
                        if (!kb) continue;
                        kb += kvh * HD;
                    } else {
                        kb = k + (size_t)(j - pos0) * kdim + kvh * HD;
                    }
                    logits[n_key] = in2_attn_dot(qa, kb, HD) * (1.0f / sqrtf((float)HD)) + bias[hh * EXT + dist];
                    ++n_key;
                }
                in2_softmax_sink(logits, n_key, sink[(size_t)t * NH + hh], alpha);
                float *o = attn_out + (size_t)t * qdim + hh * HD;
                for (int j = 0; j < n_key; ++j) {
                    const float a = alpha[j];
                    const float *vb;
                    if (g_kv) {
                        vb = in2_kv_v(g_kv, vvis, j0 + j);
                        if (!vb) continue;
                        vb += kvh * HD;
                    } else {
                        vb = v + (size_t)(j0 + j - pos0) * kdim + kvh * HD;
                    }
                    in2_attn_accum(o, vb, a, HD);
                }
            }
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step5 failed attn_out=%p alpha=%p logits=%p\n", il, (void*)attn_out, (void*)alpha, (void*)logits); }
    if (DBG) fprintf(stderr, "[L%d] step5 rc=%d\n", il, rc);
    if (il == 1 && getenv("IN2_DUMP_ATTN")) {
        FILE *af = fopen(getenv("IN2_DUMP_ATTN"), "wb");
        if (af) {
            fwrite(sink, sizeof(float), (size_t)n_tok * NH, af);
            for (int hh2 = 0; hh2 < NH; ++hh2) fwrite(&bias[hh2 * EXT], sizeof(float), 1, af);
            fwrite(attn_out, sizeof(float), (size_t)n_tok * qdim, af);
            fclose(af);
        }
    }

    TKPH(4);
    /* 6. gated output projection into its own buffer: h still holds the layer input, which the
     *    residual needs. sigmoid(attn_gate(xn)) * attn_out, then attn_output. */
    float *gate = (float *)calloc((size_t)n_tok * qdim, sizeof(float));
    float *ao = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (gate && ao && rc == WUBU_INKLING2_OK) {
        /* gated output projection: batch the gate across positions, apply sigmoid*attn_out elementwise, then batch
         * attn_output over the gated buffer. attn_output is (qdim, d), so out[o] = sum_u W[o*qdim+u] * gate[u]. */
        if (in2_matvec_nt(ctx, &L->wgate, xn, n_tok, d, qdim, gate) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
        else {
            for (int t = 0; t < n_tok; ++t)
                for (int i = 0; i < qdim; ++i) {
                    const float gv = 1.0f / (1.0f + expf(-gate[(size_t)t * qdim + i]));
                    gate[(size_t)t * qdim + i] = attn_out[(size_t)t * qdim + i] * gv;
                }
            if (in2_matvec_nt(ctx, &L->wo, gate, n_tok, qdim, d, ao) != WUBU_INKLING2_OK) rc = WUBU_INKLING2_ERR_IO;
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step6 failed gate=%p ao=%p\n", il, (void*)gate, (void*)ao); }
    if (DBG) fprintf(stderr, "[L%d] step6 rc=%d\n", il, rc);

    if (getenv("IN2_DUMPX")) {
        static int _visits[64];
        const int _visit = (il >= 0 && il < 64) ? _visits[il]++ : 0;
        /* Layer-scoped: this site is inside the per-layer function, so an unscoped fixed path
         * silently keeps the LAST layer's buffers (the instrument lied once already). */
        const char *_dl = getenv("IN2_DUMPX_L");
        const int _want = _dl ? atoi(_dl) : -1;
        if (_want < 0 || _want == il) {
            char p[512];
            #define IN2_DX(NAME, PTR, N) do {                                       \
                snprintf(p, sizeof p, "C:/Users/eman5/AppData/Local/Temp/d_%s_l%d_v%d.bin", (NAME), il, _visit); \
                FILE *_f = fopen(p, "wb");                                          \
                if (_f) { fwrite((PTR), sizeof(float), (size_t)(N), _f); fclose(_f); } \
            } while (0)
            IN2_DX("xn", xn, n_tok * d);
            IN2_DX("q", q, n_tok * qdim);
            IN2_DX("k", k, n_tok * kdim);
            IN2_DX("v", v, n_tok * kdim);
            IN2_DX("attn", attn_out, n_tok * qdim);
            IN2_DX("ao", ao, n_tok * d);
            IN2_DX("gate", gate, n_tok * qdim);
            IN2_DX("bias", bias, NH * EXT);
            IN2_DX("rs", rs_all, n_tok * NH * DREL);
            IN2_DX("sink", sink, n_tok * NH);
            IN2_DX("alpha", alpha, n_tok + pos0 + 1);
            IN2_DX("logits", logits, n_tok + pos0 + 1);
            IN2_DX("kraw", kraw, n_tok * kdim);
            IN2_DX("kt", kt, n_tok * kdim);
            IN2_DX("vraw", vraw, n_tok * kdim);
            IN2_DX("vt", vt, n_tok * kdim);
            IN2_DX("kc", kc, 4 * kdim);
            IN2_DX("vc", vc, 4 * kdim);
            #undef IN2_DX
            if (_want == 0 && getenv("IN2_DUMP_W")) {
                float *W = in2_tensor_to_f32(ctx, &L->wo, NULL);
                FILE *_f = fopen("C:/Users/eman5/AppData/Local/Temp/d_wo_l0.bin", "wb");
                if (W && _f) {
                    fwrite(W, sizeof(float), (size_t)d * qdim, _f);
                    fprintf(stderr, "dumped wo layer0 %d x %d\n", d, qdim);
                }
                if (_f) fclose(_f);
                free(W);
            }
            fprintf(stderr, "dumped layer %d vectors (xn/q/k/v/attn/ao/gate)\n", il);
        }
    }

    TKPH(5);
    /* 7. post-attention short conv, then the residual rule:
     *    boundary (il % block_size == 0): the attention output becomes the residual and block_residual
     *    is banked into block_states (a no-op here, single block);
     *    otherwise: residual = block_residual + attention output. */
    float *pc = in2_tensor_to_f32(ctx, &L->attn_post_conv, NULL);
    float *res = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (pc && res && rc == WUBU_INKLING2_OK) {
        in2_conv_stream(g_kv2, vvis, 0, ao, n_tok, d, pc, 4, res);
        const int boundary = (cfg->attn_res_block > 0) && (il % cfg->attn_res_block == 0);
        for (int t = 0; t < n_tok; ++t)
            for (int i = 0; i < d; ++i) {
                const size_t idx = (size_t)t * d + i;
                if (!boundary) res[idx] = h[idx] + res[idx];
            }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step7 failed pc=%p res=%p\n", il, (void*)pc, (void*)res); }
    if (DBG) fprintf(stderr, "[L%d] step7 rc=%d\n", il, rc);
    if (il == 1 && getenv("IN2_DUMP_AO")) {
        FILE *of = fopen(getenv("IN2_DUMP_AO"), "wb");
        if (of) { fwrite(ao,  sizeof(float), (size_t)n_tok * d, of);
                  fwrite(res, sizeof(float), (size_t)n_tok * d, of); fclose(of); }
    }

    TKPH(6);
    /* 8. mlp_input: the same depth mix, one query, over [block_states; residual]. With no banked blocks
     *    it degenerates to an unweighted RMS norm. */
    float *fn = in2_tensor_to_f32(ctx, &L->ffn_norm, NULL);
    float *moein = (float *)calloc((size_t)n_tok * d, sizeof(float));
    float *moeout = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (fn && moein && moeout && rc == WUBU_INKLING2_OK) {
        for (int t = 0; t < n_tok; ++t) {
            float *tmp = (float *)calloc((size_t)d, sizeof(float));
            float *g2 = (float *)calloc((size_t)d, sizeof(float));
            float *frq = in2_tensor_to_f32(ctx, &L->ffn_res_q, NULL);
            if (frq) in2_mix(blocks, n_blocks, n_tok * d, t * d, res + (size_t)t * d, frq, 1, d, eps, tmp);
            else      in2_rmsnorm_unweighted(res + (size_t)t * d, d, eps, tmp);
            free(frq);
            /* the mlp slot's message gate acts on THIS mix (the reference's node_3652 = [mlp mix, msg]),
             * and its output is normalised with ffn_norm - not attn_norm. */
            if (msgs && g2) {
                const float *dw = in2_tensor_borrow(ctx, &L->loop_down[3]);
                const float *fw = in2_tensor_borrow(ctx, &L->loop_forget[3]);
                const float *gw = in2_tensor_borrow(ctx, &L->loop_gain[3]);
                if (dw && fw && gw) in2_message_gate(tmp, msgs + ((size_t)t * 4 + 3) * d, dw, fw, gw,
                                                     d, cfg->loop_gate_rank, eps, g2);
                else                memcpy(g2, tmp, sizeof(float) * d);
                
                in2_rmsnorm(g2, fn, d, eps, moein + (size_t)t * d);
            } else {
                in2_rmsnorm(tmp, fn, d, eps, moein + (size_t)t * d);
            }
            free(g2);
            free(tmp);
        }
        rc = in2_moe(ctx, m, cfg, il, moein, n_tok, moeout);
        TKPH(7);
        if (il == cfg->loop_core_start && getenv("IN2_DUMP_MSG2")) {
            static int _vr = 0;
            char _p[512];
            const char *_dn = getenv("IN2_DUMP_MSG2");
            snprintf(_p, sizeof _p, "%s/res_v%d.bin", _dn, _vr);
            FILE *_f = fopen(_p, "wb"); if (_f) { fwrite(res, sizeof(float), (size_t)n_tok * d, _f); fclose(_f); }
            snprintf(_p, sizeof _p, "%s/moein_v%d.bin", _dn, _vr);
            _f = fopen(_p, "wb"); if (_f) { fwrite(moein, sizeof(float), (size_t)n_tok * d, _f); fclose(_f); }
            snprintf(_p, sizeof _p, "%s/moeout_v%d.bin", _dn, _vr);
            _f = fopen(_p, "wb"); if (_f) { fwrite(moeout, sizeof(float), (size_t)n_tok * d, _f); fclose(_f); }
            ++_vr;
        }
    } else { rc = WUBU_INKLING2_ERR_IO; if (DBG) fprintf(stderr, "[L%d] step8 failed fn=%p moein=%p moeout=%p (moe rc=%d)\n", il, (void*)fn, (void*)moein, (void*)moeout, rc); }
    if (DBG) fprintf(stderr, "[L%d] step8 rc=%d\n", il, rc);

    /* IN2_DUMPX_LAYER selects one layer; without it every layer overwrites the same files and only the
     * last survives - which is why the MoE had only ever been checked on blk.22, the one layer with no
     * expert .scale. Layers 0..19 carry one and had never been verified. */
    const char *dl = getenv("IN2_DUMPX_LAYER");
    if (getenv("IN2_DUMPX") && n_tok == 1 && (!dl || atoi(dl) == il)) {
        FILE *r1 = fopen("d_res.bin","wb"); if (r1) { fwrite(res,sizeof(float),(size_t)n_tok*d,r1); fclose(r1); }
        FILE *r2 = fopen("d_moein.bin","wb"); if (r2) { fwrite(moein,sizeof(float),(size_t)n_tok*d,r2); fclose(r2); }
        FILE *r3 = fopen("d_moe.bin","wb"); if (r3) { fwrite(moeout,sizeof(float),(size_t)n_tok*d,r3); fclose(r3); }
    }

    /* 9. post-mlp conv, then residual + h. */
    float *pm = in2_tensor_to_f32(ctx, &L->ffn_post_conv, NULL);
    float *mo = (float *)calloc((size_t)n_tok * d, sizeof(float));
    if (pm && mo && rc == WUBU_INKLING2_OK) {
        const clock_t _tl0 = clock();
        in2_conv_stream(g_kv2, vvis, 1, moeout, n_tok, d, pm, 4, mo);
        for (size_t i = 0; i < (size_t)n_tok * d; ++i) h[i] = res[i] + mo[i];
        in2_tail_sec += (double)(clock() - _tl0) / (double)CLOCKS_PER_SEC;
    } else rc = WUBU_INKLING2_ERR_IO;

    free(an); free(xn); free(qcl); free(q);
    free(kc); free(vc); free(kraw); free(vraw); free(kt); free(vt); free(k); free(v);
    free(rcl); free(P); free(rs_all); free(bias); free(sink);
    free(attn_out); free(alpha); free(logits); free(gate); free(ao);
    free(pc); free(res); free(fn); free(moein); free(moeout); free(pm); free(mo); free(mixh);
    free(xkk); free(xvv);
    return rc;
}

/* Parallel row-split of the quant matvec.
 *
 * Splitting the OUTPUT ROWS is race-free by construction (each row of y is written by exactly one chunk) and needs
 * no per-thread scratch beyond what the kernel already allocates locally. Row granularity keeps the weight walk
 * contiguous: chunk r reads elements [off + r*n_in, off + (r+chunk)*n_in), so the split is a partition of the same
 * byte range the serial call would have read.
 *
 * IN2_THREADS overrides the count; 1 restores the serial call exactly (same code path, bit-identical output). */
int in2_matvec_quant_mt(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off,
                        const float *x, int n_out, int n_in, float *y)
{
    int nt = 0;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    if (getenv("IN2_THREADS")) nt = atoi(getenv("IN2_THREADS"));
    if (nt <= 1 || n_out < 2 * nt) return in2_matvec_quant(ctx, t, elem_off, x, n_out, n_in, y);

    int chunk = (n_out + nt - 1) / nt;
    chunk = (chunk + 15) & ~15;              /* keep chunks on 16-row boundaries for the row kernels */
    if (chunk < 16) chunk = 16;
    int bad = 0;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nt) schedule(static) reduction(|:bad)
#endif
    for (int r = 0; r < n_out; r += chunk) {
        const int n = (r + chunk <= n_out) ? chunk : (n_out - r);
        if (in2_matvec_quant(ctx, t, elem_off + (int64_t)r * n_in, x, n, n_in, y + r) != WUBU_INKLING2_OK) bad = 1;
    }
    return bad ? WUBU_INKLING2_ERR_IO : WUBU_INKLING2_OK;
}

/* Batched quantised matvec with a slice offset: out[t][r] = W[off + r] . a[t].
 *
 * in2_matvec_nt does the batching but has no element offset, and in2_matvec_quant has the offset but computes one
 * position. The MoE needs both: an expert is a slab at e * rows * n_in, applied to a bucket of tokens. The offset is
 * converted to bytes with gguf_raw_size, the same way in2_matvec_quant does, and the read goes straight from the
 * mapping when there is one.
 *
 * Deliberately NOT internally threaded (no in2_rows_parallel, no omp here): the caller parallelises over experts, and
 * two nested regions per expert is the one thing we measured to be strictly worse on mingw libgomp.
 */
int in2_matvec_quant_nt(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t elem_off,
                        const float *X, int nt, int n_out, int n_in, float *Y)
{
    if (n_in <= 0 || n_out <= 0 || nt <= 0) return WUBU_INKLING2_ERR_IO;
    const int ty = (t && t->info) ? t->info->ggml_type : -1;
    const int q8_0_act = (ty == 8 || ty == 40 || ty == 42);
    const int be = q8_0_act ? 32 : 256;

    /* One line per distinct type, so a tensor that silently takes the float path can be told apart from one whose
     * kernels simply are not verified: both look identical from the outside (same tokens, same silence). */
    if (getenv("IN2_NT_TRACE") && ty >= 0 && ty < 256) {
        static int seen[256];
        if (!seen[ty]) {
            seen[ty] = 1;
            fprintf(stderr, "NT ty=%d sup=%d ver=%d align=%d n_in=%d n_out=%d nt=%d (%s)\n",
                    ty, wubu_iq_supported(ty), in2_ty_verified(ty), (n_in % be) == 0, n_in, n_out, nt,
                    (t && t->info && t->info->name[0]) ? t->info->name : "?");
        }
    }
    if (getenv("IN2_NOFAST") == NULL && t && t->info && wubu_iq_supported(ty) && in2_ty_verified(ty) &&
        (n_in % be) == 0) {
        const int64_t rb = gguf_raw_size(ty, n_in);
        const int64_t boff = (elem_off == 0) ? 0 : gguf_raw_size(ty, elem_off);
        if (rb > 0 && boff >= 0) {
            wubu_in2_tensor_t sh = *t;
            sh.file_offset = t->file_offset + boff;
            const unsigned char *msrc = wubu_inkling2_map_ptr(ctx, &sh);
            unsigned char *whole = msrc ? NULL : (unsigned char *) malloc((size_t) rb * (size_t) n_out);
            const int have = msrc ? 1 : (whole && wubu_inkling2_read(ctx, &sh, whole, rb * n_out) == WUBU_INKLING2_OK);
            if (have) {
                const unsigned char *src = msrc ? msrc : whole;
                const size_t ablk = q8_0_act ? sizeof(in2_q8_0_blk) : sizeof(in2_q8k_blk);
                const int nab = n_in / be;
                void *act = malloc(ablk * (size_t) nab * (size_t) nt);
                if (act) {
                    for (int i = 0; i < nt; ++i) {
                        void *a = (unsigned char *) act + ablk * (size_t) nab * (size_t) i;
                        if (q8_0_act) in2_q8_0_quantise(X + (size_t) i * n_in, nab, (in2_q8_0_blk *) a);
                        else          in2_q8k_quantise(X + (size_t) i * n_in, nab, (in2_q8k_blk *) a);
                    }
                    const clock_t _q0 = clock();
                    for (int g = 0; g < nt; g += 8) {
                        const int nn = (nt - g < 8) ? (nt - g) : 8;
                        const void *ap[8];
                        float *op[8];
                        for (int i = 0; i < nn; ++i) {
                            ap[i] = (unsigned char *) act + ablk * (size_t) nab * (size_t) (g + i);
                            op[i] = Y + (size_t) (g + i) * n_out;
                        }
                        /* The attention projections arrive here, not through in2_matvec_quant, so the resident
                         * tier needs its own hook at this call site too. Only nn==1 (a decode step) is taken:
                         * the resident entry runs the single-token kernel, and a prefill batch would need the
                         * multi-token one. */
                        int gpu_took = 0;
                        if (nn == 1 && getenv("IN2_GPU_DENSE") && t && t->info && t->info->name[0] &&
                            strstr(t->info->name, "exps") == NULL) {
                            in2_gpu_init();
                            if (in2_gpu_state == 1) {
                                const int ab = (int) (ablk * (size_t) nab);
                                if (in2_gpu_resident(ty, src, (size_t) rb * (size_t) n_out, ap[0], ab,
                                                     op[0], n_in, n_out, (int64_t) rb) == 0) {
                                    gpu_took = 1;
                                    if (in2_gpu_calls == 0)
                                        fprintf(stderr, "GPU tier: resident dense matvec engaged (%s)\n",
                                                t->info->name);
                                    in2_gpu_calls += 1;
                                }
                            }
                        }
                        if (!gpu_took) wubu_iq_rows(ty, src, (size_t) rb, n_in, ap, nn, op, 0, n_out);
                    }
                    if (ty >= 0 && ty < 256) {
                        in2_ty_sec[ty] += (double) (clock() - _q0) / (double) CLOCKS_PER_SEC;
                        in2_ty_calls[ty] += 1;
                        in2_ty_macs[ty] += (double) nt * (double) n_out * (double) n_in;
                    }
                    free(act);
                    free(whole);
                    return WUBU_INKLING2_OK;
                }
            }
            free(whole);
        }
    }

    /* Same answer, one position at a time. */
    for (int i = 0; i < nt; ++i)
        if (in2_matvec_quant(ctx, t, elem_off, X + (size_t) i * n_in, n_out, n_in, Y + (size_t) i * n_out)
            != WUBU_INKLING2_OK)
            return WUBU_INKLING2_ERR_IO;
    return WUBU_INKLING2_OK;
}

/* Expert-major MoE for multi-token forwards. See the call site in in2_moe for why this exists. */
static int in2_moe_batch(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                         int il, const float *x, int n_tok, float *out,
                         const float *Wr, const float *cb, const float *gs, const float *ds)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NE = cfg->n_expert, TOPK = cfg->n_expert_used, DFF = cfg->d_ff_expert;
    /* Chunk the tokens so the per-(token,slot) expert outputs stay bounded - a 1M prefill cannot allocate
     * n_tok * TOPK * d at once (74 GB), so chunking is not optional. But the chunk IS the win: with 128 experts and
     * 8-of-128 routing, a chunk of 32 tokens already selects almost every expert, so the slab is read 128 times per
     * chunk either way and a small chunk only halves the reads (measured 47.5 -> 38.4 s at 320 tokens, 19%). At
     * CH = 512 the same 320 tokens fit in one chunk, every slab is read once, and the amortisation is real:
     * 512 x 8 x 2304 floats = 37.7 MB of per-(token,slot) output, which is a fair trade. */
    const int CH = n_tok < 512 ? n_tok : 512;
    int rc = WUBU_INKLING2_OK;

    int *idx = (int *) malloc((size_t) CH * TOPK * sizeof(int));
    float *wt = (float *) malloc((size_t) CH * TOPK * sizeof(float));
    float *ok = (float *) malloc((size_t) CH * TOPK * d * sizeof(float));
    int *cnt = (int *) malloc((size_t) NE * sizeof(int));
    int *start = (int *) malloc(((size_t) NE + 1) * sizeof(int));
    int *fill = (int *) malloc((size_t) NE * sizeof(int));
    int *bt = (int *) malloc((size_t) CH * TOPK * sizeof(int));
    int *bk = (int *) malloc((size_t) CH * TOPK * sizeof(int));
    if (!idx || !wt || !ok || !cnt || !start || !fill || !bt || !bk) {
        rc = WUBU_INKLING2_ERR_IO;
        goto done;
    }

    for (int t0 = 0; t0 < n_tok; t0 += CH) {
        const int nt_ = (n_tok - t0 < CH) ? (n_tok - t0) : CH;

        for (int t = 0; t < nt_; ++t) {
            int *ip = idx + (size_t) t * TOPK;
            float *wp = wt + (size_t) t * TOPK;
            const int got = in2_router(x + (size_t) (t0 + t) * d, Wr, cb, NE, d, TOPK,
                                       cfg->expert_weights_scale, ip, wp);
            for (int k = got; k < TOPK; ++k) { ip[k] = -1; wp[k] = 0.0f; }
            for (int i = 0; i < d; ++i) out[(size_t) (t0 + t) * d + i] = 0.0f;
        }

        /* counting sort of (token, slot) by expert */
        for (int e = 0; e < NE; ++e) cnt[e] = 0;
        for (int t = 0; t < nt_; ++t)
            for (int k = 0; k < TOPK; ++k)
                if (idx[(size_t) t * TOPK + k] >= 0) ++cnt[idx[(size_t) t * TOPK + k]];
        start[0] = 0;
        for (int e = 0; e < NE; ++e) start[e + 1] = start[e] + cnt[e];
        memcpy(fill, start, (size_t) NE * sizeof(int));
        for (int t = 0; t < nt_; ++t)
            for (int k = 0; k < TOPK; ++k) {
                const int e = idx[(size_t) t * TOPK + k];
                if (e < 0) continue;
                const int at = fill[e]++;
                bt[at] = t;
                bk[at] = k;
            }

        /* Expert-major: one slab read per expert for its whole bucket, instead of once per token. */
        int bad = 0;
#ifdef _OPENMP
        #pragma omp parallel for num_threads(in2_threads()) schedule(dynamic) reduction(|:bad)
#endif
        for (int e = 0; e < NE; ++e) {
            const int nb = cnt[e];
            if (nb <= 0) continue;
            float *Xe = (float *) malloc((size_t) nb * d * sizeof(float));
            float *Ge = (float *) malloc((size_t) nb * 2 * DFF * sizeof(float));
            float *Ae = (float *) malloc((size_t) nb * DFF * sizeof(float));
            float *Oe = (float *) malloc((size_t) nb * d * sizeof(float));
            if (!Xe || !Ge || !Ae || !Oe) { free(Xe); free(Ge); free(Ae); free(Oe); bad = 1; continue; }
            for (int i = 0; i < nb; ++i)
                memcpy(Xe + (size_t) i * d, x + (size_t) (t0 + bt[start[e] + i]) * d, sizeof(float) * (size_t) d);
            if (in2_matvec_quant_nt(ctx, &L->gate_up, (int64_t) e * 2 * DFF * d, Xe, nb, 2 * DFF, d, Ge)
                != WUBU_INKLING2_OK) {
                bad = 1;
            } else {
                const float gsc = gs ? gs[e] : 1.0f;
                for (int i = 0; i < nb * 2 * DFF; ++i) Ge[i] *= gsc;
                for (int i = 0; i < nb; ++i)
                    in2_expert_act(Ge + (size_t) i * 2 * DFF, Ge + (size_t) i * 2 * DFF + DFF, DFF,
                                   Ae + (size_t) i * DFF);
                if (in2_matvec_quant_nt(ctx, &L->down, (int64_t) e * d * DFF, Ae, nb, d, DFF, Oe)
                    != WUBU_INKLING2_OK) bad = 1;
            }
            if (!bad)
                for (int i = 0; i < nb; ++i) {
                    const int t = bt[start[e] + i], k = bk[start[e] + i];
                    memcpy(ok + ((size_t) t * TOPK + (size_t) k) * d, Oe + (size_t) i * d,
                           sizeof(float) * (size_t) d);
                }
            free(Xe); free(Ge); free(Ae); free(Oe);
        }
        if (bad) { rc = WUBU_INKLING2_ERR_IO; break; }

        /* Serial k-order accumulation - the same expression in the same order as the per-token path, so the bits
         * are unchanged. This is the constraint that makes the restructure more than a loop swap. */
        for (int t = 0; t < nt_; ++t) {
            float *o = out + (size_t) (t0 + t) * d;
            for (int k = 0; k < TOPK; ++k) {
                const int e = idx[(size_t) t * TOPK + k];
                if (e < 0) continue;
                const float dsc = ds ? ds[e] : 1.0f;
                const float *op = ok + ((size_t) t * TOPK + (size_t) k) * d;
                const float wv = wt[(size_t) t * TOPK + k];
                for (int i = 0; i < d; ++i) o[i] += wv * op[i] * dsc;
            }
        }
    }

done:
    free(idx); free(wt); free(ok); free(cnt); free(start); free(fill); free(bt); free(bk);
    return rc;
}

/* MoE over n_tok tokens: route, then accumulate the selected experts. */
int in2_moe(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
            int il, const float *x, int n_tok, float *out)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NE = cfg->n_expert, TOPK = cfg->n_expert_used, DFF = cfg->d_ff_expert;
    /* Borrowed: the router matrix is a read-only matvec weight and was being copied (1.1 MB) and freed on every visit. */
    const float *Wr = in2_tensor_borrow(ctx, &L->router);
    float *cb = in2_tensor_to_f32(ctx, &L->exp_bias, NULL);
    float *gs = in2_tensor_to_f32(ctx, &L->gate_up_scale, NULL);
    float *ds = in2_tensor_to_f32(ctx, &L->down_scale, NULL);
    if (!Wr) return WUBU_INKLING2_ERR_IO;

    float *gu = (float *)malloc((size_t)2 * DFF * d * sizeof(float));
    float *dn = (float *)malloc((size_t)d * DFF * sizeof(float));
    float *g  = (float *)malloc((size_t)TOPK * 2 * DFF * sizeof(float));
    float *act = (float *)malloc((size_t)TOPK * DFF * sizeof(float));
    float *eo = (float *)malloc((size_t)TOPK * d * sizeof(float));
    int rc = (gu && dn && g && act && eo) ? WUBU_INKLING2_OK : WUBU_INKLING2_ERR_IO;
    if (getenv("IN2_DEBUG"))
        fprintf(stderr, "[moe L%d] d=%d NE=%d TOPK=%d DFF=%d scale=%.3f bufs=%p/%p/%p/%p/%p rc=%d\n",
                il, d, NE, TOPK, DFF, (double)cfg->expert_weights_scale, (void*)gu, (void*)dn, (void*)g,
                (void*)act, (void*)eo, rc);
    if (rc != WUBU_INKLING2_OK) { free(cb); free(gs); free(ds); free(gu); free(dn); free(g); free(act); free(eo); return rc; }

    const clock_t moe_t0 = clock();
    /* Multi-token forwards go expert-major (IN2_MOE_1T=1 restores the per-token path for A/B). The per-token loop
     * below streams each selected expert's slab once per token, so prefill re-reads ~1.5 GB of weights for every
     * position - measured as TY16 calls scaling exactly with token count at a flat 8.7 GOPS, and as prefill gaining
     * only 2.24x from 1 to 8 threads. Bucketing by expert reads each slab once per bucket instead. */
    if (n_tok >= 2 && !getenv("IN2_MOE_1T")) {
        rc = in2_moe_batch(ctx, m, cfg, il, x, n_tok, out, Wr, cb, gs, ds);
        free(cb); free(gs); free(ds); free(gu); free(dn); free(g); free(act); free(eo);
        if (getenv("IN2_TIME"))
            fprintf(stderr, "  moe L%-2d %.3fs [expert-major]\n", il,
                    (double) (clock() - moe_t0) / (double) CLOCKS_PER_SEC);
        return rc;
    }
    for (int t = 0; t < n_tok && rc == WUBU_INKLING2_OK; ++t) {
        int idx[256]; float w[256];
        const int got = in2_router(x + (size_t)t * d, Wr, cb, NE, d, TOPK, cfg->expert_weights_scale, idx, w);
        for (int i = 0; i < d; ++i) out[(size_t)t * d + i] = 0.0f;
        /* Experts are independent, so the k loop is the natural parallel axis: with n_tok == 1 it is the only one
         * that gives a thread a whole contiguous expert block to stream, which is what this phase is bound by.
         * Each k writes its own g/act/eo slice, so nothing races, and the weighted sum stays a SERIAL loop in k
         * order - the same expression in the same order as the one-expert-at-a-time version, so the bits match. */
        if (!getenv("IN2_STAGE") && got > 1 && in2_threads() > 1) {
            int bad = 0;
#ifdef _OPENMP
            #pragma omp parallel for num_threads(in2_threads()) schedule(dynamic) reduction(|:bad)
#endif
            for (int k = 0; k < got; ++k) {
                const int e = idx[k];
                const float gsc = gs ? gs[e] : 1.0f;
                float *gk = g  + (size_t)k * 2 * DFF;
                float *ak = act + (size_t)k * DFF;
                float *ok = eo + (size_t)k * d;
                if (in2_matvec_quant(ctx, &L->gate_up, (int64_t)e * 2 * DFF * d,
                                     x + (size_t)t * d, 2 * DFF, d, gk) != WUBU_INKLING2_OK) { bad = 1; continue; }
                for (int i = 0; i < 2 * DFF; ++i) gk[i] *= gsc;
                in2_expert_act(gk, gk + DFF, DFF, ak);
                if (in2_matvec_quant(ctx, &L->down, (int64_t)e * d * DFF, ak, d, DFF, ok) != WUBU_INKLING2_OK) {
                    bad = 1; continue;
                }
            }
            if (bad) { rc = WUBU_INKLING2_ERR_IO; break; }
            for (int k = 0; k < got; ++k) {
                const int e = idx[k];
                const float dsc = ds ? ds[e] : 1.0f;
                const float *ok = eo + (size_t)k * d;
                for (int i = 0; i < d; ++i) out[(size_t)t * d + i] += w[k] * ok[i] * dsc;
            }
            continue;
        }
        for (int k = 0; k < got; ++k) {
            const int e = idx[k];
            const float gsc = gs ? gs[e] : 1.0f, dsc = ds ? ds[e] : 1.0f;
            /* fused: dequantise a row, dot it, discard it (IN2_STAGE=1 restores the materialising path) */
            /* Fused everywhere. I predicted staged would win at n_tok > 1 because it dequantises each expert
             * once for all positions, so this was made conditional on n_tok == 1 - and measured the opposite:
             * 5-token forward 47,495 ms staged against 42,774 ms fused. The predicted 5x saving in element work
             * did not materialise because element count was never the binding cost. Kept unconditional, since
             * the measurement beat the reasoning. */
            if (!getenv("IN2_STAGE")) {
                const int ra = in2_matvec_quant_mt(ctx, &L->gate_up, (int64_t)e * 2 * DFF * d,
                                                   x + (size_t)t * d, 2 * DFF, d, g);
                if (ra != WUBU_INKLING2_OK) {
                    if (getenv("IN2_DEBUG")) fprintf(stderr, "[moe L%d] fused gate_up e%d rc=%d\n", il, e, ra);
                    rc = WUBU_INKLING2_ERR_IO; break;
                }
                for (int i = 0; i < 2 * DFF; ++i) g[i] *= gsc;
                in2_expert_act(g, g + DFF, DFF, act);
                if (in2_matvec_quant_mt(ctx, &L->down, (int64_t)e * d * DFF, act, d, DFF, eo) != WUBU_INKLING2_OK) {
                    rc = WUBU_INKLING2_ERR_IO; break;
                }
                for (int i = 0; i < d; ++i) out[(size_t)t * d + i] += w[k] * eo[i] * dsc;
                continue;
            }
            const int r1 = in2_dequant_slice(ctx, &L->gate_up, (int64_t)e * 2 * DFF * d, (int64_t)2 * DFF * d, gu);
            const int r2 = in2_dequant_slice(ctx, &L->down,    (int64_t)e * d * DFF,    (int64_t)d * DFF,    dn);
            if (r1 != WUBU_INKLING2_OK || r2 != WUBU_INKLING2_OK) {
                if (getenv("IN2_DEBUG")) fprintf(stderr, "[moe L%d] dequant e%d rc %d/%d  types %d/%d\n", il, e, r1, r2,
                        L->gate_up.info ? L->gate_up.info->ggml_type : -1, L->down.info ? L->down.info->ggml_type : -1);
                rc = WUBU_INKLING2_ERR_IO; break;
            }
            in2_matvec(gu, x + (size_t)t * d, d, 2 * DFF, g);
            for (int i = 0; i < 2 * DFF; ++i) g[i] *= gsc;
            in2_expert_act(g, g + DFF, DFF, act);
            in2_matvec(dn, act, DFF, d, eo);
            for (int i = 0; i < d; ++i) out[(size_t)t * d + i] += w[k] * eo[i] * dsc;
        }
    }
    free(cb); free(gs); free(ds); free(gu); free(dn); free(g); free(act); free(eo);
    if (getenv("IN2_TIME"))
        fprintf(stderr, "  moe L%-2d %.3fs%s\n", il, (double)(clock() - moe_t0) / (double)CLOCKS_PER_SEC,
                getenv("IN2_STAGE") ? " [staged]" : " [fused]");
    return rc;
}

/* Depth mix over [block_states; partial]: one learned query per requested output.
 * With n_blocks == 0 and n_q == 1 this is exactly an unweighted RMS norm, which is why layer 0's
 * mlp_input (and every layer's mlp_input before any block has been banked) is a norm in disguise. */
/* blocks holds n_blocks entries, each a SEQUENCE of positions: entry i, position pos_off/d, lives at
 * blocks + i*block_stride + pos_off. For a single token block_stride == d and pos_off == 0, which is the
 * only case the earlier version supported. */
void in2_mix(const float *blocks, int n_blocks, int block_stride, int pos_off,
             const float *partial, const float *queries, int n_q, int d, float eps, float *out)
{
    const int n = n_blocks + 1;
    float *st = (float *)malloc((size_t)n * d * sizeof(float));
    float *nr = (float *)malloc((size_t)n * d * sizeof(float));
    float *al = (float *)malloc((size_t)n * sizeof(float));
    if (!st || !nr || !al) { free(st); free(nr); free(al); return; }
    for (int i = 0; i < n_blocks; ++i)
        memcpy(st + (size_t)i * d, blocks + (size_t)i * block_stride + pos_off, sizeof(float) * d);
    memcpy(st + (size_t)n_blocks * d, partial, sizeof(float) * d);
    for (int i = 0; i < n; ++i) in2_rmsnorm_unweighted(st + (size_t)i * d, d, eps, nr + (size_t)i * d);

    const float inv = 1.0f / sqrtf((float)d);
    for (int qi = 0; qi < n_q; ++qi) {
        const float *q = queries + (size_t)qi * d;
        float mx = -1e30f;
        for (int i = 0; i < n; ++i) {
            double acc = 0.0;
            for (int j = 0; j < d; ++j) acc += (double)q[j] * nr[(size_t)i * d + j];
            al[i] = (float)acc * inv;
            if (al[i] > mx) mx = al[i];
        }
        double sum = 0.0;
        for (int i = 0; i < n; ++i) { al[i] = expf(al[i] - mx); sum += al[i]; }
        for (int i = 0; i < n; ++i) al[i] = (float)(al[i] / sum);
        for (int j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i) acc += (double)al[i] * nr[(size_t)i * d + j];
            out[(size_t)qi * d + j] = (float)acc;
        }
    }
    free(st); free(nr); free(al);
}

/* Dynamic short conv (full-attention k/v only).
 *   taps[c][j] = static[c*k + j] + dyn[h][j]      dynamic per HEAD, static per channel
 *   out = (sum_j taps[c][j] * x[t-j][c] + x[t][c]) / sqrt(2k + 1)
 * The static post-attention/post-mlp convs divide by sqrt(k+1) instead; using the same normaliser for
 * both is a silent 1.34x error. */
void in2_shortconv_dynamic(const float *x, int n_tok, int channels, const float *stat,
                           const float *dyn_proj, const float *dyn_input, int hidden,
                           int n_head, int k, float *out)
{
    const int hd = channels / n_head;
    const float scale = 1.0f / sqrtf((float)(2 * k + 1));
    float *dw = (float *)malloc((size_t)n_tok * n_head * k * sizeof(float));
    if (!dw) return;
    for (int t = 0; t < n_tok; ++t) {
        /* dyn_proj is (n_head*k, hidden): one tap set per head, shared across that head's channels */
        in2_matvec(dyn_proj, dyn_input + (size_t)t * hidden, hidden, n_head * k,
                   dw + (size_t)t * n_head * k);
    }
    for (int t = 0; t < n_tok; ++t) {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < hd; ++d) {
                const int c = h * hd + d;
                float acc = 0.0f;
                for (int j = 0; j < k; ++j) {
                    /* tap j has age (k-1-j): the current token is the LAST tap, as in the reference graph */
                    const int src = t - (k - 1 - j);
                    if (src < 0) continue;
                    /* The dynamic taps are ordered the OPPOSITE way to the static ones: dyn index 0 multiplies the
                     * current row, static index k-1 does. Proven on the reference (node_961 - node_950 at ch0 =
                     * dyn[0]*x_t = 3.4170 * -0.5006), so pairing them by the same index is wrong. */
                    const float tap = stat[(size_t)c * k + j] + dw[((size_t)t * n_head + h) * k + (k - 1 - j)];
                    acc += tap * x[(size_t)src * channels + c];
                }
                const float res = x[(size_t)t * channels + c];
                out[(size_t)t * channels + c] = (acc + res) * scale;
            }
        }
    }
    
}

/* Messages (reference: node_3491/node_3495). Per TOKEN, four role queries attend over the block-end residuals of
 * the previous pass:
 *     msg[t][role] = sum_s softmax_s( q[role] . rms(src_s[t]) / sqrt(d) ) * rms(src_s[t])
 * The sources are exactly two -- the residuals at the end of each attn_res_block chunk of the core (reference
 * node_2321 = rms(l_out-11), node_3488 = rms(l_out-17) for EVERY core layer of pass 2; pass 3 uses l_out-29 and
 * l_out-35). Not "all previous blocks", and not position 0 of the batch: the softmax is per token, so a single
 * global vector cannot represent it. Scale 1/sqrt(d) is proven by matching the reference softmax to 4 decimals.
 *
 * The four role queries are laid out across two globals of [24, 2304] (12 core layers x 2 rows each); the reference
 * concatenates them into one [12, 4, 2304] tensor. Row->role order, verified against the reference softmax weights:
 *     role 0 (q)   = qres[2*li]     role 1 (k)   = qkv[2*li]
 *     role 2 (v)   = qkv[2*li + 1]  role 3 (mlp) = qres[2*li + 1]
 * out is [token][role][d]. srcs is [source][token][d]. */
void in2_messages(const float *srcs, int n_src, int n_tok,
                  const float *qres, const float *qkv, int li, int d, float eps, float *out)
{
    if (!srcs || n_src <= 0 || !qres || !qkv || !out) return;
    const float *q[4];
    q[0] = qres + (size_t)(2 * li) * d;
    q[1] = qkv  + (size_t)(2 * li) * d;
    q[2] = qkv  + (size_t)(2 * li + 1) * d;
    q[3] = qres + (size_t)(2 * li + 1) * d;

    float *nr = (float *)malloc((size_t)n_src * d * sizeof(float));
    float *al = (float *)malloc((size_t)n_src * sizeof(float));
    if (!nr || !al) { free(nr); free(al); return; }
    const float inv = 1.0f / sqrtf((float)d);

    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < n_src; ++s)
            in2_rmsnorm_unweighted(srcs + ((size_t)s * n_tok + t) * d, d, eps, nr + (size_t)s * d);
        for (int r = 0; r < 4; ++r) {
            float mx = -1e30f;
            for (int s = 0; s < n_src; ++s) {
                double acc = 0.0;
                for (int j = 0; j < d; ++j) acc += (double)q[r][j] * nr[(size_t)s * d + j];
                al[s] = (float)acc * inv;
                if (al[s] > mx) mx = al[s];
            }
            double sum = 0.0;
            for (int s = 0; s < n_src; ++s) { al[s] = expf(al[s] - mx); sum += al[s]; }
            for (int s = 0; s < n_src; ++s) al[s] = (float)(al[s] / sum);
            for (int j = 0; j < d; ++j) {
                double acc = 0.0;
                for (int s = 0; s < n_src; ++s) acc += (double)al[s] * nr[(size_t)s * d + j];
                out[((size_t)t * 4 + r) * d + j] = (float)acc;
            }
        }
    }
    free(nr); free(al);
}

/* Message gate: out = (1 - tanh(forget(g))) * h + tanh(msg(g)) * message, g = down(cat(rms(h), rms(m))). */
void in2_message_gate(const float *h, const float *message, const float *down, const float *forget,
                      const float *msgp, int d, int rank, float eps, float *out)
{
    float *cat = (float *)malloc((size_t)2 * d * sizeof(float));
    float *g = (float *)malloc((size_t)rank * sizeof(float));
    float *rh = (float *)malloc((size_t)d * sizeof(float));
    float *rm = (float *)malloc((size_t)d * sizeof(float));
    if (!cat || !g || !rh || !rm) { free(cat); free(g); free(rh); free(rm); return; }
    /* Separate unweighted RMS norms of x and m, concatenated as g = [rms(x); rms(m)] - x's columns first.
     * Authority: project-maya src/ik2/ik2_kernels.cuh:194-197, the loop_gates spec:
     *     g = D [rms(x); rms(m)] ;  y = (1 - tanh(F g)) x + tanh(G g) m
     * with the norms taken WITHOUT a weight at eps.
     *
     * An experiment that followed llama.cpp's inkling2.cpp:257-258 instead - one joint rms_norm over the
     * 2d concatenation - was measured and then REVERTED: it moved the single-token top-1 from 14776 to 220
     * (' '), which looked like an improvement, but the reference implementation of this model disagrees
     * with that form. A plausible-looking token is not evidence about which formula the model was trained
     * with. IN2_JOINT_GATE=1 keeps that form reachable for A/B only. */
    if (getenv("IN2_JOINT_GATE")) {
        double ss = 0.0;
        for (int i = 0; i < d; ++i) { ss += (double)h[i] * h[i]; ss += (double)message[i] * message[i]; }
        const float sc = 1.0f / sqrtf((float)(ss / (2.0 * d)) + eps);
        for (int i = 0; i < d; ++i) { cat[i] = h[i] * sc; cat[d + i] = message[i] * sc; }
    } else {
        in2_rmsnorm_unweighted(h, d, eps, rh);
        in2_rmsnorm_unweighted(message, d, eps, rm);
        memcpy(cat, rh, sizeof(float) * d);
        memcpy(cat + d, rm, sizeof(float) * d);
    }
    in2_matvec(down, cat, 2 * d, rank, g);
    for (int i = 0; i < d; ++i) {
        double af = 0.0, am = 0.0;
        for (int r = 0; r < rank; ++r) {
            af += (double)forget[(size_t)i * rank + r] * g[r];
            am += (double)msgp[(size_t)i * rank + r] * g[r];
        }
        out[i] = (1.0f - tanhf((float)af)) * h[i] + tanhf((float)am) * message[i];
    }
    free(cat); free(g); free(rh); free(rm);
}

/* Full-attention layer: three depth-mixed states -> three input norms -> tau-scaled q, dynamic k/v
 * convs, attention over every key (the bias still dies outside 0..1023). Tail is identical to the
 * sliding layer: post-attn conv, boundary rule, mlp_input mix, MoE, post-mlp conv. */
int in2_layer_full(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
                   int il, float *h, int n_tok, int pos0, const float *blocks, int n_blocks,
                   const float *msgs)
{
    const wubu_in2_layer_t *L = &m->layer[il];
    const int d = cfg->d_model, NH = cfg->n_head, NKV = cfg->n_kv_head, HD = cfg->head_dim;
    const int qdim = NH * HD, kdim = NKV * HD;
    const float eps = cfg->rms_eps;

    /* 1. three depth-mixed states from the 3-query attention_input */
    const float *wq_n = in2_tensor_borrow(ctx, &L->attn_res_q);
    float *state = (float *)calloc((size_t)3 * d, sizeof(float));
    if (!wq_n || !state) { free(state); return WUBU_INKLING2_ERR_IO; }
    /* superseded by in2_layer_full_seq, kept only for single-token reference; block_stride == d here */
    in2_mix(blocks, n_blocks, d, 0, h, wq_n, 3, d, eps, state);   /* rows: q, k, v */

    float *qn = in2_tensor_to_f32(ctx, &L->q_inp_norm, NULL);
    float *kn = in2_tensor_to_f32(ctx, &L->k_inp_norm, NULL);
    float *vn = in2_tensor_to_f32(ctx, &L->v_inp_norm, NULL);
    if (!qn || !kn || !vn) { free(state); free(qn); free(kn); free(vn);
                             return WUBU_INKLING2_ERR_IO; }
    float *xq = (float *)calloc((size_t)d, sizeof(float));
    float *xk = (float *)calloc((size_t)d, sizeof(float));
    float *xv = (float *)calloc((size_t)d, sizeof(float));
    /* blk.11 and blk.17 (full layers inside the loop core) carry all four loop-gate tensors as well as
     * the three input norms, so the messages must gate the three mixed states here too - otherwise those
     * 24 tensors are dead weight, which no converter emits. blk.23 has no gates and passes msgs == NULL. */
    for (int sl = 0; sl < 4 && msgs; ++sl) {
        const float *dw = in2_tensor_borrow(ctx, &L->loop_down[sl]);
        const float *fw = in2_tensor_borrow(ctx, &L->loop_forget[sl]);
        const float *gw = in2_tensor_borrow(ctx, &L->loop_gain[sl]);
        if (dw && fw && gw) {
            float *gated = (float *)calloc((size_t)d, sizeof(float));
            float *dst = (sl == 0) ? state : (sl == 1 ? state + d : (sl == 2 ? state + 2 * d : NULL));
            if (dst) {
                in2_message_gate(dst, msgs + (size_t)sl * d, dw, fw, gw, d, cfg->loop_gate_rank, eps, gated);
                memcpy(dst, gated, sizeof(float) * d);
            } else {
                /* slot 3 gates the mlp input; it is applied after the depth mix, below */
                in2_message_gate(state, msgs + (size_t)sl * d, dw, fw, gw, d, cfg->loop_gate_rank, eps, gated);
                memcpy(h, gated, sizeof(float) * d);
            }
            free(gated);
        }
        
    }
    in2_rmsnorm(state,           qn, d, eps, xq);
    in2_rmsnorm(state + d,       kn, d, eps, xk);
    in2_rmsnorm(state + 2 * d,   vn, d, eps, xv);

    /* 2. q with the scaled clip and the log-position tau (tau == 1 at position 0, so a first-token run
     *    does not exercise it - that has to wait for multi-token runs). */
    float *Wq = in2_tensor_to_f32(ctx, &L->wq, NULL);
    float *qcl = in2_tensor_to_f32(ctx, &L->q_clip, NULL);
    float *q = (float *)calloc((size_t)qdim, sizeof(float));
    if (Wq && q) {
        if (!getenv("IN2_NO_FULL_KERNEL")) in2_matvec_quant(ctx, &L->wq, 0, xq, qdim, d, q);
        else                          in2_matvec(Wq, xq, d, qdim, q);
        in2_l2_clip(q, qcl, NH, HD, 22.6274166f);
        const float tau = 1.0f + 0.1f * logf(fmaxf(((float)pos0 + 1.0f) / 8192.0f, 1.0f));
        for (int i = 0; i < qdim; ++i) q[i] *= tau;
    } else { free(state); free(qn); free(kn); free(vn);
             free(xq); free(xk); free(xv); free(Wq); free(qcl); free(q); return WUBU_INKLING2_ERR_IO; }

    /* 3. k and v with the DYNAMIC short conv (static per channel + per-head projected taps) */
    float *Wk = in2_tensor_to_f32(ctx, &L->wk, NULL);
    float *Wv = in2_tensor_to_f32(ctx, &L->wv, NULL);
    float *kc = in2_tensor_to_f32(ctx, &L->k_conv, NULL);
    float *vc = in2_tensor_to_f32(ctx, &L->v_conv, NULL);
    float *kp = in2_tensor_to_f32(ctx, &L->k_conv_proj, NULL);
    float *vp = in2_tensor_to_f32(ctx, &L->v_conv_proj, NULL);
    float *kraw = (float *)calloc((size_t)kdim, sizeof(float));
    float *vraw = (float *)calloc((size_t)kdim, sizeof(float));
    float *kt = (float *)calloc((size_t)kdim, sizeof(float));
    float *vt = (float *)calloc((size_t)kdim, sizeof(float));
    float *k = (float *)calloc((size_t)kdim, sizeof(float));
    float *v = (float *)calloc((size_t)kdim, sizeof(float));
    int rc = WUBU_INKLING2_OK;
    if (!Wk || !Wv || !kc || !vc || !kp || !vp || !kraw || !vraw || !kt || !vt || !k || !v) rc = WUBU_INKLING2_ERR_IO;
    if (rc == WUBU_INKLING2_OK) {
        if (!getenv("IN2_NO_FULL_KERNEL")) in2_matvec_quant(ctx, &L->wk, 0, xk, kdim, d, kraw);
        else                          in2_matvec(Wk, xk, d, kdim, kraw);
        if (!getenv("IN2_NO_FULL_KERNEL")) in2_matvec_quant(ctx, &L->wv, 0, xv, kdim, d, vraw);
        else                          in2_matvec(Wv, xv, d, kdim, vraw);
        in2_shortconv_dynamic(kraw, 1, kdim, kc, kp, xk, d, NKV, 4, kt);
        in2_shortconv_dynamic(vraw, 1, kdim, vc, vp, xv, d, NKV, 4, vt);
        memcpy(k, kt, sizeof(float) * kdim);
        in2_l2_clip(k, NULL, NKV, HD, 22.6274166f);
        memcpy(v, vt, sizeof(float) * kdim);
        for (int hh = 0; hh < NKV; ++hh) {
            float *row = v + hh * HD;
            in2_rmsnorm_unweighted(row, HD, eps, row);
        }
    }

    /* 4. relative bias + sinks, both conditioned on the q input */
    float *Wrel = in2_tensor_to_f32(ctx, &L->wrel, NULL);
    float *rcl = in2_tensor_to_f32(ctx, &L->rel_clip, NULL);
    float *P = in2_tensor_to_f32(ctx, &L->rel_bias, NULL);
    float *Wsk = in2_tensor_to_f32(ctx, &L->sink, NULL);
    const int DREL = 16, EXT = 1024;
    float *rs = (float *)calloc((size_t)NH * DREL, sizeof(float));
    float *bias = (float *)calloc((size_t)NH * EXT, sizeof(float));
    float *sink = (float *)calloc((size_t)NH, sizeof(float));
    if (Wrel && rcl && P && Wsk && rs && bias && sink) {
        in2_matvec(Wrel, xq, d, NH * DREL, rs);
        in2_l2_clip(rs, rcl, NH, DREL, 8.0f);
        in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);
        in2_matvec(Wsk, xq, d, NH, sink);
    } else rc = WUBU_INKLING2_ERR_IO;

    /* 5. attention over every key (single token here, so one key plus the sink) */
    float *attn_out = (float *)calloc((size_t)qdim, sizeof(float));
    float *logit = (float *)malloc(sizeof(float) * (size_t)n_tok);
    float *alpha = (float *)malloc(sizeof(float) * (size_t)n_tok);
    if (attn_out && logit && alpha && rc == WUBU_INKLING2_OK) {
        for (int hh = 0; hh < NH; ++hh) {
            const int kvh = hh / (NH / NKV);
            int n_key = 0;
            for (int j = 0; j < n_tok; ++j) {
                const int dist = pos0 - (pos0 + j - n_tok + 1);
                (void)dist;
                double acc = 0.0;
                for (int e = 0; e < HD; ++e) acc += (double)q[hh * HD + e] * k[kvh * HD + e];
                logit[n_key] = (float)acc * (1.0f / sqrtf((float)HD)) + bias[hh * EXT + 0];
                ++n_key;
            }
            in2_softmax_sink(logit, n_key, sink[hh], alpha);
            float *o = attn_out + hh * HD;
            for (int j = 0; j < n_key; ++j)
                for (int e = 0; e < HD; ++e) o[e] += alpha[j] * v[kvh * HD + e];
        }
    } else rc = WUBU_INKLING2_ERR_IO;

    /* 6. gated output projection */
    float *Wg = in2_tensor_to_f32(ctx, &L->wgate, NULL);
    float *Wo = in2_tensor_to_f32(ctx, &L->wo, NULL);
    float *gate = (float *)calloc((size_t)qdim, sizeof(float));
    float *ao = (float *)calloc((size_t)d, sizeof(float));
    if (Wg && Wo && gate && ao && rc == WUBU_INKLING2_OK) {
        in2_matvec(Wg, xq, d, qdim, gate);
        for (int i = 0; i < qdim; ++i) gate[i] = attn_out[i] * (1.0f / (1.0f + expf(-gate[i])));
        in2_matvec(Wo, gate, qdim, d, ao);
    } else rc = WUBU_INKLING2_ERR_IO;

    /* 7. same tail as the sliding layer */
    float *pc = in2_tensor_to_f32(ctx, &L->attn_post_conv, NULL);
    float *res = (float *)calloc((size_t)d, sizeof(float));
    if (pc && res && rc == WUBU_INKLING2_OK) {
        in2_shortconv(ao, 1, d, pc, 4, res);
        const int boundary = (cfg->attn_res_block > 0) && (il % cfg->attn_res_block == 0);
        if (!boundary) for (int i = 0; i < d; ++i) res[i] = h[i] + res[i];
    } else rc = WUBU_INKLING2_ERR_IO;

    float *fq = in2_tensor_to_f32(ctx, &L->ffn_res_q, NULL);
    float *mixed = (float *)calloc((size_t)d, sizeof(float));
    float *fn = in2_tensor_to_f32(ctx, &L->ffn_norm, NULL);
    float *moein = (float *)calloc((size_t)d, sizeof(float));
    float *moeout = (float *)calloc((size_t)d, sizeof(float));
    if (fq && mixed && fn && moein && moeout && rc == WUBU_INKLING2_OK) {
        in2_mix(blocks, n_blocks, d, 0, res, fq, 1, d, eps, mixed);
        in2_rmsnorm(mixed, fn, d, eps, moein);
        rc = in2_moe(ctx, m, cfg, il, moein, 1, moeout);
    } else rc = WUBU_INKLING2_ERR_IO;

    float *pm = in2_tensor_to_f32(ctx, &L->ffn_post_conv, NULL);
    float *mo = (float *)calloc((size_t)d, sizeof(float));
    if (pm && mo && rc == WUBU_INKLING2_OK) {
        in2_shortconv(moeout, 1, d, pm, 4, mo);
        for (int i = 0; i < d; ++i) h[i] = res[i] + mo[i];
    } else rc = WUBU_INKLING2_ERR_IO;

    free(state); free(qn); free(kn); free(vn); free(xq); free(xk); free(xv);
    free(Wq); free(qcl); free(q); free(Wk); free(Wv); free(kc); free(vc); free(kp); free(vp);
    free(kraw); free(vraw); free(kt); free(vt); free(k); free(v);
    free(Wrel); free(rcl); free(P); free(Wsk); free(rs); free(bias); free(sink);
    free(attn_out); free(logit); free(alpha); free(Wg); free(Wo); free(gate); free(ao);
    free(pc); free(res); free(fq); free(mixed); free(fn); free(moein); free(moeout); free(pm); free(mo);
    return rc;
}

/* Dequantise a whole tensor into f32. Caller frees. Returns NULL on failure. */
/* Dequantise-once cache for the dense weights.
 *
 * A dense weight never changes, yet in2_tensor_to_f32 re-dequantised it on EVERY layer visit: wq/wo/wgate are
 * 2304x3072 and wk/wv 2304x1024, so a layer pulls ~110 MB of floats through scalar dequantisation, and 24
 * layers x 3 passes = 72 visits do it again every token - roughly 8 GB of redundant dequantisation per token.
 *
 * The cache keeps the float copy and returns a malloc'd DUPLICATE, so every existing caller's free() stays
 * correct and no call site changes. The duplicate costs a memcpy (a few ms at 28 MB) against the
 * dequantisation it replaces (tens of ms), which is the whole trade. Off by default; IN2_DQ_CACHE=1 enables.
 * Memory: the dense set is ~2.6 GB for this file, so it is opt-in rather than always-on. */
typedef struct { const gguf_tensor_info *key; float *buf; int64_t n; } in2_dq_ent;
static in2_dq_ent in2_dq_tab[2048];
#define IN2_DQ_MAX 2048
int in2_dq_n;
size_t in2_dq_bytes;   /* bytes currently held by the dequant cache */
/* One budget for the whole cache, shared by the insert path and the borrow/evict path. It was a function-local static
 * inside the insert path, which meant the eviction code could not see it. */
static size_t dq_budget = 0;
long in2_dq_miss;      /* dequantisations that could not be cached (table full or over budget) */
/* Traffic accounting for the f32 materialisation path. `to_f32` copies on a cache hit and dequantises on a miss;
 * `borrow` hands out the table's own buffer. Without these the cost of turning quantised weights into floats is
 * invisible - the GGUF byte counts say nothing about it. */
unsigned long long in2_dq_produced;   /* f32 bytes dequantised (cache miss) */
unsigned long long in2_dq_copied;     /* f32 bytes memcpy'd out of the cache (hit) */
unsigned long long in2_dq_fast_calls; /* calls into in2_tensor_to_f32 */
unsigned long long in2_dq_borrow_hit; /* borrow hits: no copy, no dequant */
unsigned long long in2_dq_borrow_miss;/* borrow misses: dequantised and installed */

float *in2_tensor_to_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, int64_t *n_out)
{
    if (!ctx || !t || !t->info) return NULL;
    int64_t ne = 1;
    for (int d = 0; d < t->info->n_dims; ++d) ne *= t->info->dims[d];

    /* On by default. It was a 0.8%% effect when the big matrices still went through the float path, which is why it
     * was left behind a flag. The per-visit tensors are the hot spot, but not for the reason an earlier comment
     * gave: attn_res_q is a 2304-float vector (9 KB), measured from the GGUF header, not a 2304x2304 matrix.
     * caller's ownership contract is unchanged. IN2_NO_DQ_CACHE disables it. */
    const int cache_on = g_force_cache || getenv("IN2_NO_DQ_CACHE") == NULL;
    ++in2_dq_fast_calls;
    if (cache_on) {
        for (int i = 0; i < in2_dq_n; ++i) {
            if (in2_dq_tab[i].key == t->info) {
                float *cp = (float *)malloc((size_t)in2_dq_tab[i].n * sizeof(float));
                if (!cp) return NULL;
                memcpy(cp, in2_dq_tab[i].buf, (size_t)in2_dq_tab[i].n * sizeof(float));
                in2_dq_copied += (unsigned long long) in2_dq_tab[i].n * sizeof(float);
                if (in2_dq_tab[i].n > 262144)   /* >1 MB of floats: these are the ones worth naming */
                    fprintf(stderr, "DQTRACE copy %-40s %.1f MB\n", t->info->name ? t->info->name : "?",
                            (double) in2_dq_tab[i].n * sizeof(float) / 1048576.0);
                if (n_out) *n_out = in2_dq_tab[i].n;
                return cp;
            }
        }
    }

    float *buf = (float *)malloc((size_t)ne * sizeof(float));
    if (!buf) return NULL;
    if (wubu_inkling2_read_f32(ctx, t, buf, ne) != WUBU_INKLING2_OK) { free(buf); return NULL; }
    in2_dq_produced += (unsigned long long) ne * sizeof(float);

    /* Byte budget, not just an entry count: 512 entries of "whatever was touched" is unbounded in practice, and the
     * model is 8.5 GB on a 16 GB box. Measured peak working set of a 24-token run was 10.1 GB; the cache is a pure
     * speed cache, so capping it changes no arithmetic. IN2_DQ_MB overrides. */
    if (dq_budget == 0) {
        const char *_dqmb = getenv("IN2_DQ_MB");
        /* 1024 MB, not 512: once the router and the loop message gates borrow (4c7e6f5) the borrowed working set is
         * ~558 MB, so a 512 MB budget sits just under it and thrashes - measured 2.41 tok/s with 1880 misses and
         * 4460 MB re-dequantised per 8 tokens, against 3.63 tok/s, 235 misses and 557 MB at 1024. Going higher buys
         * nothing (2048 and 3000 measure the same), which is what a working-set-sized cache looks like. */
        dq_budget = (size_t)(_dqmb ? atoi(_dqmb) : 1024) * 1024u * 1024u;
    }
    if (cache_on && in2_dq_n < IN2_DQ_MAX && in2_dq_bytes + (size_t)ne * sizeof(float) <= dq_budget) {
        float *keep = (float *)malloc((size_t)ne * sizeof(float));
        if (keep) {
            memcpy(keep, buf, (size_t)ne * sizeof(float));
            in2_dq_tab[in2_dq_n].key = t->info;
            in2_dq_tab[in2_dq_n].buf = keep;
            in2_dq_tab[in2_dq_n].n   = ne;
            in2_dq_bytes += (size_t)ne * sizeof(float);
            ++in2_dq_n;
        }
    } else if (cache_on) {
        ++in2_dq_miss;
    }
    if (n_out) *n_out = ne;
    return buf;
}

int in2_dq_force_begin(void);
void in2_dq_force_end(int);

/* Borrowed, table-owned view of a weight matrix.
 *
 * The DQ cache returns a COPY so every caller's free() stays valid - correct, but it means a constant 3072x2304
 * weight matrix gets copied out of the cache on every layer visit, which is 28 MB x 8 full-layer visits per token.
 * This returns the table's buffer instead and never hands back caller-owned memory (it forces the cache on for the
 * call), so a caller that does not free anything cannot leak.
 *
 * Only for read-only weight use. Never free the result. */
const float *in2_tensor_borrow(gguf_ctx *ctx, const wubu_in2_tensor_t *t)
{
    extern int in2_dq_force_begin(void);
    extern void in2_dq_force_end(int);
    if (!ctx || !t || !t->info) return NULL;
    /* Hit path must NOT go through in2_tensor_to_f32: that contract is "return a copy the caller owns", so it
     * memcpy's the whole matrix (28 MB for a 3072x2304 path) and the copy is dropped immediately after. That copy
     * was ~448 MB of pure memcpy per decode token, 11 ms per full-layer visit, and it made the full-attention
     * layers look 10x more expensive than sliding ones doing identical arithmetic. Look the table up directly. */
    for (int i = 0; i < in2_dq_n; ++i)
        if (in2_dq_tab[i].key == t->info) { ++in2_dq_borrow_hit; return in2_dq_tab[i].buf; }
    ++in2_dq_borrow_miss;

    const int saved = in2_dq_force_begin();
    int64_t ne = 0;
    float *v = in2_tensor_to_f32(ctx, t, &ne);
    in2_dq_force_end(saved);
    if (!v) return NULL;
    /* first touch: the cache now owns a copy of its own, so drop ours and hand back the table's pointer */
    for (int i = 0; i < in2_dq_n; ++i)
        if (in2_dq_tab[i].key == t->info) { free(v); return in2_dq_tab[i].buf; }

    /* The table refused it (entry cap or byte budget). Handing `v` back here would be a LEAK, not a fallback: every
     * caller holds the returned pointer as table-owned and never frees it - that is the contract of borrow. It used to
     * do exactly that, and the cost was measured at 200-390 MB of new pages per decode token (WorkingSetSize climbing
     * 3090 -> 3726 MB over four steps, 95k page faults per step with zero disk reads, and peak RSS 7.0 GB at 12 tokens
     * vs 10.3 GB at 24). Bounded memory is worth an occasional re-dequantise; an unbounded leak is not.
     * Slot 0 is the oldest entry because nothing here ever removes or reorders, so index order is insertion order. */
    const size_t want = (size_t) ne * sizeof(float);
    while (in2_dq_n > 0 && (in2_dq_n >= IN2_DQ_MAX || in2_dq_bytes + want > dq_budget)) {
        in2_dq_bytes -= (size_t) in2_dq_tab[0].n * sizeof(float);
        free(in2_dq_tab[0].buf);
        memmove(&in2_dq_tab[0], &in2_dq_tab[1], sizeof in2_dq_tab[0] * (size_t) (in2_dq_n - 1));
        --in2_dq_n;
    }
    in2_dq_tab[in2_dq_n].key = t->info;
    in2_dq_tab[in2_dq_n].buf = v;
    in2_dq_tab[in2_dq_n].n   = ne;
    in2_dq_bytes += want;
    ++in2_dq_n;
    return v;   /* now owned by the table, so a later eviction frees it */
}

/* Token embedding lookup: token_embd is (n_embd, n_vocab) so token t occupies columns t.
 * Rows are the embedding axis, which is dims[0], so the gather is strided over the vocab axis. */
int in2_embed(gguf_ctx *ctx, const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg,
              int token, float *out)
{
    if (!m->tok_embd.info) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    const gguf_tensor_info *ti = m->tok_embd.info;
    if (ti->n_dims != 2 || ti->dims[0] != (int64_t)cfg->d_model) return WUBU_INKLING2_ERR_SHAPE;
    if (token < 0 || token >= cfg->vocab_size) return WUBU_INKLING2_ERR_SHAPE;

    /* One row, not the whole matrix. This used to dequantise all 201024 x 2304 embedding values (1.85 GB, past any
     * cache budget, so it happened again on every single token) to copy out 2304 floats. The row path reads 1.2 MB of
     * quantised bytes when the row is block-aligned and produces exactly the same floats; if it is not aligned (a
     * file-dependent property) fall back to the whole-tensor path rather than reading shifted blocks. */
    const int64_t stride = ti->dims[0];
    if (wubu_inkling2_read_row_f32(ctx, &m->tok_embd, (int64_t)token, out, stride) == WUBU_INKLING2_OK)
        return WUBU_INKLING2_OK;

    int64_t ne = 0;
    float *emb = in2_tensor_to_f32(ctx, &m->tok_embd, &ne);
    if (!emb) return WUBU_INKLING2_ERR_IO;
    for (int64_t i = 0; i < stride; ++i) out[i] = emb[(int64_t)token * stride + i];
    free(emb);
    return WUBU_INKLING2_OK;
}
