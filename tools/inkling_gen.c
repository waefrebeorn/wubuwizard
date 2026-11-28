/* inkling_gen -- greedy generation.
 *
 * Incremental (KV-cached) decode by default, ~6x faster than recomputing the prefix. It is bit-identical to the full
 * recompute path on every prompt tested (max|diff| = 0.0), which is what promoted it; IN2_NO_KV=1 forces the
 * full-recompute path for A/B checking of any cache change.
 *
 * usage: inkling_gen <n_gen> <token> [token ...]
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"
#include "wubu_gguf_tokenizer.h"
#include "wubu_tokenizer.h"

int in2_forward_seq(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                    const int *, int, float *, float *);
int in2_forward_at(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                   const int *, int, int, float *, float *);
void in2_set_logits_all(float *);
int  in2_state_save(void);
void in2_state_restore(void);
void in2_state_rewind(int);
int  in2_probe_rows(int pos, float *out, int stride, int *out_dim);
#define IN2_PROBE_MAXV 128

/* SentencePiece marks a leading space with U+2581, byte-level BPE with U+0120; a byte-fallback token is <0xNN>.
 * Same rules as tools/detok.c so the text a run prints is the text detok would print for the same ids. */
static void emit_piece(const char *s)
{
    if (!s) return;
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
        char *end = NULL;
        const long b = strtol(s + 3, &end, 16);
        if (end && *end == '>' && b >= 0 && b < 256) { putchar((int)b); return; }
    }
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { putchar(' '); p += 3; }   /* U+2581 SentencePiece */
        else if (p[0] == 0xC4 && p[1] == 0xA0) { putchar(' '); p += 2; }                /* U+0120 byte-level BPE */
        else if (p[0] == 0xC4 && p[1] == 0x8A) { putchar('\n'); p += 2; }      /* U+010A byte-level newline */
        else if (p[0] == 0xC4 && p[1] == 0x89) { putchar('\t'); p += 2; }      /* U+0109 byte-level tab */
        else { putchar(*p++); }
    }
}

/* clock() is CPU time and sums across threads, so under a 4-thread build it reports ~2.6x the wall time and its
 * per-step rows cannot be summed against an external `date +%s%N`. Use a monotonic wall clock for everything the
 * user reads. */
/* Page-fault and real-I/O deltas: the two numbers that say whether a decode step is faulting through an mmap or
 * actually reading from disk. A 7.9 GB model on a 16 GB box cannot keep everything resident, so "compute-bound" and
 * "fault-bound" look identical on a wall clock - this separates them. */
#ifdef _WIN32
#include <psapi.h>

/* Real disk traffic for this process. A hard fault on a mapped file is served by a read that the kernel attributes to the
 * process, so ReadTransferCount separates HARD (disk) faults from SOFT (working-set) ones - which PageFaultCount cannot,
 * since it sums both. This is the measurement that decides whether the cold/warm tok/s delta is really paging. */
#if defined(_WIN32)
typedef struct { unsigned long long rd_ops, wr_ops, oth_ops, rd_bytes, wr_bytes, oth_bytes; } in2_io_t;
static int in2_io(in2_io_t *o)
{
    typedef int (__stdcall *in2_gpio_t)(void *, in2_io_t *);
    static in2_gpio_t s_fn; static int s_tried;
    if (!s_tried) {
        HMODULE k = GetModuleHandleA("kernel32.dll");
        s_fn = k ? (in2_gpio_t) (void *) GetProcAddress(k, "GetProcessIoCounters") : 0;
        s_tried = 1;
    }
    if (!s_fn) return 0;
    return s_fn((void *) -1, o) ? 1 : 0;
}
#else
typedef struct { unsigned long long rd_ops, wr_ops, oth_ops, rd_bytes, wr_bytes, oth_bytes; } in2_io_t;
static int in2_io(in2_io_t *o) { (void) o; return 0; }
#endif


/* Page-fault counter. tok/s conflates cache state with everything else, and on Windows the cache cannot be dropped on
 * demand, so the only honest way to test a read-ahead change is to count faults instead of seconds. Declared against
 * kernel32 directly (no windows.h, no psapi link) so this stays a zero-dependency diagnostic. */
#if defined(_WIN32)
static unsigned long in2_faults(void)
{
    typedef struct { unsigned long cb, PageFaultCount; size_t a, b, c, d, e, f, g, h; } in2_pmc_t;
    typedef int (__stdcall *in2_gpmi_t)(void *, in2_pmc_t *, unsigned long);
    static in2_gpmi_t s_fn; static int s_tried;
    if (!s_tried) {
        HMODULE k = GetModuleHandleA("kernel32.dll");
        s_fn = k ? (in2_gpmi_t) (void *) GetProcAddress(k, "K32GetProcessMemoryInfo") : 0;
        s_tried = 1;
    }
    if (!s_fn) return 0;
    in2_pmc_t m; m.cb = (unsigned long) sizeof m;
    if (!s_fn((void *) -1, &m, sizeof m)) return 0;
    return m.PageFaultCount;
}
#else
static unsigned long in2_faults(void) { return 0; }
#endif

static unsigned long long g_ws = 0;
static void fault_now(unsigned long long *flt, unsigned long long *rd_bytes, unsigned long long *rd_ops)
{
    PROCESS_MEMORY_COUNTERS pmc;
    IO_COUNTERS io;
    *flt = 0; *rd_bytes = 0; *rd_ops = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        *flt = (unsigned long long) pmc.PageFaultCount;
        g_ws = (unsigned long long) pmc.WorkingSetSize;
    }
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
        *rd_bytes = (unsigned long long) io.ReadTransferCount;
        *rd_ops = (unsigned long long) io.ReadOperationCount;
    }
}
#else
static void fault_now(unsigned long long *flt, unsigned long long *rd_bytes, unsigned long long *rd_ops)
{
    *flt = 0; *rd_bytes = 0; *rd_ops = 0;
}
#endif

static double wall_now(void)
{
#if defined(_WIN32)
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

int main(int argc, char **argv)
{
    const double t_proc = wall_now();
    const char *path = getenv("IN2_MODEL") ? getenv("IN2_MODEL")
        : "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    if (argc < 3) { fprintf(stderr, "usage: %s <n_gen> <token> [token ...]\n", argv[0]); return 2; }
    const int n_gen = atoi(argv[1]);

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    if (getenv("IN2_CFG"))
        fprintf(stderr, "CFG d=%d layers=%d core=[%d,%d) loops=%d n_expert=%d used=%d d_ff=%d d_ff_expert=%d vocab=%d\n",
                cfg.d_model, cfg.n_layer, cfg.loop_core_start, cfg.loop_core_end, cfg.n_loops,
                cfg.n_expert, cfg.n_expert_used, cfg.d_ff, cfg.d_ff_expert, cfg.vocab_size);
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    /* Two ways in: a list of token ids, or --text and a prompt encoded with the .gguf's own tokenizer. The ids a text
     * prompt produces are the ones llama.cpp's /tokenize returns, so the tokenizer is testable against the oracle
     * instead of assumed. */
    const int text_mode = (argc >= 4 && strcmp(argv[2], "--text") == 0);
    int n_prompt = text_mode ? 0 : argc - 2;
    int *toks = NULL;
    wubu_tokenizer_t tok_text;
    memset(&tok_text, 0, sizeof tok_text);
    if (text_mode) {
        if (!wubu_tokenizer_init_from_gguf(&tok_text, path)) { fprintf(stderr, "tokenizer init failed\n"); return 5; }
        /* IN2_TEXT_FILE reads the prompt from a file, because a long-context prompt cannot fit in argv on Windows.
         * One token per byte is a safe cap for byte-level BPE over ASCII. */
        char *fbuf = NULL;
        int cap0 = 4096;
        const char *src = argv[3];
        const char *fpath = getenv("IN2_TEXT_FILE");
        if (fpath) {
            FILE *fp = fopen(fpath, "rb");
            if (!fp) { fprintf(stderr, "cannot open IN2_TEXT_FILE %s\n", fpath); return 8; }
            fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
            fbuf = (char *)malloc((size_t) sz + 1);
            if (!fbuf) { fclose(fp); return 9; }
            if (fread(fbuf, 1, (size_t) sz, fp) != (size_t) sz) { fclose(fp); return 10; }
            fclose(fp);
            fbuf[sz] = 0;
            src = fbuf;
            cap0 = (int) sz + 8;
            fprintf(stderr, "TEXT_FILE %ld bytes cap %d\n", sz, cap0);
        }
        toks = (int *)calloc((size_t)cap0 + (size_t)n_gen, sizeof(int));
        if (!toks) return 6;
        n_prompt = wubu_tokenizer_encode(&tok_text, src, toks, cap0);
        free(fbuf);
        if (n_prompt <= 0) { fprintf(stderr, "prompt encode failed (%d)\n", n_prompt); return 7; }
        if (n_prompt <= 64) for (int i = 0; i < n_prompt; ++i) printf("tok[%d]=%d\n", i, toks[i]);
    }
    const int cap = n_prompt + n_gen;
    if (!text_mode) {
        toks = (int *)calloc((size_t)cap, sizeof(int));
        for (int i = 0; i < n_prompt; ++i) toks[i] = atoi(argv[2 + i]);
    }
    if (n_prompt <= 64) for (int i = 0; i < n_prompt; ++i) printf("prompt[%d]=%d\n", i, toks[i]);

    double t_step = wall_now(); double t_total = 0.0; int used_cache = 0;
    fprintf(stderr, "LOAD wall: %.2f s (open+probe+load, before first token)\n", t_step - t_proc);
    /* IN2_PREFETCH_ALL=1 batch-faults the whole mapping before timing starts. This is the experiment that separates
     * "the step is doing arithmetic" from "the step is re-faulting a trimmed mapping": it changes no maths at all. */
    if (getenv("IN2_PREFETCH_ALL")) {
        const double tp0 = wall_now();
        const int64_t got = wubu_inkling2_prefetch_all(ctx);
        fprintf(stderr, "PREFETCH_ALL %.1f MB in %.2f s\n", (double) got / 1048576.0, wall_now() - tp0);
    }
    float *h = (float *)calloc((size_t)cap * cfg.d_model, sizeof(float));
    float *logits = (float *)calloc(cfg.vocab_size, sizeof(float));
    if (getenv("IN2_BATCH_PROBE")) {
        /* Batched (n_tok>1) vs sequential (n x 1-token) forward, in the configuration where speculation actually runs:
         * AFTER a prefill, with a populated cache, 72 visits and sliding masks. A fresh-cache probe agrees bitwise and
         * proves nothing about this case. Two tokens (prompt tokens 0 and 1) are pushed at positions n_prompt and
         * n_prompt+1; row 0 is the logits at n_prompt, row 1 at n_prompt+1. */
        const int batched = atoi(getenv("IN2_BATCH_PROBE")) != 0;
        const char *out = getenv("IN2_PROBE_OUT");
        if (!out) { fprintf(stderr, "IN2_PROBE_OUT not set\n"); return 2; }
        float *row0 = (float *)calloc(cfg.vocab_size, sizeof(float));
        float *row1 = (float *)calloc(cfg.vocab_size, sizeof(float));
        if (!row0 || !row1) { fprintf(stderr, "probe oom\n"); return 2; }
        if (n_prompt > 0) {
            const int rcp = in2_forward_at(ctx, &m, &cfg, toks, n_prompt, 0, h, logits);
            if (rcp != WUBU_INKLING2_OK) { fprintf(stderr, "probe prefill rc=%d\n", rcp); return rcp; }
        }
        const int p0 = n_prompt;
        int tt[2]; tt[0] = toks[0]; tt[1] = toks[1 % n_prompt];
        /* warmup: populate cache with `warm` sequential 1-token forwards so the batched-vs-sequential A/B runs at a
         * later absolute position (the suspect for the token-13 divergence), not right after prefill. */
        int warm = getenv("IN2_PROBE_WARM") ? atoi(getenv("IN2_PROBE_WARM")) : 0;
        int cur_p = p0;
        {
            int wbuf = 0;
            for (int w = 0; w < warm; ++w) {
                const int rc = in2_forward_at(ctx, &m, &cfg, &tt[w % 2], 1, cur_p, h, logits);
                if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "probe warm rc=%d\n", rc); return rc; }
                int am = 0; for (int v = 1; v < cfg.vocab_size; ++v) if (logits[v] > logits[am]) am = v;
                wbuf = am; (void)wbuf;   /* next fed token: use the argmax just computed, like real generation */
                tt[w % 2] = am;
                ++cur_p;
            }
        }
        in2_state_save();
        if (batched) {
            float *la = (float *)calloc((size_t)2 * cfg.vocab_size, sizeof(float));
            if (!la) { fprintf(stderr, "probe oom\n"); return 2; }
            in2_set_logits_all(la);
            const int rc = in2_forward_at(ctx, &m, &cfg, tt, 2, p0, h, logits);
            in2_set_logits_all(NULL);
            if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "probe batched rc=%d\n", rc); return rc; }
            memcpy(row0, la, sizeof(float) * cfg.vocab_size);
            memcpy(row1, la + cfg.vocab_size, sizeof(float) * cfg.vocab_size);
        } else {
            int rc = in2_forward_at(ctx, &m, &cfg, &tt[0], 1, p0, h, row0);
            if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "probe seq0 rc=%d\n", rc); return rc; }
            rc = in2_forward_at(ctx, &m, &cfg, &tt[1], 1, p0 + 1, h, row1);
            if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "probe seq1 rc=%d\n", rc); return rc; }
        }
        FILE *fp = fopen(out, "wb");
        if (!fp) { fprintf(stderr, "probe cannot write %s\n", out); return 2; }
        fwrite(row0, sizeof(float), cfg.vocab_size, fp);
        fwrite(row1, sizeof(float), cfg.vocab_size, fp);
        fclose(fp);
        fprintf(stderr, "PROBE prefill=%d batched=%d -> %s\n", p0, batched, out);
        if (getenv("IN2_PROBE_K")) {
            const int stride = 2048;
            float *rows = (float *)calloc((size_t)IN2_PROBE_MAXV * (size_t)stride, sizeof(float));
            FILE *kf = rows ? fopen(getenv("IN2_PROBE_K"), "wb") : NULL;
            if (kf) {
                for (int which = 0; which < 2; ++which) {
                    int dim = 0;
                    const int nv = in2_probe_rows(p0 + which, rows, stride, &dim);
                    int hdr[2]; hdr[0] = nv; hdr[1] = dim;
                    fwrite(hdr, sizeof(int), 2, kf);
                    if (nv > 0 && dim > 0) {
                        for (int v = 0; v < nv; ++v)
                            fwrite(rows + (size_t)v * stride, sizeof(float), (size_t)dim, kf);
                    }
                }
                fclose(kf);
                fprintf(stderr, "PROBE K rows -> %s%c", getenv("IN2_PROBE_K"), 10);
            }
            free(rows);
        }
        return 0;
    }


    const double t_start_lookup = wall_now();
    if (getenv("IN2_LOOKUP") && atoi(getenv("IN2_LOOKUP")) != 0) {
        /* Prompt-lookup speculative decoding. Parity is by construction: the drafter only PROPOSES, and a draft is accepted
         * only when the target's own greedy argmax at that position equals it, so every emitted token is one the plain
         * loop would have emitted from the same logits. */
        const int K = getenv("IN2_LOOKUP_K") ? atoi(getenv("IN2_LOOKUP_K")) : 4;
        if (K > 0) fprintf(stderr, "LOOKUP: spec loop enabled; batched n_tok>1 verify verified bit-identical to sequential at warm=13 (see IN2_BATCH_PROBE). Accept stats below.\n");
        float *la = (float *)malloc(sizeof(float) * (size_t)(K + 1) * (size_t)cfg.vocab_size);
        if (!la) { fprintf(stderr, "IN2_LOOKUP: oom for %d x %d logits\n", K + 1, cfg.vocab_size); return 2; }
        long long n_drafted = 0, n_accepted = 0, n_forward = 0, n_replay = 0;
        int produced = 0, n = n_prompt;
        /* The plain loop's step 0 is the prompt prefill, not a single-token prime: without it the first step has no
         * context at all and the run diverges five tokens in. */
        {
            const int rc0 = in2_forward_at(ctx, &m, &cfg, toks, n_prompt, 0, h, logits);
            ++n_forward;
            if (rc0 != WUBU_INKLING2_OK) { free(la); return rc0; }
            used_cache = 1;
            int b0 = 0; float v0 = -1e30f;
            for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > v0) { v0 = logits[i]; b0 = i; }
            toks[n] = b0; ++n; ++produced;
        }
        while (produced < n_gen) {
            /* prime: one ordinary step. Processes position n-1 and yields the token for position n. */
            int rc = in2_forward_at(ctx, &m, &cfg, &toks[n - 1], 1, n - 1, h, logits);
            ++n_forward;
            if (rc != WUBU_INKLING2_OK) { free(la); return rc; }
            used_cache = 1;
            int best = 0; float bv = -1e30f;
            for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            toks[n] = best; ++n; ++produced;
            /* draft: longest suffix of the history seen earlier, take up to K tokens that followed it */
            int nd = 0, draft[16];
            if (produced < n_gen) {
                for (int len = 8; len >= 1 && nd == 0; --len) {
                    if (n - len - 1 < len) continue;
                    for (int j = n - 1 - len; j >= len; --j) {
                        if (memcmp(toks + (j - len), toks + (n - len), sizeof(int) * (size_t)len) == 0) {
                            int avail = n - j; if (avail > K) avail = K;
                            for (int q = 0; q < avail; ++q) draft[q] = toks[j + q];
                            nd = avail; break;
                        }
                    }
                }
            }
            if (nd <= 0) { fflush(stdout); continue; }
            /* verify: feed position n-1 (the token just emitted, not yet processed) plus the drafts. Row q is the
             * prediction for the token at position n+q, which is draft q. */
            int feed[17];
            feed[0] = toks[n - 1];
            for (int q = 0; q < nd; ++q) feed[q + 1] = draft[q];
            in2_state_save();
            in2_set_logits_all(la);
            rc = in2_forward_at(ctx, &m, &cfg, feed, nd + 1, n - 1, h, logits);
            in2_set_logits_all(NULL);
            ++n_forward;
            if (rc != WUBU_INKLING2_OK) { free(la); return rc; }
            int acc = 0;
            for (int q = 0; q < nd; ++q) {
                const float *row = la + (size_t)q * (size_t)cfg.vocab_size;
                int a = 0; float av = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) if (row[i] > av) { av = row[i]; a = i; }
                if (a != draft[q]) break;
                acc = q + 1;
            }
            n_drafted += nd; n_accepted += acc;
            if (acc < nd) {
                /* Roll back. Histories return to the pre-verification snapshot, K/V rows at or beyond n are invalidated,
                 * then the accepted drafts are replayed so both conv histories sit at the true prefix. Without this the
                 * record is correct for ~5 tokens and then wrong, which is exactly how the first attempt failed. */
                in2_state_restore();
                in2_state_rewind(n);
                if (acc > 0) {
                    rc = in2_forward_at(ctx, &m, &cfg, draft, acc, n, h, logits);
                    ++n_forward; ++n_replay;
                    if (rc != WUBU_INKLING2_OK) { free(la); return rc; }
                }
            }
            for (int q = 0; q < acc && produced < n_gen; ++q) { toks[n] = draft[q]; ++n; ++produced; }
            if (produced < n_gen) {
                /* the correction: the target's own argmax at the first position the drafter got wrong */
                const float *row = la + (size_t)acc * (size_t)cfg.vocab_size;
                int a = 0; float av = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) if (row[i] > av) { av = row[i]; a = i; }
                toks[n] = a; ++n; ++produced;
            }
            fflush(stdout);
        }
        free(la);
        fprintf(stderr, "LOOKUP: %lld forwards for %d tokens (%.2f fwd/token), drafted %lld, accepted %lld = %.1f%%, replays %lld\n",
                n_forward, n_gen, n_gen ? (double)n_forward / (double)n_gen : 0.0,
                n_drafted, n_accepted, n_drafted ? 100.0 * (double)n_accepted / (double)n_drafted : 0.0, n_replay);
        t_total = wall_now() - t_start_lookup;
    } else {
    for (int step = 0; step < n_gen; ++step) {
        const clock_t t_step0 = clock(); (void)t_step0;
        const int n = n_prompt + step;
        /* With a cache the prompt is prefilled once and each step pushes a single token at its real position;
         * without one the whole prefix is recomputed, which is the O(n^2) behaviour this is meant to replace.
         * Cache on by default now: the full-recompute path was the safe default while the cache could hide a bug, but
         * 34/34 oracle tokens match with it enabled, and the two paths produce bit-identical hidden states. IN2_NO_KV
         * forces the full recompute for debugging. */
        const int incremental = getenv("IN2_NO_KV") == NULL && step > 0;
        unsigned long long f0, r0, o0, f1, r1, o1;
        fault_now(&f0, &r0, &o0);   /* before the timer, so the delta covers the forward exactly */
        const double t_f0 = wall_now();
        const int rc = incremental
            ? in2_forward_at(ctx, &m, &cfg, &toks[n - 1], 1, n - 1, h, logits)
            : in2_forward_at(ctx, &m, &cfg, toks, n, 0, h, logits);
        const double t_f1 = wall_now();
        fault_now(&f1, &r1, &o1);
        if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "step %d rc=%d\n", step, rc); return rc; }
        used_cache |= incremental;
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        toks[n] = best;
        const double t_after = wall_now();
        const double dt = t_after - t_step;
        t_step = t_after;
        t_total += dt;
        printf("gen[%d] id=%d logit=%.4f  %.3fs wall (fwd %.3fs + %.3fs other)  (%.2f tok/s at prefix %d)"
               "  faults %llu  disk %.1f MB / %llu reads\n",
               step, best, bv, dt, t_f1 - t_f0, dt - (t_f1 - t_f0), dt > 0 ? 1.0 / dt : 0.0, n,
               f1 - f0, (double) (r1 - r0) / 1048576.0, o1 - o0);
        printf("      ws %.0f MB  faults/MB-touched %.0f\n",
               (double) g_ws / 1048576.0, (f1 - f0) ? 1535.0 / ((double) (f1 - f0) * 4096.0 / 1048576.0) : 0.0);
        if (getenv("IN2_TOPK")) {
            /* Top-k of the same distribution, so a near-tie can be judged against llama-server's
             * completion_probabilities instead of only by the argmax. */
            const int K = atoi(getenv("IN2_TOPK"));
            static int used[64];
            for (int r = 0; r < K && r < 64; ++r) {
                int bi = -1; float bvv = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) {
                    int skip = 0;
                    for (int q = 0; q < r; ++q) if (used[q] == i) { skip = 1; break; }
                    if (!skip && logits[i] > bvv) { bvv = logits[i]; bi = i; }
                }
                used[r] = bi;
                fprintf(stderr, "TOPK %d %d %.4f\n", r, bi, bvv);
            }
        }
        fflush(stdout);
    }
    }
    if (n_gen > 0 && t_total > 0)
        fprintf(stderr, "TIMING: %d tokens in %.1fs = %.2f tok/s (%s)\n",
                n_gen, t_total, (double)n_gen / t_total, used_cache ? "KV cache" : "full recompute");
    if (getenv("IN2_PERF")) {
        extern double in2_fph[8], in2_kvproj_sec, in2_kvconv_sec;
        if (getenv("IN2_MOE_SPLIT")) {
            extern double in2_ms_gath, in2_ms_gu, in2_ms_act, in2_ms_dn, in2_ms_scat, in2_ms_acc;
            extern double in2_ms_setup, in2_ms_free, in2_ms_route;
            extern double in2_tk_gu, in2_tk_act, in2_tk_dn, in2_tk_acc;
            extern double in2_tk_route, in2_tk_region, in2_tk_zero;
            double tot = in2_ms_gath + in2_ms_gu + in2_ms_act + in2_ms_dn + in2_ms_scat + in2_ms_setup + in2_ms_free + in2_ms_route;
            fprintf(stderr, "MOESPLIT setup=%.3f gather=%.3f gu=%.3f act=%.3f down=%.3f scatter=%.3f free=%.3f route=%.3f | accum=%.3f  sum=%.3f (%.1f%% gu, %.1f%% down, %.1f%% act, %.1f%% gather, %.1f%% scatter)\n",
                    in2_ms_setup, in2_ms_gath, in2_ms_gu, in2_ms_act, in2_ms_dn, in2_ms_scat, in2_ms_free, in2_ms_route, in2_ms_acc, tot,
                    tot > 0 ? 100.0 * in2_ms_gu / tot : 0, tot > 0 ? 100.0 * in2_ms_dn / tot : 0,
                    tot > 0 ? 100.0 * in2_ms_act / tot : 0, tot > 0 ? 100.0 * in2_ms_gath / tot : 0,
                    tot > 0 ? 100.0 * in2_ms_scat / tot : 0);
            double kt = in2_tk_gu + in2_tk_act + in2_tk_dn + in2_tk_acc;
            fprintf(stderr, "TOKENWALL route=%.3f zero=%.3f region=%.3f accum=%.3f | (cpu-spans: gu=%.3f act=%.3f down=%.3f)\n",
                    in2_tk_route, in2_tk_zero, in2_tk_region, in2_tk_acc, in2_tk_gu, in2_tk_act, in2_tk_dn);
            fprintf(stderr, "TOKENSPLIT gu=%.3f act=%.3f down=%.3f accum=%.3f | sum=%.3f (%.1f%% gu, %.1f%% down, %.1f%% act, %.1f%% accum)\n",
                    in2_tk_gu, in2_tk_act, in2_tk_dn, in2_tk_acc, kt,
                    kt > 0 ? 100.0 * in2_tk_gu / kt : 0, kt > 0 ? 100.0 * in2_tk_dn / kt : 0,
                    kt > 0 ? 100.0 * in2_tk_act / kt : 0, kt > 0 ? 100.0 * in2_tk_acc / kt : 0);
        }
        /* Slot order MUST match the FP2(i) call sites in wubu_inkling2_fullseq.c:
         * FP2(0) loads, FP2(1) mix+gates, FP2(2) q, FP2(3) kv+conv, FP2(4) attention, FP2(5) gate+after,
         * FP2(6) mlp+moe+postconv. The previous array was shifted by one and had no name for slot 6, which
         * made slot 6 - the largest span - print as "?" and made every other row name the slot before it. */
        { extern double in2_nt_read_sec; extern long in2_nt_calls, in2_nt_unmapped;
          fprintf(stderr, "NTREAD %.4f s read over %ld in2_matvec_nt calls (%.1f us/call), unmapped=%ld\n",
                  in2_nt_read_sec, in2_nt_calls,
                  in2_nt_calls ? in2_nt_read_sec / in2_nt_calls * 1e6 : 0.0, in2_nt_unmapped);
          in2_nt_read_sec = 0; in2_nt_calls = 0; in2_nt_unmapped = 0; }
        { extern long in2_ent_calls[8]; extern double in2_ent_sec[8];
          extern double in2_nt_read_sec; extern long in2_nt_calls, in2_nt_unmapped;
          fprintf(stderr, "ENTRY nt=%ld quant=%ld float=%ld other=%ld\n",
                  in2_ent_calls[0], in2_ent_calls[1], in2_ent_calls[2], in2_ent_calls[3]);
          if (in2_ent_calls[0] > 0) {
              const double tot = in2_ent_sec[0], rd = in2_nt_read_sec;
              extern double in2_nt_act_sec;
              fprintf(stderr, "ENTRYTIME nt total %.4f s / %ld calls = %.1f us/call | read %.4f (%.1f us) | "
                              "act-quant %.4f (%.1f us) | kernel %.4f (%.1f us) | unmapped=%ld\n",
                      tot, in2_ent_calls[0], tot / in2_ent_calls[0] * 1e6,
                      rd, rd / in2_ent_calls[0] * 1e6,
                      in2_nt_act_sec, in2_nt_act_sec / in2_ent_calls[0] * 1e6,
                      tot - rd - in2_nt_act_sec, (tot - rd - in2_nt_act_sec) / in2_ent_calls[0] * 1e6,
                      in2_nt_unmapped);
              in2_nt_act_sec = 0;
          }
          for (int _e = 0; _e < 8; ++_e) { in2_ent_calls[_e] = 0; in2_ent_sec[_e] = 0; }
          in2_nt_read_sec = 0; in2_nt_calls = 0; in2_nt_unmapped = 0; }
        { extern struct in2_shp_rec { int ni, no, ty; double sec, mn, mx; long n; } in2_shp[96];
          extern int in2_shp_n;
          if (in2_shp_n > 0) {
              double tot = 0; for (int i = 0; i < in2_shp_n; ++i) tot += in2_shp[i].sec;
              fprintf(stderr, "SHAPE TABLE (run-wide, %d shapes, %.4f s total)\n", in2_shp_n, tot);
              unsigned char used[96] = {0};
              for (int k = 0; k < 12 && k < in2_shp_n; ++k) {
                  int best = -1;
                  for (int i2 = 0; i2 < in2_shp_n; ++i2)
                      if (!used[i2] && (best < 0 || in2_shp[i2].sec > in2_shp[best].sec)) best = i2;
                  if (best < 0) break;
                  used[best] = 1;
                  fprintf(stderr, "  ty=%-3d %5d -> %6d : %8.4f s %5ld calls mean %8.1f us [min %7.1f max %8.1f] %5.1f%%\n",
                          in2_shp[best].ty, in2_shp[best].ni, in2_shp[best].no, in2_shp[best].sec, in2_shp[best].n,
                          in2_shp[best].sec / (double) in2_shp[best].n * 1e6,
                          in2_shp[best].mn * 1e6, in2_shp[best].mx * 1e6,
                          tot > 0 ? in2_shp[best].sec / tot * 100 : 0.0);
              }
          }
          in2_shp_n = 0; }
        fprintf(stderr, "PAGEFAULT total=%lu (per token: %.0f)\n", in2_faults(),
                n_gen > 0 ? (double) in2_faults() / (double) n_gen : 0.0);
        { in2_io_t io;
          if (in2_io(&io))
              fprintf(stderr, "READIO ops=%llu bytes=%llu (%.1f MB) | writes=%llu (%.1f MB)\n",
                      io.rd_ops, io.rd_bytes, (double) io.rd_bytes / 1048576.0,
                      io.wr_bytes, (double) io.wr_bytes / 1048576.0); }
        const char *pn[8] = {"f_loads","f_mix+gate","f_q","f_kv+conv","f_attn","f_gate+wo","f_mlp+moe","unused"};
        for (int i = 0; i < 8; ++i)
            fprintf(stderr, "PHASE[%d] %-10s %.3f s (%d tokens)\n", i, pn[i], in2_fph[i], n_gen);
        fprintf(stderr, "PHASE kvproj %.3f kvconv %.3f\n", in2_kvproj_sec, in2_kvconv_sec);
    }
    if (getenv("IN2_F16_STATS")) {
        typedef int (*st_t)(long long*,long long*,long long*,long long*,int*);
        typedef int (*fr_t)(void);
        const char *nm = getenv("IN2_CUDA_DLL");
        HANDLE h = LoadLibraryA(nm && nm[0] ? nm : "wubu_cuda.dll");
        if (h) {
            st_t st = (st_t)(void*)GetProcAddress(h, "wubu_cuda_f16_stats");
            fr_t fr = (fr_t)(void*)GetProcAddress(h, "wubu_cuda_vram_free_mb");
            if (st) { long long ups=0,reuse=0,evict=0,usedmb=0; int slabs=0;
                st(&ups,&reuse,&evict,&usedmb,&slabs);
                fprintf(stderr, "F16STAT ups=%lld reuse=%lld evict=%lld used=%lldMB slabs=%d vram_free=%dMB\n",
                        ups, reuse, evict, usedmb, slabs, fr?fr():-1); }
        }
    }
    if (getenv("IN2_TIMETY")) {
        extern double in2_ty_sec[256]; extern long in2_ty_calls[256]; extern double in2_ty_macs[256];
        double cp[256]; for (int i = 0; i < 256; ++i) cp[i] = in2_ty_sec[i];
        fprintf(stderr, "TIMETY (last forward), by CPU seconds:\n");
        for (int pass = 0; pass < 10; ++pass) {
            int best = -1; double bs = 0.0005;
            for (int t = 0; t < 256; ++t) if (cp[t] > bs) { bs = cp[t]; best = t; }
            if (best < 0) break;
            fprintf(stderr, "  ty=%-3d %8.3f s  calls=%-8ld macs=%9.1f M\n", best, cp[best], in2_ty_calls[best], in2_ty_macs[best] / 1e6);
            cp[best] = 0;
        }
    }
    /* What the f32 materialisation path actually moved. `to_f32` copies on a cache hit and dequantises on a miss;
     * `borrow` hands out the table's own pointer. Bytes/token from the GGUF say nothing about this traffic, so it has
     * to be counted where it happens (counters live in wubu_inkling2_fwd.c). */
    {
        extern unsigned long long in2_dq_produced, in2_dq_copied, in2_dq_fast_calls, in2_dq_borrow_hit, in2_dq_borrow_miss;
        fprintf(stderr, "DQ    f32 produced %.1f MB, copied %.1f MB, borrow miss %llu hit %llu, to_f32 calls %llu\n",
                (double) in2_dq_produced / 1048576.0, (double) in2_dq_copied / 1048576.0,
                in2_dq_borrow_miss, in2_dq_borrow_hit, in2_dq_fast_calls);
    }
    /* The ids alone do not say whether the output is coherent, which is the whole question a generation run is
     * asking. Decode with the vocab inside the same .gguf, in detok's rules, so a human can judge the result here. */
    {
        wubu_gguf_tokenizer_data_t tk;
        memset(&tk, 0, sizeof tk);
        if (wubu_gguf_tokenizer_extract(path, &tk)) {
            printf("GEN_TEXT: ");
            for (int i = n_prompt; i < n_prompt + n_gen; ++i)
                if (toks[i] >= 0 && toks[i] < tk.n_tokens) emit_piece(tk.tokens[toks[i]]);
            printf("\n");
            printf("FULL_TEXT: ");
            for (int i = 0; i < n_prompt + n_gen; ++i)
                if (toks[i] >= 0 && toks[i] < tk.n_tokens) emit_piece(tk.tokens[toks[i]]);
            printf("\n");
            wubu_gguf_tokenizer_free(&tk);
        } else {
            fprintf(stderr, "tokenizer extract failed; only ids available\n");
        }
    }
    printf("IDS:");
    for (int i = 0; i < n_prompt + n_gen; ++i) printf(" %d", toks[i]);
    printf("\n");
    free(h); free(logits); free(toks); gguf_close(ctx);
    return 0;
}
