/*
 * wubu_thread_spec.c — CPU thread-specialization analog (doc H02).
 * Two pinned thread pools (prefill / decode). See header. Self-contained C11.
 */
#define WUBU_HOSTED
#include "wubu_gnu_compat.h"
#include "wubu_thread_spec.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

#if defined(__linux__)
#include <unistd.h>
#endif

typedef struct {
    wubu_ts_job_fn fn;
    void *arg;
    int active;
} ts_job_t;

typedef struct {
    pthread_t thread;
    int core;
    pthread_mutex_t m;
    pthread_cond_t cv;
    int stop;
    ts_job_t job;          /* one in-flight job at a time (simple, correct) */
    int has_job;
    int done;
} ts_worker_t;

struct wubu_thread_spec {
    ts_worker_t *workers;
    int n_workers;
    int n_prefill;
    int n_decode;
};

/* Parse "0-3" or "4,5,6,7" into a core list. Returns count written. */
static int parse_cores(const char *spec, int *out, int maxout) {
    if (!spec || !*spec) return 0;
    int n = 0;
    const char *p = spec;
    int lo = -1;            /* pending range start, or -1 */
    while (*p && n < maxout) {
        if (*p == '-') {
            int hi = 0; p++;
            while (*p >= '0' && *p <= '9') { hi = hi * 10 + (*p - '0'); p++; }
            if (lo >= 0) { for (int c = lo; c <= hi && n < maxout; c++) out[n++] = c; lo = -1; }
        } else if (*p >= '0' && *p <= '9') {
            int v = 0;
            while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
            if (lo >= 0) {
                /* v is part of a range already emitted; ignore */
                lo = -1;
            } else {
                /* standalone number — but could be followed by '-', so defer:
                 * if next non-digit char is '-', treat v as range start. */
                if (p[0] == '-') {
                    lo = v;
                } else {
                    out[n++] = v;
                }
            }
        } else {
            p++;
        }
    }
    return n;
}

static void *worker_main(void *arg) {
    ts_worker_t *w = (ts_worker_t *)arg;
    for (;;) {
        pthread_mutex_lock(&w->m);
        while (!w->has_job && !w->stop)
            pthread_cond_wait(&w->cv, &w->m);
        if (w->stop && !w->has_job) { pthread_mutex_unlock(&w->m); break; }
        ts_job_t job = w->job;
        w->has_job = 0;
        w->done = 0;
        pthread_mutex_unlock(&w->m);

        if (job.fn) job.fn(job.arg);

        pthread_mutex_lock(&w->m);
        w->done = 1;
        pthread_cond_broadcast(&w->cv);
        pthread_mutex_unlock(&w->m);
    }
    return NULL;
}

static void pin_core(int core) {
#if defined(__linux__)
    if (core < 0) return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
}

wubu_thread_spec_t *wubu_thread_spec_create(const char *prefill_cores,
                                            const char *decode_cores) {
    int pcores[256], dcores[256];
    int np = parse_cores(prefill_cores, pcores, 256);
    int nd = parse_cores(decode_cores, dcores, 256);
    if (np == 0 && nd == 0) { np = 1; pcores[0] = -1; nd = 1; dcores[0] = -1; }
    if (np == 0) { np = 1; pcores[0] = -1; }
    if (nd == 0) { nd = 1; dcores[0] = -1; }

    wubu_thread_spec_t *ts = (wubu_thread_spec_t *)calloc(1, sizeof(*ts));
    if (!ts) return NULL;
    ts->n_prefill = np;
    ts->n_decode = nd;
    ts->n_workers = np + nd;
    ts->workers = (ts_worker_t *)calloc(ts->n_workers, sizeof(ts_worker_t));
    if (!ts->workers) { free(ts); return NULL; }

    for (int i = 0; i < ts->n_workers; i++) {
        ts_worker_t *w = &ts->workers[i];
        w->core = (i < np) ? pcores[i] : dcores[i - np];
        w->stop = 0; w->has_job = 0; w->done = 1;
        pthread_mutex_init(&w->m, NULL);
        pthread_cond_init(&w->cv, NULL);
        pthread_create(&w->thread, NULL, worker_main, w);
        pin_core(w->core);
    }
    return ts;
}

void wubu_thread_spec_free(wubu_thread_spec_t *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->n_workers; i++) {
        ts_worker_t *w = &ts->workers[i];
        pthread_mutex_lock(&w->m);
        w->stop = 1;
        pthread_cond_broadcast(&w->cv);
        pthread_mutex_unlock(&w->m);
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->m);
        pthread_cond_destroy(&w->cv);
    }
    free(ts->workers);
    free(ts);
}

int wubu_thread_spec_submit(wubu_thread_spec_t *ts, wubu_ts_role_t role,
                            wubu_ts_job_fn fn, void *arg) {
    if (!ts || !fn) return -1;
    int start = (role == WUBU_TS_DECODE) ? ts->n_prefill : 0;
    int count = (role == WUBU_TS_DECODE) ? ts->n_decode : ts->n_prefill;
    /* Round-robin over the role's workers. */
    static int rr_prefill = 0, rr_decode = 0;
    int pick;
    if (role == WUBU_TS_DECODE) pick = start + (rr_decode++ % count);
    else                        pick = start + (rr_prefill++ % count);

    ts_worker_t *w = &ts->workers[pick];
    pthread_mutex_lock(&w->m);
    while (w->has_job) pthread_cond_wait(&w->cv, &w->m);
    w->job.fn = fn;
    w->job.arg = arg;
    w->has_job = 1;
    w->done = 0;
    pthread_cond_broadcast(&w->cv);
    pthread_mutex_unlock(&w->m);
    return 0;
}

void wubu_thread_spec_wait(wubu_thread_spec_t *ts, wubu_ts_role_t role) {
    if (!ts) return;
    int start = (role == WUBU_TS_DECODE) ? ts->n_prefill : 0;
    int count = (role == WUBU_TS_DECODE) ? ts->n_decode : ts->n_prefill;
    for (int i = 0; i < count; i++) {
        ts_worker_t *w = &ts->workers[start + i];
        pthread_mutex_lock(&w->m);
        while (w->has_job) pthread_cond_wait(&w->cv, &w->m);
        pthread_mutex_unlock(&w->m);
    }
}

void wubu_thread_spec_cores(const wubu_thread_spec_t *ts,
                            int *prefill_n, int *decode_n) {
    if (prefill_n) *prefill_n = ts ? ts->n_prefill : 0;
    if (decode_n)  *decode_n  = ts ? ts->n_decode  : 0;
}
