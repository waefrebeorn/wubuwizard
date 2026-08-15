/*
 * wubu_tandem.c — N64 RCP-style tandem pipeline (doc "tandem").
 * Two stages (A=prefill/RSP, B=decode/RDP) run in tandem over a ring handoff.
 * Self-contained C11 + POSIX threads.
 */
#define WUBU_HOSTED
#include "wubu_gnu_compat.h"
#include "wubu_tandem.h"
#include "wubu_hwcaps.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

#if defined(__linux__)
#include <unistd.h>
#endif

typedef struct {
    int        frame;
    void      *arg;
    int        stage_done[2];  /* [0]=A done, [1]=B done */
    int        in_use;
} tandem_frame_t;

struct wubu_tandem {
    int n_a, n_b;
    wubu_tandem_fn fn_a, fn_b;

    /* Ring of frames handed A->B. Simplest correct model: a single in-flight
     * frame; A produces, B consumes. Ping-pong: while B runs frame N, A builds
     * N+1 into the spare slot. ring=2 gives true overlap. */
    tandem_frame_t *ring;
    int ring_sz;

    pthread_mutex_t m;
    pthread_cond_t  cv_a;   /* A waits for a free slot */
    pthread_cond_t  cv_b;   /* B waits for a ready slot */
    int produced;           /* frames A has filled, not yet consumed */
    int consumed;           /* frames B has finished */
    int next_frame;
    int stop;

    uint64_t frames, a_busy, b_busy;

    /* worker threads */
    pthread_t *a_threads, *b_threads;
    int n_a_t, n_b_t;
    int *a_cores_buf, *b_cores_buf;
    int a_idx, b_idx;
};

static void pin(int core) {
#if defined(__linux__)
    if (core < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
}

/* Parse core list like "0-3" / "4,5,6,7"; returns count. */
static int parse_cores(const char *s, int *out, int max) {
    if (!s || !*s) return 0;
    int n = 0, lo = -1; const char *p = s;
    while (*p && n < max) {
        if (*p == '-') {
            int hi = 0; p++;
            while (*p >= '0' && *p <= '9') { hi = hi*10 + (*p-'0'); p++; }
            if (lo >= 0) { for (int c = lo; c <= hi && n < max; c++) out[n++] = c; lo = -1; }
        } else if (*p >= '0' && *p <= '9') {
            int v = 0; while (*p >= '0' && *p <= '9') { v = v*10 + (*p-'0'); p++; }
            if (p[0] == '-') lo = v; else out[n++] = v;
        } else p++;
    }
    return n;
}

static void *stage_a(void *arg) {
    wubu_tandem_t *t = (wubu_tandem_t *)arg;
    pin(t->n_a_t > 0 ? t->a_cores_buf[t->a_idx++ % t->n_a_t] : -1);
    for (;;) {
        pthread_mutex_lock(&t->m);
        /* Wait for a frame that A hasn't processed yet. */
        while (!(t->produced > t->consumed && !t->ring[t->consumed % t->ring_sz].stage_done[0])
               && !t->stop)
            pthread_cond_wait(&t->cv_a, &t->m);
        if (t->stop && !(t->produced > t->consumed)) {
            pthread_mutex_unlock(&t->m); break;
        }
        int slot = t->consumed % t->ring_sz;
        void *a = t->ring[slot].arg;
        int frame = t->ring[slot].frame;
        pthread_mutex_unlock(&t->m);

        /* Run A stage work */
        if (t->fn_a) t->fn_a(a, 0, frame);
        t->a_busy++;

        pthread_mutex_lock(&t->m);
        t->ring[slot].stage_done[0] = 1;
        pthread_cond_broadcast(&t->cv_b);
        pthread_mutex_unlock(&t->m);
    }
    return NULL;
}

static void *stage_b(void *arg) {
    wubu_tandem_t *t = (wubu_tandem_t *)arg;
    pin(t->n_b_t > 0 ? t->b_cores_buf[t->b_idx++ % t->n_b_t] : -1);
    for (;;) {
        pthread_mutex_lock(&t->m);
        while (t->consumed >= t->produced && !t->stop)
            pthread_cond_wait(&t->cv_b, &t->m);
        if (t->stop && t->consumed >= t->produced) {
            pthread_mutex_unlock(&t->m); break;
        }
        int slot = t->consumed % t->ring_sz;
        /* Wait until A finished this frame */
        while (!t->ring[slot].stage_done[0] && !t->stop)
            pthread_cond_wait(&t->cv_b, &t->m);
        pthread_mutex_unlock(&t->m);

        if (t->fn_b) t->fn_b(t->ring[slot].arg, 1, t->ring[slot].frame);
        t->b_busy++;

        pthread_mutex_lock(&t->m);
        t->ring[slot].in_use = 0;
        t->consumed++;
        t->frames++;
        pthread_cond_broadcast(&t->cv_a);
        pthread_mutex_unlock(&t->m);
    }
    return NULL;
}

wubu_tandem_t *wubu_tandem_create(int n_a, int n_b,
                                  const char *a_cores, const char *b_cores,
                                  int ring) {
    if (n_a < 1) n_a = 1;
    if (n_b < 1) n_b = 1;
    if (ring < 2) ring = 2;

    wubu_tandem_t *t = (wubu_tandem_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->n_a = n_a; t->n_b = n_b;
    t->ring_sz = ring;
    t->ring = (tandem_frame_t *)calloc(ring, sizeof(tandem_frame_t));
    if (!t->ring) { free(t); return NULL; }

    pthread_mutex_init(&t->m, NULL);
    pthread_cond_init(&t->cv_a, NULL);
    pthread_cond_init(&t->cv_b, NULL);

    /* core buffers */
    int ac[256], bc[256];
    t->n_a_t = parse_cores(a_cores, ac, 256);
    t->n_b_t = parse_cores(b_cores, bc, 256);
    if (t->n_a_t > 0) { t->a_cores_buf = malloc(t->n_a_t * sizeof(int)); memcpy(t->a_cores_buf, ac, t->n_a_t*sizeof(int)); }
    if (t->n_b_t > 0) { t->b_cores_buf = malloc(t->n_b_t * sizeof(int)); memcpy(t->b_cores_buf, bc, t->n_b_t*sizeof(int)); }
    t->a_idx = 0; t->b_idx = 0;

    /* HW hint: if AVX512 present, note it (used by callers for GEMV path) */
    (void)wubu_hwcaps_get();

    t->a_threads = calloc(n_a, sizeof(pthread_t));
    t->b_threads = calloc(n_b, sizeof(pthread_t));
    for (int i = 0; i < n_a; i++) pthread_create(&t->a_threads[i], NULL, stage_a, t);
    for (int i = 0; i < n_b; i++) pthread_create(&t->b_threads[i], NULL, stage_b, t);
    return t;
}

void wubu_tandem_free(wubu_tandem_t *t) {
    if (!t) return;
    pthread_mutex_lock(&t->m);
    t->stop = 1;
    pthread_cond_broadcast(&t->cv_a);
    pthread_cond_broadcast(&t->cv_b);
    pthread_mutex_unlock(&t->m);
    for (int i = 0; i < t->n_a; i++) pthread_join(t->a_threads[i], NULL);
    for (int i = 0; i < t->n_b; i++) pthread_join(t->b_threads[i], NULL);
    free(t->a_threads); free(t->b_threads);
    free(t->a_cores_buf); free(t->b_cores_buf);
    free(t->ring);
    pthread_mutex_destroy(&t->m);
    pthread_cond_destroy(&t->cv_a);
    pthread_cond_destroy(&t->cv_b);
    free(t);
}

void wubu_tandem_set_a(wubu_tandem_t *t, wubu_tandem_fn fn) { if (t) t->fn_a = fn; }
void wubu_tandem_set_b(wubu_tandem_t *t, wubu_tandem_fn fn) { if (t) t->fn_b = fn; }

/* Block until all submitted frames have been fully consumed by stage B.
 * Guarantees t->frames is final before any stats read. */
void wubu_tandem_drain(wubu_tandem_t *t) {
    if (!t) return;
    pthread_mutex_lock(&t->m);
    while (t->consumed < t->produced && !t->stop)
        pthread_cond_wait(&t->cv_a, &t->m);
    pthread_mutex_unlock(&t->m);
}

int wubu_tandem_submit(wubu_tandem_t *t, void *arg) {
    if (!t) return -1;
    /* Synchronous submit: wait for a free slot, fill it, wait for B to finish. */
    pthread_mutex_lock(&t->m);
    while (t->produced - t->consumed >= t->ring_sz - 1 && !t->stop)
        pthread_cond_wait(&t->cv_a, &t->m);
    if (t->stop) { pthread_mutex_unlock(&t->m); return -1; }
    int slot = t->produced % t->ring_sz;
    int frame = t->next_frame++;
    t->ring[slot].in_use = 1;
    t->ring[slot].arg = arg;
    t->ring[slot].frame = frame;
    t->ring[slot].stage_done[0] = 0;
    t->ring[slot].stage_done[1] = 0;
    t->produced++;
    pthread_mutex_unlock(&t->m);

    /* A runs in its own pool (stage_a worker). Just signal A's pool and wait
     * for B to finish this frame. This gives true N64-style tandem overlap:
     * stage A (prefill) and stage B (decode) execute concurrently. */
    pthread_mutex_lock(&t->m);
    pthread_cond_broadcast(&t->cv_a);   /* wake a stage_a worker */
    pthread_cond_broadcast(&t->cv_b);   /* also safe if A already done */
    /* Wait for B to consume this frame */
    while (t->consumed <= frame && !t->stop)
        pthread_cond_wait(&t->cv_a, &t->m);
    int done = t->consumed > frame;
    pthread_mutex_unlock(&t->m);
    return done ? 0 : -1;
}

void wubu_tandem_stats(const wubu_tandem_t *t,
                       uint64_t *frames, uint64_t *a_busy, uint64_t *b_busy) {
    if (!t) { if (frames) *frames = 0; if (a_busy) *a_busy = 0; if (b_busy) *b_busy = 0; return; }
    /* Lock so we read a consistent snapshot (matches the B-thread writer). */
    pthread_mutex_lock((pthread_mutex_t *)&t->m);
    if (frames) *frames = t->frames;
    if (a_busy) *a_busy = t->a_busy;
    if (b_busy) *b_busy = t->b_busy;
    pthread_mutex_unlock((pthread_mutex_t *)&t->m);
}
