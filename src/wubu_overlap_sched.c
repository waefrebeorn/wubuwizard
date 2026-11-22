/*
 * wubu_overlap_sched.c -- EG-09 overlapping scheduler. Pure C11,
 * deterministic; the "decode" is a token counter and a deferred result.
 */
#include "wubu_overlap_sched.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    int remaining;
    int active;
    int wait;                  /* consecutive steps not admitted */
} overlap_req_t;

struct wubu_overlap_sched {
    overlap_req_t *reqs;
    int max_requests;
    int n_reqs;
    wubu_overlap_result_t *queue;
    int queue_capacity;
    int head, pending;
    uint64_t produced, drained;
    int steps;
    int max_wait;
    int max_batch;
    int rr;
};

wubu_overlap_sched_t *wubu_overlap_sched_create(int max_requests,
                                                 int queue_capacity,
                                                 int max_batch) {
    wubu_overlap_sched_t *s;
    if (max_requests <= 0 || queue_capacity <= 0 || max_batch <= 0)
        return NULL;
    s = (wubu_overlap_sched_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->reqs = (overlap_req_t *)calloc((size_t)max_requests, sizeof(*s->reqs));
    s->queue = (wubu_overlap_result_t *)calloc((size_t)queue_capacity,
                                               sizeof(*s->queue));
    if (!s->reqs || !s->queue) {
        wubu_overlap_sched_free(s);
        return NULL;
    }
    s->max_requests = max_requests;
    s->queue_capacity = queue_capacity;
    s->max_batch = max_batch > max_requests ? max_requests : max_batch;
    return s;
}

void wubu_overlap_sched_free(wubu_overlap_sched_t *s) {
    if (!s) return;
    free(s->reqs);
    free(s->queue);
    free(s);
}

int wubu_overlap_sched_add(wubu_overlap_sched_t *s, int req_id, int n_tokens) {
    if (!s || n_tokens <= 0 || s->n_reqs >= s->max_requests) return -1;
    for (int i = 0; i < s->n_reqs; i++)
        if (s->reqs[i].id == req_id) return -1;
    s->reqs[s->n_reqs].id = req_id;
    s->reqs[s->n_reqs].remaining = n_tokens;
    s->reqs[s->n_reqs].active = 1;
    s->n_reqs++;
    return 0;
}

int wubu_overlap_sched_step(wubu_overlap_sched_t *s,
                            wubu_overlap_result_t *out, int cap,
                            int *n_drained) {
    int active = 0;
    if (!s || (cap > 0 && !out) || !n_drained) return -1;
    *n_drained = 0;

    /* Phase 1: drain results produced by the PREVIOUS step. */
    while (s->pending > 0 && *n_drained < cap) {
        int idx = s->head;
        if (out) out[*n_drained] = s->queue[idx];
        s->head = (s->head + 1) % s->queue_capacity;
        s->pending--;
        (*n_drained)++;
        s->drained++;
    }

    /* Phase 2: admit up to max_batch active requests, round-robin. */
    for (int k = 0; k < s->n_reqs; k++) {
        overlap_req_t *r = &s->reqs[s->rr];
        s->rr = (s->rr + 1) % s->n_reqs;
        if (!r->active || r->remaining <= 0) continue;
        if (active >= s->max_batch) {
            r->wait++;
            if (r->wait > s->max_wait) s->max_wait = r->wait;
            continue;
        }
        if (s->pending >= s->queue_capacity) return -1;
        int tail = (s->head + s->pending) % s->queue_capacity;
        s->queue[tail].req_id = r->id;
        s->queue[tail].token = 1000 * r->id + s->steps;
        s->queue[tail].step = s->steps;
        s->pending++;
        s->produced++;
        r->remaining--;
        r->wait = 0;
        if (r->remaining == 0) r->active = 0;
        active++;
    }
    s->steps++;
    return active;
}

uint64_t wubu_overlap_sched_produced(const wubu_overlap_sched_t *s) {
    return s ? s->produced : 0;
}

uint64_t wubu_overlap_sched_drained(const wubu_overlap_sched_t *s) {
    return s ? s->drained : 0;
}

int wubu_overlap_sched_step_count(const wubu_overlap_sched_t *s) {
    return s ? s->steps : -1;
}

int wubu_overlap_sched_pending(const wubu_overlap_sched_t *s) {
    return s ? s->pending : -1;
}

int wubu_overlap_sched_active(const wubu_overlap_sched_t *s) {
    if (!s) return -1;
    int n = 0;
    for (int i = 0; i < s->n_reqs; i++) if (s->reqs[i].active) n++;
    return n;
}

int wubu_overlap_sched_max_wait(const wubu_overlap_sched_t *s) {
    return s ? s->max_wait : -1;
}
