/*
 * wubu_admission.c -- EG-10 shared admission validator. Pure C11.
 */
#include "wubu_admission.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    uint64_t tokens;
    int admitted_step;
    int live;
} admission_req_t;

struct wubu_admission {
    uint64_t budget;
    uint64_t reserved;
    admission_req_t *reqs;
    int max_queue;
    int timeout_steps;
    uint64_t accepted, rejected, expired;
};

wubu_admission_t *wubu_admission_create(uint64_t token_budget,
                                         int max_queue, int timeout_steps) {
    wubu_admission_t *a;
    if (token_budget == 0 || max_queue <= 0 || timeout_steps < 0) return NULL;
    a = (wubu_admission_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->reqs = (admission_req_t *)calloc((size_t)max_queue,
                                        sizeof(*a->reqs));
    if (!a->reqs) { free(a); return NULL; }
    for (int i = 0; i < max_queue; i++) a->reqs[i].id = -1;
    a->budget = token_budget;
    a->max_queue = max_queue;
    a->timeout_steps = timeout_steps;
    return a;
}

void wubu_admission_free(wubu_admission_t *a) {
    if (!a) return;
    free(a->reqs);
    free(a);
}

static admission_req_t *find_req(wubu_admission_t *a, int id) {
    for (int i = 0; i < a->max_queue; i++)
        if (a->reqs[i].live && a->reqs[i].id == id) return &a->reqs[i];
    return NULL;
}

wubu_admit_status_t wubu_admission_validate(wubu_admission_t *a, int req_id,
                                              uint64_t tokens, int now_step) {
    admission_req_t *slot = NULL;
    wubu_admit_status_t st;
    if (!a || req_id < 0 || tokens == 0 || now_step < 0)
        return WUBU_ADMIT_ERR_ARG;
    if (find_req(a, req_id)) { st = WUBU_ADMIT_DUPLICATE; goto reject; }
    if (a->reserved + tokens > a->budget) { st = WUBU_ADMIT_OVER_BUDGET; goto reject; }
    for (int i = 0; i < a->max_queue; i++) {
        if (!a->reqs[i].live) { slot = &a->reqs[i]; break; }
    }
    if (!slot) { st = WUBU_ADMIT_QUEUE_FULL; goto reject; }
    slot->id = req_id;
    slot->tokens = tokens;
    slot->admitted_step = now_step;
    slot->live = 1;
    a->reserved += tokens;
    a->accepted++;
    return WUBU_ADMIT_OK;

reject:
    a->rejected++;
    return st;
}

int wubu_admission_release(wubu_admission_t *a, int req_id) {
    admission_req_t *r;
    if (!a) return -1;
    r = find_req(a, req_id);
    if (!r) return -1;
    a->reserved -= r->tokens;
    r->live = 0;
    r->id = -1;
    return 0;
}

int wubu_admission_step(wubu_admission_t *a, int now_step) {
    int expired = 0;
    if (!a || now_step < 0) return 0;
    for (int i = 0; i < a->max_queue; i++) {
        admission_req_t *r = &a->reqs[i];
        if (!r->live) continue;
        if (now_step - r->admitted_step > a->timeout_steps) {
            a->reserved -= r->tokens;
            r->live = 0;
            r->id = -1;
            expired++;
            a->expired++;
        }
    }
    return expired;
}

uint64_t wubu_admission_reserved(const wubu_admission_t *a) {
    return a ? a->reserved : 0;
}

int wubu_admission_queue_len(const wubu_admission_t *a) {
    int n = 0;
    if (!a) return -1;
    for (int i = 0; i < a->max_queue; i++) if (a->reqs[i].live) n++;
    return n;
}

uint64_t wubu_admission_accepted(const wubu_admission_t *a) {
    return a ? a->accepted : 0;
}

uint64_t wubu_admission_rejected(const wubu_admission_t *a) {
    return a ? a->rejected : 0;
}

uint64_t wubu_admission_expired(const wubu_admission_t *a) {
    return a ? a->expired : 0;
}
