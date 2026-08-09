/*
 * wubu_priority_store.c -- the PRIORITY STORE (see the header).
 * The diagnose consults it before the next mutation.
 */
#include "wubu_priority_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wubu_prio_init(wubu_priority_store_t *ps, float protect_fisher,
                   float mutate_importance)
{
    if (!ps) return -1;
    memset(ps, 0, sizeof(*ps));
    ps->protect_fisher = protect_fisher > 0 ? protect_fisher : 0.5f;
    ps->mutate_importance = mutate_importance > 0 ? mutate_importance : 0.3f;
    return 0;
}

static wubu_prio_cell_t *find_cell(wubu_priority_store_t *ps, uint8_t idx)
{
    for (int i = 0; i < ps->n; i++)
        if (ps->cells[i].cell_idx == idx) return &ps->cells[i];
    return NULL;
}

static wubu_prio_cell_t *ensure_cell(wubu_priority_store_t *ps, uint8_t idx)
{
    wubu_prio_cell_t *c = find_cell(ps, idx);
    if (c) return c;
    if (ps->n >= 32) return NULL;
    c = &ps->cells[ps->n++];
    memset(c, 0, sizeof(*c));
    c->cell_idx = idx;
    return c;
}

int wubu_prio_register(wubu_priority_store_t *ps, uint8_t cell_idx,
                       uint8_t family, float importance, uint64_t batch)
{
    if (!ps) return -1;
    wubu_prio_cell_t *c = ensure_cell(ps, cell_idx);
    if (!c) return -1;
    c->family = family;
    c->importance = importance < 0 ? 0 : (importance > 1 ? 1 : importance);
    (void)batch;
    return 0;
}

void wubu_prio_update_fisher(wubu_priority_store_t *ps, uint8_t cell_idx,
                             float grad_norm, float beta)
{
    if (!ps) return;
    wubu_prio_cell_t *c = ensure_cell(ps, cell_idx);
    if (!c) return;
    float b = beta < 0 ? 0.0f : (beta > 1 ? 1 : beta);
    c->fisher = (1.0f - b) * c->fisher + b * grad_norm * grad_norm;
    /* re-evaluate the protection: high Fisher + a recent reject */
    c->protected = (c->fisher >= ps->protect_fisher && c->n_rejected > 0) ? 1 : 0;
}

void wubu_prio_record_mutation(wubu_priority_store_t *ps, uint8_t cell_idx,
                               int accepted)
{
    if (!ps) return;
    wubu_prio_cell_t *c = ensure_cell(ps, cell_idx);
    if (!c) return;
    c->n_mutations++;
    if (!accepted) {
        c->n_rejected++;
        c->protected = (c->fisher >= ps->protect_fisher) ? 1 : 0;
    }
}

int wubu_prio_gate(wubu_priority_store_t *ps, uint8_t cell_idx)
{
    if (!ps) return 0;
    ps->n_consulted++;
    /* find the cell; unregistered cells are unconstrained (a fresh
     * cell has no Fisher evidence yet — allow) */
    wubu_prio_cell_t *c = NULL;
    for (int i = 0; i < ps->n; i++)
        if (ps->cells[i].cell_idx == cell_idx) { c = &ps->cells[i]; break; }
    if (!c) { ps->n_allowed++; return 1; }
    /* the protection: a cell the loss cares about (high Fisher) with a
     * recent rejection is REFUSED for further mutation */
    if (c->protected) { ps->n_refused++; return 0; }
    /* low importance = a mutation candidate (allowed) */
    ps->n_allowed++;
    return 1;
}

void wubu_prio_set_delta(wubu_priority_store_t *ps, uint8_t cell_idx,
                         float delta)
{
    if (!ps) return;
    wubu_prio_cell_t *c = ensure_cell(ps, cell_idx);
    if (!c) return;
    c->precision_delta = delta;
}

/* the sidecar header (the tiny binary format) */
typedef struct {
    uint32_t magic;   /* 0xPR10 0x4E 0x54 0x59 */
    uint32_t n;
    uint64_t consulted, refused, allowed;
} prio_sidecar_hdr_t;

long wubu_prio_save(const wubu_priority_store_t *ps, void *buf, long cap)
{
    if (!ps || !buf || cap <= 0) return 0;
    prio_sidecar_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x50524930u;   /* "PRI0" */
    hdr.n = (uint32_t)ps->n;
    hdr.consulted = ps->n_consulted;
    hdr.refused = ps->n_refused;
    hdr.allowed = ps->n_allowed;
    long need = (long)(sizeof(hdr) + (size_t)ps->n * sizeof(wubu_prio_cell_t));
    if (cap < need) return 0;
    long off = 0;
    memcpy((char *)buf + off, &hdr, sizeof(hdr)); off += (long)sizeof(hdr);
    for (int i = 0; i < ps->n; i++) {
        memcpy((char *)buf + off, &ps->cells[i], sizeof(wubu_prio_cell_t));
        off += (long)sizeof(wubu_prio_cell_t);
    }
    return off;
}

int wubu_prio_load(wubu_priority_store_t *ps, const void *buf, long n)
{
    if (!ps || !buf || n <= 0) return -1;
    prio_sidecar_hdr_t hdr;
    long off = 0;
    if (n < (long)sizeof(hdr)) return -1;
    memcpy(&hdr, (const char *)buf + off, sizeof(hdr)); off += (long)sizeof(hdr);
    if (hdr.magic != 0x50524930u) return -1;
    int cnt = (int)hdr.n;
    if (cnt > 32) cnt = 32;
    for (int i = 0; i < cnt; i++) {
        memcpy(&ps->cells[i], (const char *)buf + off, sizeof(wubu_prio_cell_t));
        off += (long)sizeof(wubu_prio_cell_t);
    }
    ps->n = cnt;
    ps->n_consulted = hdr.consulted;
    ps->n_refused = hdr.refused;
    ps->n_allowed = hdr.allowed;
    return cnt;
}

void wubu_prio_stats(const wubu_priority_store_t *ps, char *buf, size_t cap)
{
    if (!ps || !buf || cap == 0) return;
    int n_prot = 0;
    for (int i = 0; i < ps->n; i++)
        if (ps->cells[i].protected) n_prot++;
    snprintf(buf, cap,
             "cells=%d protected=%d consulted=%llu refused=%llu allowed=%llu",
             ps->n, n_prot,
             (unsigned long long)ps->n_consulted,
             (unsigned long long)ps->n_refused,
             (unsigned long long)ps->n_allowed);
}
