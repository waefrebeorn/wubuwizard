/* wubu_swa_reclaim.c -- multi-owner SWA cell reclaim (LP-06). */
#include "wubu_swa_reclaim.h"

#include <stdlib.h>
#include <string.h>

#define WUBU_SWA_MAGIC UINT64_C(0x5357415245434c4d) /* "SWARECLM" */

typedef struct {
    int64_t *pos;
    int      count;
    int      cap;
} swa_log;

struct wubu_swa_reclaim {
    uint64_t  magic;
    int       n_cells;
    int       n_owners;
    int64_t  *cell_pos;    /* -1 = empty */
    int      *cell_owner;  /* -1 = none */
    int64_t  *bound;       /* oldest still-needed position per owner */
    int       n_sinks;     /* EG-04: pinned first-N positions */
    swa_log  *logs;        /* overwritten positions, one log per owner */
};

static int valid(const wubu_swa_reclaim_t *c)
{
    return c && c->magic == WUBU_SWA_MAGIC;
}

wubu_swa_reclaim_t *wubu_swa_reclaim_create(int n_cells, int n_owners)
{
    wubu_swa_reclaim_t *c;
    int i;

    if (n_cells <= 0 || n_owners <= 0) return NULL;
    c = (wubu_swa_reclaim_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cell_pos   = (int64_t *)malloc(sizeof(int64_t) * (size_t)n_cells);
    c->cell_owner = (int *)malloc(sizeof(int) * (size_t)n_cells);
    c->bound      = (int64_t *)calloc((size_t)n_owners, sizeof(int64_t));
    c->logs       = (swa_log *)calloc((size_t)n_owners, sizeof(swa_log));
    if (!c->cell_pos || !c->cell_owner || !c->bound || !c->logs) {
        free(c->cell_pos);
        free(c->cell_owner);
        free(c->bound);
        free(c->logs);
        free(c);
        return NULL;
    }
    for (i = 0; i < n_cells; ++i) {
        c->cell_pos[i]   = -1;
        c->cell_owner[i] = -1;
    }
    c->magic    = WUBU_SWA_MAGIC;
    c->n_cells  = n_cells;
    c->n_owners = n_owners;
    return c;
}

void wubu_swa_reclaim_free(wubu_swa_reclaim_t *c)
{
    int i;

    if (!valid(c)) return;
    for (i = 0; i < c->n_owners; ++i) free(c->logs[i].pos);
    c->magic = 0;
    free(c->cell_pos);
    free(c->cell_owner);
    free(c->bound);
    free(c->logs);
    free(c);
}

wubu_swa_status_t wubu_swa_reclaim_set_bound(wubu_swa_reclaim_t *c,
                                             int owner, int64_t bound)
{
    if (!valid(c)) return WUBU_SWA_ERR_ARG;
    if (owner < 0 || owner >= c->n_owners) return WUBU_SWA_ERR_OWNER;
    if (bound < 0) return WUBU_SWA_ERR_POS;
    c->bound[owner] = bound;
    return WUBU_SWA_OK;
}

int64_t wubu_swa_reclaim_bound(const wubu_swa_reclaim_t *c, int owner)
{
    if (!valid(c) || owner < 0 || owner >= c->n_owners) return -1;
    return c->bound[owner];
}

int wubu_swa_reclaim_can_reuse(const wubu_swa_reclaim_t *c, int cell,
                               int owner, int64_t pos)
{
    int o;
    int64_t cur;

    if (!valid(c)) return 0;
    if (cell < 0 || cell >= c->n_cells) return 0;
    if (owner < 0 || owner >= c->n_owners) return 0;
    if (pos < 0) return 0;
    /* The requester cannot write a position already outside its window. */
    if (pos < c->bound[owner]) return 0;

    cur = c->cell_pos[cell];
    if (cur < 0) return 1; /* empty cell holds nothing any owner needs */

    /* EG-04: attention sinks are pinned regardless of the sliding window. */
    if (cur < c->n_sinks) return 0;

    /* Masked for every owner: cur < bound[o] for all o. One live vetoes. */
    for (o = 0; o < c->n_owners; ++o) {
        if (cur >= c->bound[o]) return 0;
    }
    return 1;
}

static wubu_swa_status_t log_append(swa_log *log, int64_t pos)
{
    int64_t *grown;
    int cap;

    if (log->count < log->cap) {
        log->pos[log->count++] = pos;
        return WUBU_SWA_OK;
    }
    cap = log->cap ? log->cap * 2 : 8;
    if (cap <= log->cap) return WUBU_SWA_ERR_CAP;
    if ((size_t)cap > (size_t)-1 / sizeof(int64_t)) return WUBU_SWA_ERR_CAP;
    grown = (int64_t *)realloc(log->pos, sizeof(int64_t) * (size_t)cap);
    if (!grown) return WUBU_SWA_ERR_OOM;
    log->pos = grown;
    log->cap = cap;
    log->pos[log->count++] = pos;
    return WUBU_SWA_OK;
}

wubu_swa_status_t wubu_swa_reclaim_reuse(wubu_swa_reclaim_t *c, int cell,
                                         int owner, int64_t pos)
{
    wubu_swa_status_t st;
    int64_t displaced;
    int prev_owner;

    if (!valid(c)) return WUBU_SWA_ERR_ARG;
    if (cell < 0 || cell >= c->n_cells) return WUBU_SWA_ERR_CELL;
    if (owner < 0 || owner >= c->n_owners) return WUBU_SWA_ERR_OWNER;
    if (pos < 0) return WUBU_SWA_ERR_POS;
    if (!wubu_swa_reclaim_can_reuse(c, cell, owner, pos))
        return WUBU_SWA_ERR_UNSAFE;

    displaced  = c->cell_pos[cell];
    prev_owner = c->cell_owner[cell];
    if (displaced >= 0 && prev_owner >= 0 && prev_owner < c->n_owners) {
        st = log_append(&c->logs[prev_owner], displaced);
        if (st != WUBU_SWA_OK) return st; /* cell still untouched */
    }

    c->cell_pos[cell]   = pos;
    c->cell_owner[cell] = owner;
    return WUBU_SWA_OK;
}

int64_t wubu_swa_reclaim_cell_pos(const wubu_swa_reclaim_t *c, int cell)
{
    if (!valid(c) || cell < 0 || cell >= c->n_cells) return -1;
    return c->cell_pos[cell];
}

wubu_swa_status_t wubu_swa_set_sinks(wubu_swa_reclaim_t *c, int n_sinks)
{
    if (!valid(c)) return WUBU_SWA_ERR_ARG;
    if (n_sinks < 0 || n_sinks > c->n_cells) return WUBU_SWA_ERR_POS;
    c->n_sinks = n_sinks;
    return WUBU_SWA_OK;
}

int wubu_swa_sinks(const wubu_swa_reclaim_t *c)
{
    return valid(c) ? c->n_sinks : -1;
}

int     wubu_swa_reclaim_cell_owner(const wubu_swa_reclaim_t *c, int cell)
{
    if (!valid(c) || cell < 0 || cell >= c->n_cells) return -1;
    return c->cell_owner[cell];
}

int wubu_swa_reclaim_overwritten_count(const wubu_swa_reclaim_t *c, int owner)
{
    if (!valid(c) || owner < 0 || owner >= c->n_owners) return 0;
    return c->logs[owner].count;
}

int64_t wubu_swa_reclaim_overwritten_at(const wubu_swa_reclaim_t *c,
                                        int owner, int index)
{
    if (!valid(c) || owner < 0 || owner >= c->n_owners) return -1;
    if (index < 0 || index >= c->logs[owner].count) return -1;
    return c->logs[owner].pos[index];
}

size_t wubu_swa_reclaim_snapshot_size(const wubu_swa_reclaim_t *c)
{
    size_t sz;
    int o;

    if (!valid(c)) return 0;
    sz = 2u * sizeof(int)
       + (size_t)c->n_owners * (sizeof(int64_t) + sizeof(int))
       + (size_t)c->n_cells  * (sizeof(int64_t) + sizeof(int))
       + sizeof(int); /* n_sinks */
    for (o = 0; o < c->n_owners; ++o)
        sz += (size_t)c->logs[o].count * sizeof(int64_t);
    return sz;
}

void wubu_swa_reclaim_snapshot(const wubu_swa_reclaim_t *c, unsigned char *out)
{
    size_t off = 0;
    int i, o;

    if (!valid(c) || !out) return;
    memcpy(out + off, &c->n_cells,  sizeof(int)); off += sizeof(int);
    memcpy(out + off, &c->n_owners, sizeof(int)); off += sizeof(int);
    memcpy(out + off, &c->n_sinks,  sizeof(int)); off += sizeof(int);
    for (o = 0; o < c->n_owners; ++o) {
        memcpy(out + off, &c->bound[o], sizeof(int64_t)); off += sizeof(int64_t);
        memcpy(out + off, &c->logs[o].count, sizeof(int)); off += sizeof(int);
    }
    for (o = 0; o < c->n_owners; ++o) {
        if (c->logs[o].count > 0) {
            memcpy(out + off, c->logs[o].pos,
                   (size_t)c->logs[o].count * sizeof(int64_t));
            off += (size_t)c->logs[o].count * sizeof(int64_t);
        }
    }
    for (i = 0; i < c->n_cells; ++i) {
        memcpy(out + off, &c->cell_pos[i], sizeof(int64_t)); off += sizeof(int64_t);
        memcpy(out + off, &c->cell_owner[i], sizeof(int)); off += sizeof(int);
    }
}
