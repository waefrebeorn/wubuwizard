/*
 * wubu_orbits.c -- NESTED-SPHERE MEMORY (research/060, AN12 wave 3).
 *
 * The fractal stacking: an item's address is its nested polar path
 * (radius, angle) per level down the recursion — spheres in orbits
 * (the angle) and spheres inside spheres (the radius recursion). The
 * hive is the physical backing; capacity is ring-bounded per level,
 * depth is unbounded (nest() grows the recursion — infinite memory).
 *
 * C11, self-contained.
 */
#include "wubu_orbits.h"
#include "wubu_std.h"        /* M_PI without WUBU_HOSTED (no GNU dep) */
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_orbits {
    wubu_hive_t *hive;
    wubu_orbits_cfg_t cfg;
    int depth;                 /* current nesting depth */
    size_t *level_live;        /* [depth] live count per level (ring) */
};

/* the hive slot payload: the address + the item */
typedef struct {
    wubu_orbit_addr_t addr;
    void *item;
} orbit_slot_t;

/* the oldest-cell finder (ring discipline: recycle the oldest at a
 * level when the level is full) */
struct oldest_ctx { wubu_orbits_t *o; orbit_slot_t *oldest; double min_key; };
static int find_oldest(void *ptr, void *user)
{
    struct oldest_ctx *c = (struct oldest_ctx *)user;
    orbit_slot_t *s = (orbit_slot_t *)ptr;
    double key = s->addr.r[0] * 1e6 + s->addr.theta[0];
    if (!c->oldest || key < c->min_key) {
        c->oldest = s;
        c->min_key = key;
    }
    return 0;
}

/* the slot's addr is a STRUCT FIELD, not a heap wrapper — free only
 * the arrays it owns (never free(&slot->addr)) */
static void slot_addr_free(wubu_orbit_addr_t *a);

wubu_orbits_t *wubu_orbits_init(wubu_hive_t *hive,
                                const wubu_orbits_cfg_t *cfg)
{
    if (!hive || !cfg || cfg->max_depth < 1 || cfg->R_max <= 0)
        return NULL;
    wubu_orbits_t *o = (wubu_orbits_t *)calloc(1, sizeof(*o));
    if (!o) return NULL;
    o->hive = hive;
    o->cfg = *cfg;
    if (o->cfg.cap_per_level == 0) o->cfg.cap_per_level = 64;
    o->depth = 1;
    o->level_live = (size_t *)calloc((size_t)cfg->max_depth,
                                     sizeof(size_t));
    if (!o->level_live) { free(o); return NULL; }
    return o;
}

void wubu_orbits_free(wubu_orbits_t *o)
{
    if (!o) return;
    /* free every slot's address, then the hive (the hive is caller's
     * to clear — we only free our payloads) */
    for (wubu_hive_block_t *blk = o->hive->head; blk; blk = blk->next)
        for (size_t i = 0; i < blk->cap; i++) {
            if (blk->skip[i]) continue;
            orbit_slot_t *s = (orbit_slot_t *)blk->slots[i];
            slot_addr_free(&s->addr);
            free(s);
        }
    wubu_hive_clear(o->hive);
    free(o->level_live);
    free(o);
}

void wubu_orbits_addr_free(wubu_orbit_addr_t *a)
{
    if (!a) return;
    free(a->r);
    free(a->theta);
    free(a);
}

/* the slot's addr is a STRUCT FIELD, not a heap wrapper — free only
 * the arrays it owns (never free(&slot->addr)) */
static void slot_addr_free(wubu_orbit_addr_t *a)
{
    free(a->r);
    free(a->theta);
    a->r = NULL;
    a->theta = NULL;
}

static int addr_alloc(wubu_orbit_addr_t *a, int levels)
{
    a->r = (double *)calloc((size_t)levels, sizeof(double));
    a->theta = (double *)calloc((size_t)levels, sizeof(double));
    if (!a->r || !a->theta) return -1;
    a->n_levels = levels;
    return 0;
}

/* the polar recursion: level l takes the leading 2-D pair of the
 * current sub-vector -> (radius, angle), then recurses on the rest —
 * a sphere nested inside the previous one. Radius = distance from the
 * central mass (depth); angle = position on the sphere
 * (specialization). Deterministic and self-similar (the fractal). */
static void polar_walk(const wubu_orbits_t *o, const float *x, int d,
                       wubu_orbit_addr_t *a)
{
    const double R = o->cfg.R_max;
    const float *p = x;
    int rem = d;
    for (int l = 0; l < a->n_levels && l < o->depth && rem >= 2; l++) {
        double r = sqrt((double)p[0] * p[0] + (double)p[1] * p[1]);
        double rr = r / R;                   /* normalized radius */
        if (rr > 0.999) rr = 0.999;          /* stay in the ball */
        a->r[l] = rr;
        double theta = atan2((double)p[1], (double)p[0]);
        if (theta < 0) theta += 2.0 * M_PI;
        a->theta[l] = theta;
        p += 2;                              /* the inner sphere */
        rem -= 2;
    }
}

wubu_orbit_addr_t *wubu_orbits_addr(const wubu_orbits_t *o,
                                    const float *x, int d)
{
    if (!o || !x || d < 2) return NULL;
    wubu_orbit_addr_t *a = (wubu_orbit_addr_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    if (addr_alloc(a, o->depth) != 0) { free(a); return NULL; }
    polar_walk(o, x, d, a);
    return a;
}

int wubu_orbits_write(wubu_orbits_t *o, const float *x, int d, void *item)
{
    if (!o || !x || d < 2) return -1;
    wubu_orbit_addr_t *a = wubu_orbits_addr(o, x, d);
    if (!a) return -1;
    /* ring discipline: the innermost level's live count is capped */
    int last = o->depth - 1;
    if (o->level_live[last] >= o->cfg.cap_per_level) {
        struct oldest_ctx c = { o, NULL, 0 };
        wubu_hive_foreach(o->hive, find_oldest, &c);
        if (c.oldest) {
            wubu_hive_erase(o->hive, c.oldest);
            slot_addr_free(&c.oldest->addr);
            free(c.oldest);
            if (o->level_live[last] > 0) o->level_live[last]--;
        }
    }
    orbit_slot_t *s = (orbit_slot_t *)calloc(1, sizeof(*s));
    if (!s) { wubu_orbits_addr_free(a); return -1; }
    /* transfer the arrays (the slot owns them now); the wrapper `a`
     * is freed without touching the arrays */
    s->addr.r = a->r;
    s->addr.theta = a->theta;
    s->addr.n_levels = a->n_levels;
    free(a);
    s->item = item;
    if (wubu_hive_insert(o->hive, s) != 0) {
        slot_addr_free(&s->addr);
        free(s);
        return -1;
    }
    o->level_live[last]++;
    return 0;
}

void *wubu_orbits_read(const wubu_orbits_t *o, const wubu_orbit_addr_t *addr)
{
    if (!o || !addr) return NULL;
    void *found = NULL;
    /* walk the hive for a slot whose address matches at every level */
    for (wubu_hive_block_t *blk = o->hive->head; blk; blk = blk->next) {
        for (size_t i = 0; i < blk->cap; i++) {
            if (blk->skip[i]) continue;
            orbit_slot_t *s = (orbit_slot_t *)blk->slots[i];
            int match = s->addr.n_levels == addr->n_levels;
            for (int l = 0; l < addr->n_levels && match; l++) {
                if (fabs(s->addr.r[l] - addr->r[l]) > 1e-6) match = 0;
                if (fabs(s->addr.theta[l] - addr->theta[l]) > 1e-3) match = 0;
            }
            if (match) { found = s->item; break; }
        }
        if (found) break;
    }
    return found;
}

int wubu_orbits_nest(wubu_orbits_t *o)
{
    if (!o) return -1;
    if (o->depth >= (int)o->cfg.max_depth) return -1;  /* at the cap */
    o->depth++;
    return 0;
}

int wubu_orbits_depth(const wubu_orbits_t *o)
{
    return o ? o->depth : 0;
}
