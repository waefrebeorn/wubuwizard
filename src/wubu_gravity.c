/*
 * wubu_gravity.c -- THE GRAVITY FIELD (research/060, INDEX AN12).
 *
 * The organizing force of the amoeba weights: a central mass (the
 * Colonel boot core) surrounded by cells in orbit on the Poincaré
 * ball. Gravity F = G·M·m/r² organizes the system — radius = depth,
 * angle = specialization. The core (r < boot_r) is protected: it
 * never splits, never shrinks below the boot minimum; the body grows
 * outward around it.
 *
 * C11, self-contained.
 */
#include "wubu_gravity.h"
#include "wubu_std.h"        /* M_PI without WUBU_HOSTED (no GNU dep) */
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_gravity {
    wubu_gravity_cfg_t cfg;
    wubu_gravity_cell_t *cells;   /* [cfg.n_cells] */
    size_t live;                  /* cells in orbit */
    double r_max;                 /* 1/sqrt(c) — the ball's boundary */
};

static double clamp_r(const wubu_gravity_t *g, double r)
{
    if (r < 0) r = 0;
    double m = g->r_max * 0.999;
    if (r > m) r = m;
    return r;
}

wubu_gravity_t *wubu_gravity_init(const wubu_gravity_cfg_t *cfg)
{
    if (!cfg || cfg->n_cells == 0 || cfg->c <= 0 || cfg->M <= 0)
        return NULL;
    wubu_gravity_t *g = (wubu_gravity_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->cfg = *cfg;
    if (g->cfg.G <= 0) g->cfg.G = 1.0;
    if (g->cfg.boot_r <= 0) g->cfg.boot_r = 0.3;
    g->r_max = 1.0 / sqrt(cfg->c);
    g->cells = (wubu_gravity_cell_t *)calloc(cfg->n_cells,
                                             sizeof(wubu_gravity_cell_t));
    if (!g->cells) { free(g); return NULL; }
    /* seed: cells spread across the ball — the first few inside the
     * boot radius (the core), the rest outward (the body grows around
     * the core). */
    double spread = g->r_max * 0.9;   /* seed up to 90% of the ball */
    for (size_t i = 0; i < cfg->n_cells; i++) {
        wubu_gravity_cell_t *c = &g->cells[i];
        c->id = (int)i;
        c->mass = 1.0;
        c->r = g->cfg.boot_r * 0.5
             + (spread - g->cfg.boot_r * 0.5) * (double)i / (double)cfg->n_cells;
        c->theta = (double)i * 2.0 * M_PI / (double)cfg->n_cells;
        c->vr = 0;
        /* the stable circular orbit velocity at this radius */
        c->vtheta = sqrt(g->cfg.G * g->cfg.M / (c->r + 1e-9));
        c->core = (c->r < g->cfg.boot_r) ? 1 : 0;
        c->flags = 0;
    }
    g->live = cfg->n_cells;
    return g;
}

void wubu_gravity_free(wubu_gravity_t *g)
{
    if (!g) return;
    free(g->cells);
    free(g);
}

int wubu_gravity_step(wubu_gravity_t *g)
{
    if (!g) return -1;
    for (size_t i = 0; i < g->cfg.n_cells; i++) {
        wubu_gravity_cell_t *c = &g->cells[i];
        if (c->mass <= 0) continue;           /* removed (shrink) */
        double r = c->r;
        if (r <= 1e-12) r = 1e-12;
        /* the central-mass force: F = G·M·m/r² (inward acceleration) */
        double a_radial = -g->cfg.G * g->cfg.M / (r * r);
        /* the orbit velocity at this radius: vθ² = G·M/r */
        double v_orbit = sqrt(g->cfg.G * g->cfg.M / r);
        /* radial acceleration from the tangential-velocity mismatch:
         * too slow -> falls in, too fast -> drifts out */
        double a_centrifugal = (c->vtheta * c->vtheta) / r;
        double a_net = a_centrifugal + a_radial;
        c->vr += a_net * 0.01;                /* dt = 0.01 */
        c->r = clamp_r(g, r + c->vr * 0.01);
        c->theta += c->vtheta * 0.01 / (r + 1e-9);
        /* relaxation toward the local orbit velocity */
        c->vtheta += (v_orbit - c->vtheta) * 0.05;
        /* the core stays protected: never drift a core cell out */
        if (c->core && c->r >= g->cfg.boot_r)
            c->r = g->cfg.boot_r * 0.98;
        if (c->core)
            c->vr = 0;                        /* the core is pinned */
    }
    return 0;
}

int wubu_gravity_route(const wubu_gravity_t *g, double r_in, double theta_in)
{
    if (!g) return -1;
    /* wrap theta_in into [0, 2π) */
    while (theta_in < 0) theta_in += 2.0 * M_PI;
    while (theta_in >= 2.0 * M_PI) theta_in -= 2.0 * M_PI;
    int best = -1;
    double best_d = 1e30;
    for (size_t i = 0; i < g->cfg.n_cells; i++) {
        const wubu_gravity_cell_t *c = &g->cells[i];
        if (c->mass <= 0) continue;
        double dtheta = fabs(theta_in - c->theta);
        if (dtheta > M_PI) dtheta = 2.0 * M_PI - dtheta;
        /* polar distance: radial + angular (angular scaled by radius) */
        double d = (r_in - c->r) * (r_in - c->r) + (dtheta * c->r) * (dtheta * c->r);
        if (d < best_d) { best_d = d; best = c->id; }
    }
    return best;
}

int wubu_gravity_grow(wubu_gravity_t *g, int cell_id, double dr, double dtheta)
{
    if (!g || cell_id < 0 || (size_t)cell_id >= g->cfg.n_cells) return -1;
    wubu_gravity_cell_t *parent = &g->cells[cell_id];
    if (parent->mass <= 0) return -1;
    if (parent->core) return -1;              /* the core never splits */
    /* snapshot the parent BEFORE any realloc (the realloc below may
     * free the old array — parent would dangle; the DA/ASan catch) */
    const double pr = parent->r;
    const double ptheta = parent->theta;
    /* find a free slot (a shrunk cell) — else grow a NEW block (the
     * hive doctrine: freelist pop OR a new block; the amoeba grows) */
    size_t slot = g->cfg.n_cells;
    for (size_t i = 0; i < g->cfg.n_cells; i++) {
        if (g->cells[i].mass <= 0) { slot = i; break; }
    }
    if (slot == g->cfg.n_cells) {
        /* new block: grow the cell array (like hive_insert allocating
         * a new block) */
        size_t new_cap = g->cfg.n_cells + 4;
        wubu_gravity_cell_t *nc = (wubu_gravity_cell_t *)calloc(
            new_cap, sizeof(wubu_gravity_cell_t));
        if (!nc) return -1;
        memcpy(nc, g->cells, g->cfg.n_cells * sizeof(wubu_gravity_cell_t));
        free(g->cells);
        g->cells = nc;
        g->cfg.n_cells = new_cap;
        slot = new_cap - 1;   /* the new block's first free slot */
    }
    /* RE-DERIVE the parent after the realloc (the old pointer dangles) */
    wubu_gravity_cell_t *d1 = &g->cells[cell_id];
    wubu_gravity_cell_t *d2 = &g->cells[slot];
    /* the pseudopod: two daughters, pushed outward + split in angle */
    d1->r = clamp_r(g, d1->r + dr * 0.5);
    d1->theta += dtheta * 0.5;
    d1->mass *= 0.5;
    d2->id = (int)slot;
    d2->mass = d1->mass;
    d2->r = clamp_r(g, pr + dr);
    d2->theta = ptheta - dtheta * 0.5;
    d2->vr = 0.1;                             /* the new body drifts out */
    d2->vtheta = sqrt(g->cfg.G * g->cfg.M / (d2->r + 1e-9));
    d2->core = 0;
    d2->acc = 0;
    g->live++;
    return d2->id;
}

int wubu_gravity_shrink(wubu_gravity_t *g, int cell_id)
{
    if (!g || cell_id < 0 || (size_t)cell_id >= g->cfg.n_cells) return -1;
    wubu_gravity_cell_t *c = &g->cells[cell_id];
    if (c->mass <= 0) return -1;
    if (c->core) return -1;                   /* the core is never removed */
    /* apoptosis: the mass falls inward and merges into the central mass */
    g->cfg.M += c->mass * 0.5;
    memset(c, 0, sizeof(*c));
    c->mass = 0;                              /* removed */
    g->live--;
    return 0;
}

const wubu_gravity_cell_t *wubu_gravity_cell(const wubu_gravity_t *g,
                                             int cell_id)
{
    if (!g || cell_id < 0 || (size_t)cell_id >= g->cfg.n_cells) return NULL;
    const wubu_gravity_cell_t *c = &g->cells[cell_id];
    return c->mass > 0 ? c : NULL;
}

size_t wubu_gravity_count(const wubu_gravity_t *g)
{
    return g ? g->live : 0;
}

size_t wubu_gravity_core_count(const wubu_gravity_t *g)
{
    if (!g) return 0;
    size_t n = 0;
    for (size_t i = 0; i < g->cfg.n_cells; i++) {
        const wubu_gravity_cell_t *c = &g->cells[i];
        if (c->mass > 0 && c->core) n++;
    }
    return n;
}
