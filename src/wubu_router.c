/* wubu_router.c -- the router registry (THE ROUTER SLOT).
 *
 * One slot, many physics. Routers register under a name; consumers
 * probe. The registry is a small growable table -- no static size, no
 * fixed order (Revolver: no static policy tables).
 *
 * C11, self-contained (stdlib only).
 */
#include "wubu_router.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    wubu_router_t *slots;
    size_t count;
    size_t cap;
} registry_t;

static registry_t g_registry = { NULL, 0, 0 };

int wubu_router_register(wubu_router_t *router) {
    if (!router || !router->name || !router->route) return -1;

    /* Replace an existing name: free the old ctx, keep the slot. */
    for (size_t i = 0; i < g_registry.count; i++) {
        if (strcmp(g_registry.slots[i].name, router->name) == 0) {
            if (g_registry.slots[i].free && g_registry.slots[i].ctx)
                g_registry.slots[i].free(g_registry.slots[i].ctx);
            g_registry.slots[i] = *router;
            return 0;
        }
    }

    /* Append. Grow by doubling (no fixed capacity). */
    if (g_registry.count == g_registry.cap) {
        size_t nc = g_registry.cap ? g_registry.cap * 2 : 4;
        wubu_router_t *ns = (wubu_router_t *)realloc(g_registry.slots,
                                                     nc * sizeof(*ns));
        if (!ns) return -1;
        g_registry.slots = ns;
        g_registry.cap = nc;
    }
    g_registry.slots[g_registry.count++] = *router;
    return 0;
}

wubu_router_t *wubu_router_get(const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < g_registry.count; i++)
        if (strcmp(g_registry.slots[i].name, name) == 0)
            return &g_registry.slots[i];
    return NULL;
}

size_t wubu_router_count(void) {
    return g_registry.count;
}

const char *wubu_router_name_at(size_t i) {
    if (i >= g_registry.count) return NULL;
    return g_registry.slots[i].name;
}

int wubu_router_route_named(const char *name, const float *x, int dim, int k,
                            int *out_idx, float *out_w) {
    wubu_router_t *r = wubu_router_get(name);
    if (!r || !r->route) return -1;
    return r->route(r->ctx, x, dim, k, out_idx, out_w);
}
