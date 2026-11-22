/*
 * wubu_radix_cache.c -- EG-08 radix prefix tree. Pure C11, opaque seams.
 */
#include "wubu_radix_cache.h"

#include <stdlib.h>
#include <string.h>

struct wubu_radix_node {
    int token;                          /* -1 for root */
    struct wubu_radix_node *parent;
    struct wubu_radix_node **children;
    int nchild;
    int block_id;                       /* -1 when not a terminal */
    uint64_t last;
};

struct wubu_radix_cache {
    wubu_radix_node_t *root;
    int max_nodes;
    int nodes;
    uint64_t clock;
    uint64_t hits, misses, evictions;
};

static wubu_radix_node_t *new_node(int token, wubu_radix_node_t *parent) {
    wubu_radix_node_t *n = (wubu_radix_node_t *)calloc(1, sizeof(*n));
    if (!n) return NULL;
    n->token = token;
    n->parent = parent;
    n->block_id = -1;
    return n;
}

wubu_radix_cache_t *wubu_radix_cache_create(int max_nodes) {
    wubu_radix_cache_t *c;
    if (max_nodes < 2) return NULL;
    c = (wubu_radix_cache_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->root = new_node(-1, NULL);
    if (!c->root) { free(c); return NULL; }
    c->max_nodes = max_nodes;
    c->nodes = 1;
    return c;
}

static void free_subtree(wubu_radix_node_t *n) {
    int i;
    if (!n) return;
    for (i = 0; i < n->nchild; i++) free_subtree(n->children[i]);
    free(n->children);
    free(n);
}

void wubu_radix_cache_free(wubu_radix_cache_t *c) {
    if (!c) return;
    free_subtree(c->root);
    free(c);
}

static wubu_radix_node_t *find_child(const wubu_radix_node_t *n, int token) {
    int i;
    for (i = 0; i < n->nchild; i++)
        if (n->children[i]->token == token) return n->children[i];
    return NULL;
}

static int add_child(wubu_radix_node_t *parent, wubu_radix_node_t *child) {
    wubu_radix_node_t **grown =
        (wubu_radix_node_t **)realloc(parent->children,
            (size_t)(parent->nchild + 1) * sizeof(*grown));
    if (!grown) return -1;
    parent->children = grown;
    parent->children[parent->nchild++] = child;
    return 0;
}

int wubu_radix_cache_match(wubu_radix_cache_t *c, const int *tokens, int n,
                           const wubu_radix_node_t **node_out) {
    wubu_radix_node_t *cur;
    int i = 0;
    if (!c || !tokens || n < 0 || (n > 0 && !tokens)) return -1;
    cur = c->root;
    for (i = 0; i < n; i++) {
        wubu_radix_node_t *next = find_child(cur, tokens[i]);
        if (!next) break;
        cur = next;
        cur->last = ++c->clock;
    }
    if (i == n) c->hits++; else c->misses++;
    if (node_out) *node_out = cur;
    return i;
}

int wubu_radix_cache_insert(wubu_radix_cache_t *c, const int *tokens, int n,
                            int block_id) {
    wubu_radix_node_t *cur;
    int i;
    if (!c || !tokens || n <= 0) return -1;

    /* Evict before walking so an eviction cannot invalidate the path. */
    if (c->nodes + n > c->max_nodes) {
        int drop = c->nodes + n - c->max_nodes;
        if (wubu_radix_cache_evict_lru(c, drop) < drop) return -1;
    }

    cur = c->root;
    for (i = 0; i < n; i++) {
        wubu_radix_node_t *next = find_child(cur, tokens[i]);
        if (!next) {
            next = new_node(tokens[i], cur);
            if (!next) return -1;
            if (add_child(cur, next) != 0) { free(next); return -1; }
            c->nodes++;
        }
        next->last = ++c->clock;
        cur = next;
    }
    cur->block_id = block_id;
    return 0;
}

static int evict_one_leaf(wubu_radix_cache_t *c) {
    wubu_radix_node_t *victim = NULL, *stack[4096];
    int sp = 0, i;
    stack[sp++] = c->root;
    while (sp > 0) {
        wubu_radix_node_t *n = stack[--sp];
        if (n != c->root && n->nchild == 0 &&
            (!victim || n->last < victim->last)) victim = n;
        for (i = 0; i < n->nchild; i++) {
            if (sp < 4096) stack[sp++] = n->children[i];
        }
    }
    if (!victim) return 0;
    wubu_radix_node_t *p = victim->parent;
    for (i = 0; i < p->nchild; i++) {
        if (p->children[i] == victim) {
            memmove(&p->children[i], &p->children[i + 1],
                    (size_t)(p->nchild - i - 1) * sizeof(*p->children));
            p->nchild--;
            break;
        }
    }
    free(victim->children);
    free(victim);
    c->nodes--;
    c->evictions++;
    return 1;
}

int wubu_radix_cache_evict_lru(wubu_radix_cache_t *c, int count) {
    int done = 0;
    if (!c || count <= 0) return 0;
    while (done < count && c->nodes > 1) {
        if (!evict_one_leaf(c)) break;
        done++;
    }
    return done;
}

int wubu_radix_cache_node_count(const wubu_radix_cache_t *c) {
    return c ? c->nodes : -1;
}

int wubu_radix_cache_child_count(const wubu_radix_node_t *node) {
    return node ? node->nchild : -1;
}

int wubu_radix_cache_block(const wubu_radix_node_t *node) {
    return node ? node->block_id : -1;
}

uint64_t wubu_radix_cache_hits(const wubu_radix_cache_t *c) {
    return c ? c->hits : 0;
}

uint64_t wubu_radix_cache_misses(const wubu_radix_cache_t *c) {
    return c ? c->misses : 0;
}

uint64_t wubu_radix_cache_evictions(const wubu_radix_cache_t *c) {
    return c ? c->evictions : 0;
}
