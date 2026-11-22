/*
 * wubu_grammar_fsm.c -- EG-11 grammar automaton. Pure C11, no dependencies.
 */
#include "wubu_grammar_fsm.h"

#include <stdlib.h>
#include <string.h>

typedef struct { int token; int child; } grammar_edge_t;
typedef struct {
    grammar_edge_t *edges;
    int nedge;
    int terminal;
} grammar_node_t;

struct wubu_grammar_fsm {
    grammar_node_t *nodes;
    int nnodes, node_cap;
    int **alts;
    int *alt_len;
    int nalt, alt_cap;
    int eos;
    int compiled;
};

wubu_grammar_fsm_t *wubu_grammar_fsm_create(int eos_token) {
    wubu_grammar_fsm_t *g = (wubu_grammar_fsm_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->eos = eos_token;
    return g;
}

static void free_alts(wubu_grammar_fsm_t *g) {
    for (int i = 0; i < g->nalt; i++) free(g->alts[i]);
    free(g->alts);
    free(g->alt_len);
    g->alts = NULL;
    g->alt_len = NULL;
    g->nalt = g->alt_cap = 0;
}

void wubu_grammar_fsm_free(wubu_grammar_fsm_t *g) {
    if (!g) return;
    free_alts(g);
    for (int i = 0; i < g->nnodes; i++) free(g->nodes[i].edges);
    free(g->nodes);
    free(g);
}

int wubu_grammar_fsm_add(wubu_grammar_fsm_t *g, const int *tokens, int n) {
    int *copy;
    if (!g || g->compiled || !tokens || n <= 0) return -1;
    copy = (int *)malloc((size_t)n * sizeof(int));
    if (!copy) return -1;
    memcpy(copy, tokens, (size_t)n * sizeof(int));
    if (g->nalt == g->alt_cap) {
        int cap = g->alt_cap ? g->alt_cap * 2 : 8;
        int **na = (int **)realloc(g->alts, (size_t)cap * sizeof(*na));
        int *nl;
        if (!na) { free(copy); return -1; }
        g->alts = na;
        nl = (int *)realloc(g->alt_len, (size_t)cap * sizeof(int));
        if (!nl) { free(copy); return -1; }
        g->alt_len = nl;
        g->alt_cap = cap;
    }
    g->alts[g->nalt] = copy;
    g->alt_len[g->nalt] = n;
    g->nalt++;
    return 0;
}

static int new_node(wubu_grammar_fsm_t *g) {
    if (g->nnodes == g->node_cap) {
        int cap = g->node_cap ? g->node_cap * 2 : 16;
        grammar_node_t *nn = (grammar_node_t *)realloc(g->nodes,
                                    (size_t)cap * sizeof(*nn));
        if (!nn) return -1;
        g->nodes = nn;
        g->node_cap = cap;
    }
    memset(&g->nodes[g->nnodes], 0, sizeof(g->nodes[g->nnodes]));
    return g->nnodes++;
}

static int find_edge(grammar_node_t *n, int token) {
    for (int i = 0; i < n->nedge; i++)
        if (n->edges[i].token == token) return n->edges[i].child;
    return -1;
}

static int add_edge(grammar_node_t *n, int token, int child) {
    grammar_edge_t *grown = (grammar_edge_t *)realloc(n->edges,
                        (size_t)(n->nedge + 1) * sizeof(*grown));
    if (!grown) return -1;
    n->edges = grown;
    n->edges[n->nedge].token = token;
    n->edges[n->nedge].child = child;
    n->nedge++;
    return 0;
}

int wubu_grammar_fsm_compile(wubu_grammar_fsm_t *g) {
    if (!g || g->compiled || g->nalt == 0) return -1;
    if (new_node(g) != 0) return -1; /* node 0 = root */
    for (int i = 0; i < g->nalt; i++) {
        int cur = 0;
        for (int j = 0; j < g->alt_len[i]; j++) {
            int token = g->alts[i][j];
            int child = find_edge(&g->nodes[cur], token);
            if (child < 0) {
                child = new_node(g);
                if (child < 0) return -1;
                if (add_edge(&g->nodes[cur], token, child) != 0) return -1;
            }
            cur = child;
        }
        g->nodes[cur].terminal = 1;
    }
    g->compiled = 1;
    free_alts(g);
    return 0;
}

void wubu_grammar_fsm_state_init(wubu_grammar_state_t *s) {
    if (s) s->node = 0;
}

int wubu_grammar_fsm_complete(const wubu_grammar_state_t *s) {
    return s && s->node < 0;
}

int wubu_grammar_fsm_node_count(const wubu_grammar_fsm_t *g) {
    return g ? g->nnodes : -1;
}

static void all_bits(uint64_t *mask, int words) {
    if (!mask || words <= 0) return;
    for (int i = 0; i < words; i++) mask[i] = ~(uint64_t)0;
}

int wubu_grammar_fsm_mask(const wubu_grammar_fsm_t *g,
                          const wubu_grammar_state_t *s,
                          uint64_t *mask, int mask_words) {
    if (!g || !g->compiled || !s || !mask || mask_words <= 0) return -1;
    if (s->node < 0) { all_bits(mask, mask_words); return 0; }
    if (s->node >= g->nnodes) return -1;
    for (int w = 0; w < mask_words; w++) mask[w] = 0;
    grammar_node_t *n = &g->nodes[s->node];
    for (int i = 0; i < n->nedge; i++) {
        int t = n->edges[i].token;
        if (t >= 0 && t < mask_words * 64) mask[t / 64] |= UINT64_C(1) << (t % 64);
    }
    if (n->terminal && g->eos >= 0 && g->eos < mask_words * 64)
        mask[g->eos / 64] |= UINT64_C(1) << (g->eos % 64);
    return 0;
}

int wubu_grammar_fsm_mask_batch(const wubu_grammar_fsm_t *g,
                                const wubu_grammar_state_t *states, int batch,
                                uint64_t *masks, int mask_words) {
    if (!g || !states || !masks || batch <= 0 || mask_words <= 0) return -1;
    for (int b = 0; b < batch; b++) {
        if (wubu_grammar_fsm_mask(g, &states[b],
                                  masks + (size_t)b * (size_t)mask_words,
                                  mask_words) != 0) return -1;
    }
    return 0;
}

int wubu_grammar_fsm_advance(const wubu_grammar_fsm_t *g,
                             wubu_grammar_state_t *s, int token) {
    if (!g || !g->compiled || !s) return -1;
    if (s->node < 0) return 1; /* jump-forward: grammar already satisfied */
    if (s->node >= g->nnodes) return -1;
    grammar_node_t *n = &g->nodes[s->node];
    int child = find_edge(n, token);
    if (child < 0) {
        if (n->terminal && token == g->eos) { s->node = -1; return 1; }
        return -1;
    }
    s->node = child;
    if (g->nodes[child].terminal) { s->node = -1; return 1; }
    return 0;
}

int wubu_grammar_fsm_jump_forward(const wubu_grammar_fsm_t *g,
                                   const wubu_grammar_state_t *s,
                                   uint64_t *mask, int mask_words) {
    if (!s || !mask || mask_words <= 0) return -1;
    if (wubu_grammar_fsm_complete(s)) { all_bits(mask, mask_words); return 1; }
    return wubu_grammar_fsm_mask(g, s, mask, mask_words);
}
