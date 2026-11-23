/* wubu_dag_verify.c -- EG-25c DAG verification gate. Strict C11. */
#include "wubu_dag_verify.h"

#include <stdlib.h>

typedef struct { int token, parent; } dag_node_t;

struct wubu_dag_verify {
    int cap, path_cap, n, paths;
    dag_node_t node[WUBU_DAG_MAX_NODES];
};

wubu_dag_verify_t *wubu_dag_verify_create(int node_cap, int path_cap)
{
    wubu_dag_verify_t *p;
    if (node_cap < 2 || node_cap > WUBU_DAG_MAX_NODES ||
        path_cap < 1 || path_cap > WUBU_DAG_MAX_PATHS) return NULL;
    p = (wubu_dag_verify_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->cap = node_cap;
    p->path_cap = path_cap;
    p->node[0].token = -1;
    p->node[0].parent = -1;
    p->n = 1;
    return p;
}

void wubu_dag_verify_free(wubu_dag_verify_t *p) { free(p); }

int wubu_dag_verify_add_path(wubu_dag_verify_t *p,
                             const int *path, int len)
{
    int cur = 0;
    if (!p || !path || len < 1 || len > WUBU_DAG_MAX_LEN ||
        p->paths >= p->path_cap) return -1;
    for (int i = 0; i < len; ++i) {
        int found = -1;
        for (int j = 1; j < p->n; ++j)
            if (p->node[j].parent == cur && p->node[j].token == path[i]) {
                found = j;
                break;
            }
        if (found >= 0) {
            cur = found;
            continue;
        }
        if (p->n >= p->cap) return -1;
        p->node[p->n].token = path[i];
        p->node[p->n].parent = cur;
        cur = p->n++;
    }
    ++p->paths;
    return 0;
}

int wubu_dag_verify_node_count(const wubu_dag_verify_t *p)
{
    return p ? p->n : -1;
}

int wubu_dag_verify_path_count(const wubu_dag_verify_t *p)
{
    return p ? p->paths : -1;
}

int wubu_dag_verify_plan(const wubu_dag_verify_t *p, int *out, int cap)
{
    unsigned char done[WUBU_DAG_MAX_NODES] = {0};
    int k = 0;
    if (!p || !out || cap < p->n) return -1;
    for (int round = 0; round < p->n; ++round) {
        int progress = 0;
        for (int i = 0; i < p->n; ++i) {
            if (done[i]) continue;
            if (i != 0 && !done[p->node[i].parent]) continue;
            out[k++] = p->node[i].token;
            done[i] = 1;
            progress = 1;
        }
        if (!progress) return -1;
        if (k == p->n) break;
    }
    return k;
}

int wubu_dag_verify_checks_saved(const wubu_dag_verify_t *p, int tree_checks)
{
    if (!p) return -1;
    if (tree_checks < p->n - 1) return 0;
    return tree_checks - (p->n - 1);
}
