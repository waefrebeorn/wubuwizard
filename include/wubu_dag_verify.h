#ifndef WUBU_DAG_VERIFY_H
#define WUBU_DAG_VERIFY_H

/*
 * wubu_dag_verify.h -- EG-25c: DAG verification of speculative drafts.
 * Candidate paths that share a prefix are interned into one rooted DAG, so a
 * shared node is verified once instead of once per path. The plan is a
 * parent-before-child order. This gates the dedup/count contract, not a
 * model-quality or latency result.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_DAG_MAX_NODES 128
#define WUBU_DAG_MAX_PATHS 64
#define WUBU_DAG_MAX_LEN   32

typedef struct wubu_dag_verify wubu_dag_verify_t;

wubu_dag_verify_t *wubu_dag_verify_create(int node_cap, int path_cap);
void wubu_dag_verify_free(wubu_dag_verify_t *p);

int wubu_dag_verify_add_path(wubu_dag_verify_t *p,
                             const int *path, int len);
int wubu_dag_verify_node_count(const wubu_dag_verify_t *p);
int wubu_dag_verify_path_count(const wubu_dag_verify_t *p);

/* Parent-before-child node tokens; root is -1. Returns count or -1. */
int wubu_dag_verify_plan(const wubu_dag_verify_t *p, int *out, int cap);

/* Tree checks minus unique non-root nodes (0 when there is no saving). */
int wubu_dag_verify_checks_saved(const wubu_dag_verify_t *p, int tree_checks);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_DAG_VERIFY_H */
