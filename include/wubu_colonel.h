/*
 * wubu_colonel.h -- THE LIVE COLONEL bridge (the user directive #5:
 * "agentic OS integration surface: the Brain requests Body actions
 * through the Live Colonel within the hive trust boundary").
 *
 * The Brain (the colony: diagnosis + specialist cells + the judge) is
 * the requestor; the Body (WuBuOS: the kernel, GUI, Styx/9P namespace)
 * is the executor. This module is the channel between them:
 *
 *   - REQUEST: the Brain enqueues a Body action (a capability path
 *     in the 9P namespace + a goal token + the cell that decided it)
 *   - TRUST BOUNDARY: every request carries the agent subtree + the
 *     resource bound; the 9P capability check + the resource check
 *     gate execution (AD01/AD04)
 *   - SCHEDULING: the executor pulls by priority with backoff (AD02);
 *     failed requests return to the Brain as preference pairs (P4)
 *   - DURABLE: the request queue survives restart via checkpoints
 *     (AD03) — the Body never loses a Brain request
 *
 * Pure C11, opaque, no third party. The organs stay self-contained;
 * this is the trust boundary between the two halves of the AGI.
 */
#ifndef WUBU_COLONEL_H
#define WUBU_COLONEL_H

#include <stdint.h>
#include <stddef.h>

/* the Body action kinds (what the Brain can request) */
typedef enum {
    WUBU_ACT_READ = 0,      /* read a namespace path (observe) */
    WUBU_ACT_WRITE = 1,     /* write a namespace path (persist) */
    WUBU_ACT_RUN = 2,       /* run a bounded task (compute) */
    WUBU_ACT_SPAWN = 3,     /* spawn a sub-agent (delegate) */
    WUBU_ACT_KILL = 4       /* stop a runaway task (safety) */
} wubu_act_kind_t;

/* one Brain->Body request */
typedef struct {
    uint64_t    req_id;       /* the durable request id */
    wubu_act_kind_t kind;     /* what to do */
    char        path[256];    /* the 9P namespace path (the capability) */
    char        agent_subtree[128];  /* the requester's trust boundary */
    uint16_t    goal_token;   /* the goal that drove the request */
    uint8_t     cell_idx;     /* the colony cell that decided it */
    int         priority;     /* 0 (low) .. 3 (critical) */
    int         attempt;      /* retry count (backoff) */
    /* the resource bound (AD04) */
    long        cpu_ms_max, ram_mb_max, io_kb_max;
    int         done;         /* 1 = executed */
    int         accepted;     /* the outcome (the oracle's pair) */
} wubu_request_t;

/* the colonel state (the channel) */
typedef struct {
    wubu_request_t *queue;    /* the request ring */
    int  q_n, q_cap;
    uint64_t next_req_id;
    /* the outcome ledger (feeds the oracle: done requests become
     * preference pairs) */
    uint64_t n_done, n_failed;
    long base_backoff_ms;     /* AD02 */
} wubu_colonel_t;

/* C1: init the channel. */
int wubu_colonel_init(wubu_colonel_t *col, int q_cap, long base_backoff_ms);

/* C2: the Brain requests a Body action. The 9P capability + the
 * resource bound are checked HERE (the trust boundary at enqueue).
 * Returns the request id (>0) or -1 (denied / full). */
int64_t wubu_colonel_request(wubu_colonel_t *col, wubu_act_kind_t kind,
                             const char *path, const char *agent_subtree,
                             uint16_t goal, uint8_t cell_idx, int priority,
                             long cpu_ms, long ram_mb, long io_kb);

/* C3: the executor pulls the next eligible request (highest priority,
 * respecting the backoff). Returns the index or -1 (nothing ready). */
int wubu_colonel_pull(wubu_colonel_t *col);

/* C4: the executor reports the outcome (0 = success, nonzero = fail).
 * Failed requests retry with backoff; the outcome feeds the oracle. */
int wubu_colonel_report(wubu_colonel_t *col, int idx, int ok);

/* C5: the durable checkpoint (AD03): pack the queue so it survives
 * restart. Returns bytes written. */
long wubu_colonel_save(const wubu_colonel_t *col, void *buf, long cap);

/* C6: restore from the checkpoint. Returns the request count. */
int wubu_colonel_load(wubu_colonel_t *col, const void *buf, long n);

/* C7: the channel stats. */
void wubu_colonel_stats(const wubu_colonel_t *col, char *buf, size_t cap);

/* C8: free (does NOT free the queue the caller may own). */
void wubu_colonel_free(wubu_colonel_t *col);

#endif
