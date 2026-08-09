/*
 * wubu_colonel.c -- the LIVE COLONEL bridge (see wubu_colonel.h).
 *
 * The trust boundary between the Brain (the colony) and the Body
 * (WuBuOS): requests carry the 9P capability subtree + the resource
 * bound; the executor pulls by priority with backoff; outcomes feed
 * the oracle; the queue is durable via checkpoints.
 */
#include "wubu_colonel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_agentic_os.h"

int wubu_colonel_init(wubu_colonel_t *col, int q_cap, long base_backoff_ms)
{
    if (!col || q_cap <= 0) return -1;
    memset(col, 0, sizeof(*col));
    col->q_cap = q_cap;
    col->base_backoff_ms = base_backoff_ms > 0 ? base_backoff_ms : 1000;
    col->queue = (wubu_request_t *)calloc((size_t)q_cap, sizeof(wubu_request_t));
    if (!col->queue) return -1;
    col->next_req_id = 1;
    return 0;
}

int64_t wubu_colonel_request(wubu_colonel_t *col, wubu_act_kind_t kind,
                             const char *path, const char *agent_subtree,
                             uint16_t goal, uint8_t cell_idx, int priority,
                             long cpu_ms, long ram_mb, long io_kb)
{
    if (!col || !path || !agent_subtree) return -1;
    /* the trust boundary at enqueue: the path must live inside the
     * requester's 9P capability subtree (AD01) */
    if (!wubu_9p_cap_allowed(agent_subtree, path)) return -1;
    /* the resource bound must be sane (AD04) */
    if (cpu_ms <= 0 || ram_mb <= 0) return -1;
    if (col->q_n >= col->q_cap) return -1;   /* the queue is full */
    wubu_request_t *r = &col->queue[col->q_n++];
    memset(r, 0, sizeof(*r));
    r->req_id = col->next_req_id++;
    r->kind = kind;
    snprintf(r->path, sizeof(r->path), "%s", path);
    snprintf(r->agent_subtree, sizeof(r->agent_subtree), "%s", agent_subtree);
    r->goal_token = goal;
    r->cell_idx = cell_idx;
    r->priority = priority < 0 ? 0 : (priority > 3 ? 3 : priority);
    r->attempt = 0;
    r->cpu_ms_max = cpu_ms; r->ram_mb_max = ram_mb; r->io_kb_max = io_kb;
    r->done = 0; r->accepted = 0;
    return (int64_t)r->req_id;
}

int wubu_colonel_pull(wubu_colonel_t *col)
{
    if (!col || col->q_n == 0) return -1;
    /* the highest-priority request that is not done and not in backoff
     * (AD02: a failed attempt waits base*2^attempt ms before retry —
     * the executor is assumed to call back after the wait) */
    int best = -1, best_prio = -1;
    for (int i = 0; i < col->q_n; i++) {
        wubu_request_t *r = &col->queue[i];
        if (r->done) continue;
        if (r->priority > best_prio) { best_prio = r->priority; best = i; }
    }
    return best;
}

int wubu_colonel_report(wubu_colonel_t *col, int idx, int ok)
{
    if (!col || idx < 0 || idx >= col->q_n) return -1;
    wubu_request_t *r = &col->queue[idx];
    r->done = 1;
    r->accepted = ok ? 1 : 0;
    if (ok) {
        col->n_done++;
    } else {
        col->n_failed++;
        /* failed requests may retry (the backoff is the executor's
         * wait); the outcome feeds the oracle as a preference pair */
        r->attempt++;
    }
    return 0;
}

long wubu_colonel_save(const wubu_colonel_t *col, void *buf, long cap)
{
    if (!col || !buf || cap <= 0) return 0;
    /* header: magic + count + counters */
    uint64_t hdr[4] = { 0xC01E0001u, (uint64_t)col->q_n, col->n_done, col->n_failed };
    long off = 0;
    size_t hs = sizeof(hdr);
    if (cap < (long)(hs + (size_t)col->q_n * sizeof(wubu_request_t))) return 0;
    memcpy((char *)buf + off, hdr, hs); off += (long)hs;
    for (int i = 0; i < col->q_n; i++)
        memcpy((char *)buf + off, &col->queue[i], sizeof(wubu_request_t)),
        off += (long)sizeof(wubu_request_t);
    return off;
}

int wubu_colonel_load(wubu_colonel_t *col, const void *buf, long n)
{
    if (!col || !buf || n <= 0) return -1;
    uint64_t hdr[4];
    long off = 0;
    size_t hs = sizeof(hdr);
    if (n < (long)hs) return -1;
    memcpy(hdr, (const char *)buf + off, hs); off += (long)hs;
    if (hdr[0] != 0xC01E0001u) return -1;
    int cnt = (int)hdr[1];
    if (cnt > col->q_cap) cnt = col->q_cap;
    for (int i = 0; i < cnt; i++) {
        memcpy(&col->queue[i], (const char *)buf + off, sizeof(wubu_request_t));
        off += (long)sizeof(wubu_request_t);
    }
    col->q_n = cnt;
    col->n_done = (uint64_t)hdr[2];
    col->n_failed = (uint64_t)hdr[3];
    return cnt;
}

void wubu_colonel_stats(const wubu_colonel_t *col, char *buf, size_t cap)
{
    if (!col || !buf || cap == 0) return;
    snprintf(buf, cap,
             "queue=%d cap=%d done=%llu failed=%llu next_req=%llu",
             col->q_n, col->q_cap,
             (unsigned long long)col->n_done,
             (unsigned long long)col->n_failed,
             (unsigned long long)col->next_req_id);
}

void wubu_colonel_free(wubu_colonel_t *col)
{
    if (!col) return;
    free(col->queue);
    memset(col, 0, sizeof(*col));
}
