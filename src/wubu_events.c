/*
 * wubu_events.c -- the COLONY RUN RECORDER (see the header).
 * Append-only JSONL, fsynced per event, readable by the post-mortem
 * tools (report A2, time-series A3, replay-verify A7).
 */
#include "wubu_events.h"

#include <stdlib.h>
#include <string.h>

int wubu_events_open_batch(wubu_events_t *ev, const char *path, int batch_size)
{
    if (!ev || !path) return -1;
    memset(ev, 0, sizeof(*ev));
    ev->f = fopen(path, "a");   /* append: a resume continues the stream */
    if (!ev->f) return -1;
    snprintf(ev->path, sizeof(ev->path), "%s", path);
    ev->batch_size = batch_size > 1 ? batch_size : 1;
    return 0;
}

int wubu_events_open(wubu_events_t *ev, const char *path)
{
    return wubu_events_open_batch(ev, path, 1);   /* the safe default */
}

int wubu_events_append(wubu_events_t *ev, const wubu_event_t *e)
{
    if (!ev || !ev->f || !e) return -1;
    /* one JSON line per event (the fixed field order is the schema) */
    fprintf(ev->f,
            "{\"round\":%llu,\"loss\":%.4f,\"suite\":%.4f,"
            "\"verdict\":%d,\"reason\":%d,\"rate\":%.3f,\"floor\":%.3f,"
            "\"contract_checks\":%llu,\"contract_violations\":%llu,"
            "\"cell\":%u,\"fisher\":%.4f,\"skill\":%u,\"traj\":%llu}\n",
            (unsigned long long)e->round, (double)e->loss,
            (double)e->suite_score, e->verdict, e->policy_reason,
            (double)e->mutation_rate, (double)e->fitness_floor,
            (unsigned long long)e->n_contract_checks,
            (unsigned long long)e->n_contract_violations,
            e->cell_idx, (double)e->prio_fisher, e->skill_version,
            (unsigned long long)e->traj_id);
    ev->n_written++;
    /* the GROUP COMMIT (the DB WAL standard — one fsync per batch, not
     * per event: the naive per-event fsync costs ~5ms each and the
     * multi-hour run emits thousands of events) */
    ev->batch_pending++;
    if (ev->batch_pending >= ev->batch_size) {
        fflush(ev->f);
        fsync(fileno(ev->f));
        ev->batch_pending = 0;
    }
    return 0;
}

void wubu_events_close(wubu_events_t *ev)
{
    if (!ev) return;
    if (ev->f) fclose(ev->f);
    ev->f = NULL;
}

int wubu_events_read(const char *path, wubu_event_t *out, int cap)
{
    if (!path || !out || cap <= 0) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int n = 0;
    char line[1024];
    while (n < cap && fgets(line, sizeof(line), f)) {
        /* parse the fixed-order JSONL (sscanf on the numeric fields) */
        wubu_event_t e;
        memset(&e, 0, sizeof(e));
        unsigned long long r, cc, cv, tj;
        unsigned int cell, sk;
        if (sscanf(line,
                   "{\"round\":%llu,\"loss\":%f,\"suite\":%f,"
                   "\"verdict\":%d,\"reason\":%d,\"rate\":%f,\"floor\":%f,"
                   "\"contract_checks\":%llu,\"contract_violations\":%llu,"
                   "\"cell\":%u,\"fisher\":%f,\"skill\":%u,\"traj\":%llu}",
                   &r, &e.loss, &e.suite_score, &e.verdict, &e.policy_reason,
                   &e.mutation_rate, &e.fitness_floor, &cc, &cv, &cell,
                   &e.prio_fisher, &sk, &tj) == 13) {
            e.round = r;
            e.n_contract_checks = cc;
            e.n_contract_violations = cv;
            e.cell_idx = (uint8_t)cell;
            e.skill_version = sk;
            e.traj_id = tj;
            out[n++] = e;
        }
    }
    fclose(f);
    return n;
}
