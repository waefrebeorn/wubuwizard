/*
 * wubu_events.h -- the COLONY RUN RECORDER (post-AN54 A1). C11.
 *
 * A structured, append-only event stream (one JSON line per event)
 * written next to .hive / .prio during a colony run. The events are
 * what the post-mortem tools read:
 *   - the report generator (A2) turns them into a markdown summary
 *   - the time-series export (A3) plots suite score vs loss vs rate
 *   - the replay verifier (A7) recomputes the fitness decisions and
 *     flags divergence from the live run
 *
 * Every event carries: round/batch, loss, suite score, the mutation
 * verdict, the policy reason code, the contract-check counters, and
 * the ATTRIBUTION (which cell + which evidence justified it — A4).
 *
 * Pure C11, no third party. The file is append-only + fsynced per
 * event so a kill loses nothing.
 */
#ifndef WUBU_EVENTS_H
#define WUBU_EVENTS_H

#include <stdint.h>
#include <stdio.h>

/* one event (the JSONL line's payload) */
typedef struct {
    uint64_t round;            /* the batch/round id */
    float    loss;             /* this batch's loss */
    float    suite_score;      /* the task-suite score (task_ema) */
    int      verdict;          /* 0=reject 1=accept 2=stasis */
    int      policy_reason;    /* 0=none 1=loss rising 2=suite failing 3=relax */
    float    mutation_rate;    /* the metadiag's current rate */
    float    fitness_floor;
    uint64_t n_contract_checks;
    uint64_t n_contract_violations;
    /* the mutation attribution (A4): which cell + the evidence */
    uint8_t  cell_idx;         /* the cell that mutated */
    float    prio_fisher;      /* the priority-store Fisher at the time */
    uint32_t skill_version;    /* the skill that justified it (0 = none) */
    uint64_t traj_id;          /* the traj cell that justified it (0 = none) */
} wubu_event_t;

/* the recorder state (the file handle) */
typedef struct {
    FILE *f;
    uint64_t n_written;
    char     path[512];
    int      batch_pending;   /* events buffered since the last fsync */
    int      batch_size;      /* fsync every N events (group commit) */
} wubu_events_t;

/* E1: open the recorder (append mode — a resume continues the stream).
 * batch_size > 1 uses GROUP COMMIT (the DB WAL standard: one fsync per
 * batch instead of per event — the naive per-event fsync is 5ms+ per
 * write; SQLite/Postgres batch exactly this way). A kill loses at most
 * batch_size trailing events (acceptable: the .hive/.prio sidecars are
 * the authoritative state; the events are the telemetry).
 * Returns 0 on success. */
int wubu_events_open_batch(wubu_events_t *ev, const char *path, int batch_size);

/* E2: append one event (a JSON line) + fsync (a kill loses nothing).
 * Returns 0 on success. */
int wubu_events_append(wubu_events_t *ev, const wubu_event_t *e);

/* E3: close. */
void wubu_events_close(wubu_events_t *ev);

/* E4: read the whole stream back (the post-mortem tools). Returns the
 * count read (the caller's buffer is [cap]). */
int wubu_events_read(const char *path, wubu_event_t *out, int cap);

#endif
