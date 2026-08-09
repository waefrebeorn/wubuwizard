/*
 * wubu_resources.h — THE RESOURCE LEDGER (post-AN54 A6). C11.
 *
 * Per-window RSS, CPU time, and a bandwidth estimate (events/sec as
 * the throughput proxy), read from the OS (getrusage + /proc/self).
 * The metadiag can treat the ledger as SOFT fitness — a mutation
 * that wins loss but blows the memory budget or stalls throughput
 * gets penalized, not just the loss.
 *
 * Pure C11 (POSIX getrusage + procfs — the WSL/Linux host).
 */
#ifndef WUBU_RESOURCES_H
#define WUBU_RESOURCES_H

#include <stdint.h>

/* the per-window resource snapshot */
typedef struct {
    uint64_t rss_kb;        /* the current RSS (KB) */
    double   cpu_sec;       /* the cumulative CPU seconds */
    double   cpu_delta;     /* the CPU seconds since the last snapshot */
    double   wall_delta;    /* the wall seconds since the last snapshot */
    double   throughput;    /* the events/sec (the bandwidth proxy) */
    double   soft_fitness;  /* 0..1: how healthy the resources are
                               (1 = comfortable, 0 = blown budget) */
    uint64_t budget_kb;     /* the caller's memory budget (0 = none) */
} wubu_res_t;

/* R1: snapshot the current resources. */
void wubu_res_snapshot(wubu_res_t *r);

/* R2: the windowed update — call every N rounds: fills the deltas +
 * the throughput (events since the last call / wall time) + the soft
 * fitness (RSS within the budget + CPU not spiking). */
void wubu_res_update(wubu_res_t *r, uint64_t events_since_last);

/* R3: set the memory budget (the soft fitness uses it). */
void wubu_res_set_budget(wubu_res_t *r, uint64_t budget_kb);

/* A3: the RESOURCE LEDGER CELL — a hive meta-cell snapshot (the walk
 * + the report can read the resource history, not just the live
 * counters). The runner inserts one every N rounds. */
typedef struct {
    uint64_t batch;        /* when the snapshot was taken */
    uint64_t rss_kb;       /* the RSS then */
    double   cpu_sec;      /* the cumulative CPU then */
    double   throughput;   /* the events/sec then */
    float    soft_fitness; /* the resource health then (0..1) */
} wubu_res_cell_t;

#endif
