/*
 * wubu_hive_merge.h — MULTI-CHECKPOINT LINEAGE MERGE (AN47 #6:
 * "soft extinction is local. Add safe merge/compare of two .hive
 * archives (two machines or two runs) with conflict rules under
 * contracts + priority store, so the colony can federate without
 * forking fitness history"). C11.
 *
 * The merge is SAFE: it never forks fitness history — the surviving
 * lineage is decided by the conflict rules:
 *   - a cell in one archive + absent in the other -> keep it (the
 *     union)
 *   - a cell in BOTH -> keep the one with the better fitness (the
 *     loss gate), unless the priority store protects the other (the
 *     Fisher evidence: a protected cell wins even with worse fitness)
 *   - the merged ledger + graveyard are written back as a new hive
 *     archive (federated, not forked)
 *
 * Pure C11, opaque.
 */
#ifndef WUBU_HIVE_MERGE_H
#define WUBU_HIVE_MERGE_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_diagnosis.h"
#include "wubu_priority_store.h"

/* the merge result */
typedef struct {
    uint32_t kept_left;      /* cells kept from the left archive */
    uint32_t kept_right;     /* cells kept from the right archive */
    uint32_t conflicts;      /* cells in both (resolved) */
    uint32_t protected_wins; /* conflicts the priority store decided */
} wubu_merge_stats_t;

/* M1: merge two hive archives (the ledger + graveyard rings) into a
 * NEW loop (the caller's fresh wubu_diag_loop_t with its own rings).
 * Each checkpoint brings its OWN priority evidence (two machines, two
 * runs). The conflict rule: better fitness wins, UNLESS the left
 * store protects the left record (or the right store the right one)
 * — the Fisher evidence overrides the loss gate for that lineage.
 * Returns 0 on success. */
int wubu_hive_merge(const wubu_diag_loop_t *left, const wubu_diag_loop_t *right,
                    const wubu_priority_store_t *left_prio,
                    const wubu_priority_store_t *right_prio,
                    wubu_diag_loop_t *out, wubu_merge_stats_t *stats);

/* M2: the merge stats string. */
void wubu_merge_stats_str(const wubu_merge_stats_t *s, char *buf, size_t cap);

#endif
