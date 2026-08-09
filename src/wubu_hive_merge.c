/*
 * wubu_hive_merge.c — MULTI-CHECKPOINT LINEAGE MERGE (see header).
 * Federate without forking.
 */
#include "wubu_hive_merge.h"

#include <stdio.h>
#include <string.h>

#include "wubu_priority_store.h"

int wubu_hive_merge(const wubu_diag_loop_t *left, const wubu_diag_loop_t *right,
                    const wubu_priority_store_t *left_prio,
                    const wubu_priority_store_t *right_prio,
                    wubu_diag_loop_t *out, wubu_merge_stats_t *stats)
{
    if (!left || !right || !out) return -1;
    memset(stats, 0, sizeof(*stats));

    /* 1. the union of the ledgers: for each cell in both, the better
     * fitness wins — unless the priority evidence protects it */
    for (int i = 0; i < left->ledger_n; i++) {
        const wubu_fitness_cell_t *a = &left->ledger[i];
        int conflict = 0;
        for (int j = 0; j < right->ledger_n; j++) {
            const wubu_fitness_cell_t *b = &right->ledger[j];
            if (a->batch == b->batch && a->cell_idx == b->cell_idx) {
                conflict = 1;
                stats->conflicts++;
                const wubu_fitness_cell_t *win = a;
                if (b->fitness < a->fitness) win = b;   /* lower = better */
                /* the priority evidence overrides: a PROTECTED lineage
                 * (its checkpoint's Fisher says the loss cares + a
                 * recent rejection) wins even with worse fitness */
                int prot_a = 0, prot_b = 0;
                if (left_prio)
                    for (int p = 0; p < left_prio->n; p++)
                        if (left_prio->cells[p].cell_idx == a->cell_idx &&
                            left_prio->cells[p].protected) prot_a = 1;
                if (right_prio)
                    for (int p = 0; p < right_prio->n; p++)
                        if (right_prio->cells[p].cell_idx == b->cell_idx &&
                            right_prio->cells[p].protected) prot_b = 1;
                if (prot_a && !prot_b) { win = a; stats->protected_wins++; }
                else if (prot_b && !prot_a) { win = b; stats->protected_wins++; }
                if (win == a) stats->kept_left++; else stats->kept_right++;
                /* push the winner into the merged ledger */
                if (out->ledger_n < out->ledger_cap)
                    out->ledger[out->ledger_n++] = *win;
                break;
            }
        }
        if (!conflict) {
            out->ledger[out->ledger_n++] = *a;   /* the union */
            stats->kept_left++;
        }
    }
    /* the right-only cells */
    for (int j = 0; j < right->ledger_n; j++) {
        int seen = 0;
        for (int i = 0; i < left->ledger_n; i++)
            if (left->ledger[i].batch == right->ledger[j].batch &&
                left->ledger[i].cell_idx == right->ledger[j].cell_idx) {
                seen = 1; break;
            }
        if (!seen) { out->ledger[out->ledger_n++] = right->ledger[j]; stats->kept_right++; }
    }
    /* 2. the graveyards merge as the union (the negative examples are
     * never dropped — a rejection is a fact) */
    for (int i = 0; i < left->grave_n && out->grave_n < out->grave_cap; i++)
        out->graveyard[out->grave_n++] = left->graveyard[i];
    for (int j = 0; j < right->grave_n && out->grave_n < out->grave_cap; j++)
        out->graveyard[out->grave_n++] = right->graveyard[j];
    return 0;
}

void wubu_merge_stats_str(const wubu_merge_stats_t *s, char *buf, size_t cap)
{
    if (!s || !buf || cap == 0) return;
    snprintf(buf, cap,
             "left=%u right=%u conflicts=%u protected_wins=%u",
             s->kept_left, s->kept_right, s->conflicts, s->protected_wins);
}
