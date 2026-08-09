/*
 * wubu_hive_walk.c -- the HIVE WALK tool (the user directive #2: the
 * agent can invoke 'show me the last 32 cells that improved loss on
 * coding tasks').
 *
 * Opens a hive archive written by the closed loop (wubu_diagnosis)
 * and walks it: filters by verdict (accept/reject/stasis), by cell,
 * by improvement, and prints the lineage.
 *
 * The archive format (written by wubu_diag_save): a header + the
 * ledger + the graveyard rings. The walk is read-only — the Brain's
 * memory becomes visible and versionable to the Body.
 *
 * Usage:
 *   wubu_hive_walk <archive.bin> [--accepted|--rejected|--stasis]
 *                  [--last N] [--min-improve X]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "wubu_diagnosis.h"

/* the archive file header (matches wubu_diag_save) */
typedef struct {
    uint32_t magic;          /* 0xD1A60001 */
    uint32_t ledger_n;
    uint32_t grave_n;
    uint64_t batch;
} hive_archive_hdr_t;

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: wubu_hive_walk <archive.bin> [--accepted|--rejected|--stasis] [--last N] [--min-improve X]\n");
        return 1;
    }
    const char *path = argv[1];
    int only_accept = 0, only_reject = 0, only_stasis = 0;
    int last_n = 0;
    float min_improve = 0.0f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--accepted")) only_accept = 1;
        else if (!strcmp(argv[i], "--rejected")) only_reject = 1;
        else if (!strcmp(argv[i], "--stasis")) only_stasis = 1;
        else if (!strcmp(argv[i], "--last") && i + 1 < argc) last_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-improve") && i + 1 < argc) min_improve = atof(argv[++i]);
    }

    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    hive_archive_hdr_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != 0xD1A60001u) {
        printf("not a hive archive\n");
        fclose(f);
        return 1;
    }
    wubu_fitness_cell_t *ledger = (wubu_fitness_cell_t *)calloc(hdr.ledger_n ? hdr.ledger_n : 1, sizeof(wubu_fitness_cell_t));
    wubu_fitness_cell_t *grave = (wubu_fitness_cell_t *)calloc(hdr.grave_n ? hdr.grave_n : 1, sizeof(wubu_fitness_cell_t));
    if (hdr.ledger_n) fread(ledger, sizeof(wubu_fitness_cell_t), hdr.ledger_n, f);
    if (hdr.grave_n) fread(grave, sizeof(wubu_fitness_cell_t), hdr.grave_n, f);
    fclose(f);

    printf("=== hive walk: %s (batch %llu, %u accepted, %u rejected) ===\n",
           path, (unsigned long long)hdr.batch, hdr.ledger_n, hdr.grave_n);

    /* print the ledger (accepted mutations — the lineage) */
    int shown = 0;
    for (uint32_t i = 0; i < hdr.ledger_n; i++) {
        wubu_fitness_cell_t *c = &ledger[i];
        if (only_reject || only_stasis) continue;
        if (min_improve > 0 && c->delta >= -min_improve) continue;  /* delta < 0 = improved */
        printf("  [accepted] batch %llu fit %.4f delta %+.4f cell %u\n",
               (unsigned long long)c->batch, c->fitness, c->delta, c->cell_idx);
        if (last_n && ++shown >= last_n) break;
    }
    /* the graveyard (negative examples — useful) */
    shown = 0;
    for (uint32_t i = 0; i < hdr.grave_n; i++) {
        wubu_fitness_cell_t *c = &grave[i];
        if (only_accept || only_stasis) continue;
        if (min_improve > 0 && c->delta >= -min_improve) continue;
        printf("  [rejected] batch %llu fit %.4f delta %+.4f cell %u\n",
               (unsigned long long)c->batch, c->fitness, c->delta, c->cell_idx);
        if (last_n && ++shown >= last_n) break;
    }
    free(ledger);
    free(grave);
    return 0;
}
