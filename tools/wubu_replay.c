/*
 * wubu_replay.c — the REPLAY VERIFIER (post-AN54 A7). C11.
 * Given the hive + prio + the event log, RECOMPUTE the fitness
 * decisions offline and flag divergence from the live run.
 *
 * The offline decision per event: the mutation is ACCEPTED when the
 * loss improved (held-out < prev) within the gate's tolerance, and
 * REJECTED otherwise. The verifier compares its recomputation to the
 * LIVE verdict in the event log — divergence = a bug in the gate or
 * a run artifact.
 *
 * Usage: wubu_replay <base> [--strict]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_events.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: wubu_replay <base> [--strict]\n");
        return 1;
    }
    const char *base = argv[1];
    int strict = 0;
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--strict")) strict = 1;

    char epath[640];
    snprintf(epath, sizeof(epath), "%s.events.jsonl", base);
    wubu_event_t *ev = (wubu_event_t *)calloc(8192, sizeof(wubu_event_t));
    int n = wubu_events_read(epath, ev, 8192);
    if (n <= 0) { printf("no events at %s\n", epath); return 1; }

    /* the offline recomputation: the gate ACCEPTS unless the loss
     * WORSENED by more than the tolerance (the amoeba's loss_tol
     * 0.05 — a flat or improving loss is accepted, matching the live
     * wubu_amoeba_validate). The prev is the previous event's loss
     * (the round sequence). */
    float prev_loss = ev[0].loss;
    int n_diverged = 0, n_checked = 0;
    int n_recompute_accept = 0, n_recompute_reject = 0;
    for (int i = 0; i < n; i++) {
        float held = ev[i].loss;
        /* the deterministic gate: worsened by > 0.05 -> reject;
         * flat or improved -> accept */
        int recomputed = (held > prev_loss + 0.05f) ? 0 : 1;
        if (recomputed) n_recompute_accept++; else n_recompute_reject++;
        n_checked++;
        /* the LIVE verdict: 1 = accept, 0 = reject (skip stasis) */
        if (ev[i].verdict == 1 || ev[i].verdict == 0) {
            if (recomputed != ev[i].verdict) {
                n_diverged++;
                if (n_diverged <= 5 || strict)
                    printf("  DIVERGE round %llu: live=%d recompute=%d "
                           "(loss %.4f -> %.4f)\n",
                           (unsigned long long)ev[i].round, ev[i].verdict,
                           recomputed, prev_loss, held);
            }
        }
        prev_loss = held;
    }

    printf("\nreplay of %d events: recomputed %d accept, %d reject, "
           "%d diverged\n", n_checked, n_recompute_accept,
           n_recompute_reject, n_diverged);
    if (n_diverged == 0) {
        printf("=== REPLAY VERIFIED: the offline decisions match the live run ===\n");
        return 0;
    }
    printf("=== REPLAY DIVERGENCE: %d decisions differ from the live run "
           "(investigate the gate or the artifacts) ===\n", n_diverged);
    return 1;
}
