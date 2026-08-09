/*
 * wubu_anomaly.c — the A8 ANOMALY DETECTOR (post-AN54 A8): scan one
 * event log for the release-gate anti-patterns that a rising suite
 * can HIDE:
 *
 *   1. SUITE-WHILE-VIOLATING: any window where the suite rose while
 *      contract violations tripped — the colony "improved" by
 *      breaking the floor. FAILS the release gate.
 *   2. FLAT-LOSS-WITH-ACCEPTS: the loss never moves but mutations
 *      keep getting accepted — the gate is being gamed (or the loss
 *      is scripted, the DA lesson).
 *   3. ACCEPT-ALL / REJECT-ALL: a degenerate gate (rate > 0.99 or
 *      < 0.01 across the whole run — the gate is not deciding).
 *   4. SKILL-LOSS DECOUPLE: skills accumulate but the suite never
 *      moves (the skills don't help the tasks).
 *   5. NON-MONOTONE RESUME: the round counter re-stamps (a resume
 *      that lost its place).
 *
 * Usage: ./wubu_anomaly BASE.st.events.jsonl
 * Exit: 0 = clean, 1 = anomalies found (the release gate fails).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_MAX 4096
#define WIN 64

int main(int argc, char **argv)
{
    if (argc < 2) { printf("usage: %s BASE.st.events.jsonl\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { printf("cannot open %s\n", argv[1]); return 1; }

    long n = 0, n_accept = 0, n_viol = 0, n_skills = 0, last_skill = 0;
    long re_stamps = 0, last_round = -1;
    double suite_first = 0, suite_last = 0, suite_max = 0;
    double loss_first = 0, loss_last = 0;
    double suite_win[WIN], loss_win[WIN];
    int nwin = 0, suite_rising = 0, viol_win = 0;
    int suite_while_viol = 0, flat_accepts = 0, decouple = 0;
    int anomalies = 0;
    char line[LINE_MAX];

    while (fgets(line, sizeof(line), f)) {
        long round, verdict, viol, skill;
        double loss, suite;
        if (sscanf(line, "{\"round\":%ld,\"loss\":%lf,\"suite\":%lf,"
                   "\"verdict\":%ld,", &round, &loss, &suite, &verdict) != 4)
            continue;   /* the group-commit tear */
        const char *pv = strstr(line, "\"contract_violations\":");
        const char *ps = strstr(line, "\"skill\":");
        if (!pv || !ps) continue;
        viol = atol(pv + 22);
        skill = atol(ps + 8);
        if (round <= last_round) { re_stamps++; continue; }
        last_round = round;
        n++;
        if (n == 1) { suite_first = suite; loss_first = loss; }
        suite_last = suite; loss_last = loss;
        if (suite > suite_max) suite_max = suite;
        if (verdict == 1) n_accept++;
        n_viol += viol;
        last_skill = skill;

        /* the rolling window (WIN wide) */
        if (nwin < WIN) suite_win[nwin] = suite, loss_win[nwin] = loss, nwin++;
        else {
            memmove(suite_win, suite_win + 1, (WIN - 1) * sizeof(double));
            memmove(loss_win, loss_win + 1, (WIN - 1) * sizeof(double));
            suite_win[WIN - 1] = suite; loss_win[WIN - 1] = loss;
        }
        if (nwin == WIN) {
            /* the window trend (the first vs last half) */
            double fh = 0, lh = 0;
            for (int i = 0; i < WIN / 2; i++) fh += suite_win[i], lh += suite_win[WIN / 2 + i];
            int rising = (lh / (WIN / 2)) > (fh / (WIN / 2)) + 0.005;
            if (rising && viol > 0) suite_while_viol++;
            if (rising) suite_rising++;
            if (viol > 0) viol_win++;
        }
    }
    fclose(f);
    if (n == 0) { printf("no events\n"); return 1; }

    printf("=== ANOMALY DETECTOR (the A8 release gate) ===\n");
    printf("  events=%ld accept=%ld viol=%ld suite %.4f->%.4f loss "
           "%.4f->%.4f skills=%ld\n", n, n_accept, n_viol,
           suite_first, suite_last, loss_first, loss_last, last_skill);

    /* 1. suite-while-violating (the hard veto) */
    if (suite_while_viol > 0) {
        printf("  ANOMALY: the suite ROSE while contracts tripped in %d "
               "windows — the colony improved by breaking the floor\n",
               suite_while_viol);
        anomalies++;
    } else {
        printf("  clean: no suite-while-violating windows\n");
    }

    /* 2. flat loss with accepts (the DA lesson: a scripted/gamed loss) */
    if (fabs(loss_last - loss_first) < 1e-4 && n_accept > 0) {
        printf("  ANOMALY: the loss NEVER moved (%.6f -> %.6f) but %ld "
               "mutations were accepted — the gate is not seeing real "
               "improvement\n", loss_first, loss_last, n_accept);
        anomalies++;
    } else {
        printf("  clean: the loss moved %.4f -> %.4f\n", loss_first, loss_last);
    }

    /* 3. degenerate gate */
    double rate = (double)n_accept / (double)n;
    if (rate > 0.99 || rate < 0.01) {
        printf("  ANOMALY: the accept rate %.3f is degenerate — the gate "
               "is not deciding\n", rate);
        anomalies++;
    } else {
        printf("  clean: the accept rate %.3f is a real policy\n", rate);
    }

    /* 4. skills without suite progress */
    if (last_skill > 10 && (suite_last - suite_first) < 0.01) {
        printf("  ANOMALY: %ld skills accumulated but the suite never moved "
               "(%.4f -> %.4f) — the skills don't help the tasks\n",
               last_skill, suite_first, suite_last);
        anomalies++;
    } else {
        printf("  clean: skills %ld track the suite\n", last_skill);
    }

    /* 5. resume re-stamps */
    if (re_stamps > 0) {
        printf("  WARNING: %ld non-monotone round re-stamps (a resume "
               "that lost its place)\n", re_stamps);
    }

    if (anomalies == 0) {
        printf("\n  RELEASE GATE: PASS (no anomalies)\n");
        return 0;
    }
    printf("\n  RELEASE GATE: FAIL (%d anomalies — do not ship)\n", anomalies);
    return 1;
}
