/*
 * wubu_timeseries.c — the SUITE-SCORE TIME SERIES (post-AN54 A3).
 * Reads <base>.events.jsonl and exports the trend so the multi-hour
 * run answers "did tasks improve when loss did?"
 *
 * Outputs:
 *   --csv  -> <base>.ts.csv  (round, loss, suite, rate, verdict)
 *   --plot -> an ASCII sparkline + the correlation of suite vs loss
 *
 * Usage: wubu_timeseries <base> [--csv] [--plot]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_events.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: wubu_timeseries <base> [--csv] [--plot]\n");
        return 1;
    }
    const char *base = argv[1];
    int do_csv = 0, do_plot = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")) do_csv = 1;
        if (!strcmp(argv[i], "--plot")) do_plot = 1;
    }

    char epath[640];
    snprintf(epath, sizeof(epath), "%s.events.jsonl", base);
    wubu_event_t *ev = (wubu_event_t *)calloc(8192, sizeof(wubu_event_t));
    int n = wubu_events_read(epath, ev, 8192);
    if (n < 0) { printf("cannot read %s\n", epath); return 1; }
    if (n == 0) { printf("no events at %s\n", epath); return 1; }
    printf("%d events from %s\n", n, epath);

    /* the CSV: the raw time series (the plot tool's input) */
    if (do_csv) {
        char cpath[640];
        snprintf(cpath, sizeof(cpath), "%s.ts.csv", base);
        FILE *f = fopen(cpath, "w");
        if (f) {
            fprintf(f, "round,loss,suite,rate,floor,verdict\n");
            for (int i = 0; i < n; i++)
                fprintf(f, "%llu,%.4f,%.4f,%.3f,%.3f,%d\n",
                        (unsigned long long)ev[i].round, ev[i].loss,
                        ev[i].suite_score, ev[i].mutation_rate,
                        ev[i].fitness_floor, ev[i].verdict);
            fclose(f);
            printf("csv -> %s\n", cpath);
        }
    }

    /* the plot: an ASCII sparkline per series + the correlation */
    if (do_plot) {
        printf("\nloss:      ");
        for (int i = 0; i < n; i += (n > 72 ? n / 72 : 1)) {
            float v = ev[i].loss;
            int h = (int)((10.0f - v) * 4.0f);
            if (h < 0) h = 0; if (h > 8) h = 8;
            printf("%c", " _-=*#@%%"[h]);
        }
        printf("\nsuite:     ");
        for (int i = 0; i < n; i += (n > 72 ? n / 72 : 1)) {
            float v = ev[i].suite_score;
            int h = (int)(v * 10.0f) - 2;
            if (h < 0) h = 0; if (h > 8) h = 8;
            printf("%c", " _-=*#@%%"[h]);
        }
        printf("\nrate:      ");
        for (int i = 0; i < n; i += (n > 72 ? n / 72 : 1)) {
            float v = ev[i].mutation_rate;
            int h = (int)(v * 8.0f);
            if (h < 0) h = 0; if (h > 8) h = 8;
            printf("%c", " _-=*#@%%"[h]);
        }
        printf("\n");

        /* the correlation: suite vs loss (negative = tasks improve
         * when loss improves — the answer to the question) */
        double sl = 0, ss = 0, ll = 0, m_s = 0, m_l = 0;
        for (int i = 0; i < n; i++) { m_s += ev[i].suite_score; m_l += ev[i].loss; }
        m_s /= n; m_l /= n;
        for (int i = 0; i < n; i++) {
            double ds = ev[i].suite_score - m_s, dl = ev[i].loss - m_l;
            sl += ds * dl; ss += ds * ds; ll += dl * dl;
        }
        double corr = (ss > 0 && ll > 0) ? sl / (sqrt(ss) * sqrt(ll)) : 0.0;
        printf("\ncorr(suite, loss) = %.3f  (%s)\n", corr,
               corr < -0.3 ? "tasks improve WITH loss — the colony works"
               : corr > 0.3 ? "tasks IMPROVE as loss WORSEENS — decoupled"
               : "no clear coupling");
        /* the DA statistical fix (2026-08-09): the LEVELS correlation
         * is inflated when both series are autocorrelated (the classic
         * spurious-regression problem — Afyouni 2019 effective-dof,
         * Granger/Newbold). The research-convergent check: correlate
         * the FIRST DIFFERENCES (the changes co-move, not the levels).
         * A real coupling survives differencing; a shared-trend
         * artifact does not. */
        double dl = 0, ds = 0, dls = 0, dss = 0, dll = 0;
        int dn = 0;
        for (int i = 1; i < n; i++) {
            double dloss = ev[i].loss - ev[i-1].loss;
            double dsu = ev[i].suite_score - ev[i-1].suite_score;
            dl += dloss; ds += dsu; dn++;
        }
        dl /= dn; ds /= dn;
        for (int i = 1; i < n; i++) {
            double dloss = ev[i].loss - ev[i-1].loss - dl;
            double dsu = ev[i].suite_score - ev[i-1].suite_score - ds;
            dls += dloss * dsu; dss += dsu * dsu; dll += dloss * dloss;
        }
        double dcorr = (dss > 0 && dll > 0) ? dls / (sqrt(dss) * sqrt(dll)) : 0.0;
        printf("corr(first-diff) = %.3f  (%s)\n", dcorr,
               dcorr < -0.3 ? "the CHANGES co-move — a real coupling"
               : dcorr > 0.3 ? "the changes co-move the WRONG way"
               : "the levels-correlation is a shared-trend artifact "
                 "(the changes do not co-move)");
        /* the first/last window comparison */
        int w = n > 20 ? n / 5 : n / 2;
        double s1 = 0, s2 = 0;
        for (int i = 0; i < w; i++) s1 += ev[i].suite_score;
        for (int i = n - w; i < n; i++) s2 += ev[i].suite_score;
        s1 /= w; s2 /= w;
        printf("suite first-window %.3f -> last-window %.3f (%+.3f)\n",
               s1, s2, s2 - s1);
    }
    free(ev);
    return 0;
}
