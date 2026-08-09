/*
 * wubu_runcompare.c — the A7 RUN-COMPARE tool (post-AN54 A7): two
 * event logs -> a delta report on the metrics that matter:
 *   loss curve (start/end/mean), suite (start/end/mean), accept rate,
 *   reject rate, contract violations, skills created, and the
 *   per-phase deltas (first-window vs last-window). The colony
 *   operator uses it to answer "did the new policy make the run
 *   better?" with numbers, not vibes.
 *
 * Usage: ./wubu_runcompare BASE_A.st.events.jsonl BASE_B.st.events.jsonl
 * The events are the append-only JSONL the recorder writes (the group-
 * commit tear on the trailing line is tolerated).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_MAX 4096

typedef struct {
    const char *path;
    long n;
    double loss_first, loss_last, loss_mean, loss_min;
    double suite_first, suite_last, suite_mean, suite_max;
    long n_accept, n_reject, n_violations, n_skills;
    double window_ratio;   /* last-window mean / first-window mean */
} run_stats_t;

static int parse_line(const char *line, long *round, double *loss,
                      double *suite, long *verdict, long *viol,
                      long *skill)
{
    /* the JSONL fields: round, loss, suite, verdict, reason, rate,
     * floor, contract_checks, contract_violations, cell, fisher,
     * skill, traj — a minimal tolerant parse (the field order is
     * fixed by the recorder) */
    if (sscanf(line, "{\"round\":%ld,\"loss\":%lf,\"suite\":%lf,"
               "\"verdict\":%ld,", round, loss, suite, verdict) != 4)
        return -1;
    /* the violations + skill counts via a second pass (they come later
     * in the line) */
    const char *pv = strstr(line, "\"contract_violations\":");
    const char *ps = strstr(line, "\"skill\":");
    if (!pv || !ps) return -1;
    *viol = atol(pv + 22);
    *skill = atol(ps + 8);
    return 0;
}

static int scan(const char *path, run_stats_t *st)
{
    memset(st, 0, sizeof(*st));
    st->path = path;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[LINE_MAX];
    double loss_sum = 0, suite_sum = 0;
    double first_win[10] = {0}, last_win[10] = {0};
    int n_first = 0, n_last = 0;
    long last_round = -1;
    st->loss_min = 1e30;
    while (fgets(line, sizeof(line), f)) {
        long round, verdict, viol, skill;
        double loss, suite;
        if (parse_line(line, &round, &loss, &suite, &verdict, &viol, &skill) != 0)
            continue;   /* the group-commit tear (dropped) */
        if (round <= last_round) continue;   /* a resume re-stamp */
        last_round = round;
        st->n++;
        loss_sum += loss; suite_sum += suite;
        if (loss < st->loss_min) st->loss_min = loss;
        if (suite > st->suite_max) st->suite_max = suite;
        if (st->n == 1) { st->loss_first = loss; st->suite_first = suite; }
        st->loss_last = loss; st->suite_last = suite;
        if (verdict == 1) st->n_accept++; else st->n_reject++;
        st->n_violations += viol;
        st->n_skills = skill;   /* the cumulative skill count at this
                                   round (the field is a counter, not
                                   a per-round delta — summing it
                                   double-counts; the last value is
                                   the run's skill total) */
        if (st->n <= 10) { first_win[n_first++] = suite; }
        if (n_last < 10) last_win[n_last++] = suite;
        if (n_last == 10) { memmove(last_win, last_win + 1, 9 * sizeof(double)); n_last = 9; }
    }
    fclose(f);
    if (st->n == 0) return -1;
    st->loss_mean = loss_sum / st->n;
    st->suite_mean = suite_sum / st->n;
    double fs = 0, ls = 0;
    for (int i = 0; i < n_first; i++) fs += first_win[i];
    for (int i = 0; i < n_last; i++) ls += last_win[i];
    if (n_first) fs /= n_first;
    if (n_last) ls /= n_last;
    st->window_ratio = (fs > 1e-9) ? ls / fs : 0;
    return 0;
}

static void print_stats(const char *label, const run_stats_t *s)
{
    printf("%s:\n", label);
    printf("  events=%ld loss %.4f->%.4f (mean %.4f, min %.4f)\n",
           s->n, s->loss_first, s->loss_last, s->loss_mean, s->loss_min);
    printf("  suite %.4f->%.4f (mean %.4f, max %.4f)  "
           "window-ratio %.3f\n",
           s->suite_first, s->suite_last, s->suite_mean, s->suite_max,
           s->window_ratio);
    printf("  accept=%ld reject=%ld rate=%.3f  violations=%ld skills=%ld\n",
           s->n_accept, s->n_reject,
           (s->n_accept + s->n_reject) > 0
               ? (double)s->n_accept / (double)(s->n_accept + s->n_reject) : 0,
           s->n_violations, s->n_skills);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s BASE_A.events.jsonl BASE_B.events.jsonl\n", argv[0]);
        return 1;
    }
    run_stats_t a, b;
    if (scan(argv[1], &a) != 0) { printf("cannot read %s\n", argv[1]); return 1; }
    if (scan(argv[2], &b) != 0) { printf("cannot read %s\n", argv[2]); return 1; }

    printf("=== RUN-COMPARE (the A7 delta) ===\n");
    print_stats("  A", &a);
    print_stats("  B", &b);
    printf("\n  DELTAS (B - A):\n");
    printf("    suite end     %+.4f (%s)\n", b.suite_last - a.suite_last,
           b.suite_last >= a.suite_last ? "BETTER" : "worse");
    printf("    suite mean    %+.4f\n", b.suite_mean - a.suite_mean);
    printf("    suite window  %+.3f (the last-window/first-window ratio)\n",
           b.window_ratio - a.window_ratio);
    printf("    loss end      %+.4f (%s)\n", b.loss_last - a.loss_last,
           b.loss_last <= a.loss_last ? "BETTER" : "worse");
    printf("    accept rate   %+.3f\n",
           (b.n_accept + b.n_reject) > 0 && (a.n_accept + a.n_reject) > 0
               ? (double)b.n_accept / (double)(b.n_accept + b.n_reject)
                 - (double)a.n_accept / (double)(a.n_accept + a.n_reject)
               : 0);
    printf("    violations    %+ld\n", b.n_violations - a.n_violations);

    /* the verdict: ends within a tie band -> the mean decides; any
     * violation increase is a hard veto */
    double tie = 0.002;
    int verdict = 0;
    if (b.n_violations > a.n_violations) {
        verdict = -1;   /* the safety veto — never ship a run that
                           tripped more contracts */
    } else if (b.suite_last > a.suite_last + tie) {
        verdict = 1;    /* a real end improvement */
    } else if (b.suite_last < a.suite_last - tie) {
        verdict = -1;   /* the end regressed */
    } else if (b.suite_mean > a.suite_mean + tie) {
        verdict = 1;    /* ends tied, the mean improved */
    } else if (b.suite_mean < a.suite_mean - tie) {
        verdict = -1;
    }
    printf("\n  VERDICT: %s\n",
           verdict > 0 ? "B is the better run (suite up, safety held)"
           : verdict < 0 ? "B is UNSAFE or regressed — do not ship"
           : "no clear winner (the suites are statistically tied)");
    return verdict < 0 ? 1 : 0;
}
