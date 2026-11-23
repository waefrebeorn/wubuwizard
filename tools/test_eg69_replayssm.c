/* test_eg69_replayssm.c -- EG-69 gate: checkpoint+input-buffer SSM replay. */
#include <stdio.h>
#include <math.h>

#include "wubu_replay_ssm.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define DIM 2
#define BUF 8
#define STEPS 40

static void input_at(int t, double *alpha, float *v, float *k, float *q) {
    *alpha = 0.90 + 0.001 * (t % 5);
    for (int i = 0; i < DIM; ++i) {
        v[i] = (float)((t + i + 1) % 7) * 0.125f;
        k[i] = (float)((t + 2 * i + 3) % 5) * 0.25f;
        q[i] = (float)((t + i + 2) % 3) * 0.5f;
    }
}

int main(void) {
    wubu_replay_ssm_t *p = wubu_replay_ssm_create(DIM, BUF);
    double ref[DIM][DIM] = {{0.0, 0.0}, {0.0, 0.0}};
    double maxerr = 0.0;
    CHECK(p != NULL);

    for (int t = 0; t < STEPS; ++t) {
        float v[DIM], k[DIM], q[DIM], out[DIM];
        double alpha = 0.90 + 0.001 * (t % 5);
        for (int i = 0; i < DIM; ++i) {
            v[i] = (float)((t + i + 1) % 7) * 0.125f;
            k[i] = (float)((t + 2 * i + 3) % 5) * 0.25f;
            q[i] = (float)((t + i + 2) % 3) * 0.5f;
        }
        CHECK(wubu_replay_ssm_step(p, alpha, v, k, q, out) == 0);
        /* Independent scalar recurrence: S = alpha*S + v*k^T; o = S*q. */
        for (int i = 0; i < DIM; ++i)
            for (int j = 0; j < DIM; ++j)
                ref[i][j] = alpha * ref[i][j] + (double)v[i] * (double)k[j];
        for (int i = 0; i < DIM; ++i) {
            double want = 0.0;
            for (int j = 0; j < DIM; ++j) want += ref[i][j] * (double)q[j];
            if (fabs(out[i] - want) > maxerr) maxerr = fabs(out[i] - want);
        }
    }
    CHECK(maxerr < 1e-3);
    CHECK(wubu_replay_ssm_buffered(p) <= BUF);
    printf("EG-69 replay steps=%d dim=%d buf=%d checkpoints=%d maxerr=%.8f\n",
           STEPS, DIM, wubu_replay_ssm_buffered(p),
           wubu_replay_ssm_checkpoints(p), maxerr);

    /* Draft rollback is a buffer truncation, not a state snapshot copy. */
    {
        wubu_replay_ssm_t *a = wubu_replay_ssm_create(DIM, BUF);
        wubu_replay_ssm_t *b = wubu_replay_ssm_create(DIM, BUF);
        double rb_err = 0.0;
        double alpha; float v[DIM], k[DIM], q[DIM], oa[DIM], ob[DIM];
        CHECK(a && b);
        for (int t = 0; t < 5; ++t) {
            input_at(t, &alpha, v, k, q);
            CHECK(wubu_replay_ssm_step(a, alpha, v, k, q, oa) == 0);
            CHECK(wubu_replay_ssm_step(b, alpha, v, k, q, ob) == 0);
        }
        for (int t = 5; t < 7; ++t) {
            input_at(t, &alpha, v, k, q);
            CHECK(wubu_replay_ssm_step(a, alpha, v, k, q, oa) == 0);
        }
        CHECK(wubu_replay_ssm_buffered(a) == 7);
        CHECK(wubu_replay_ssm_rollback(a, 5) == 0);
        CHECK(wubu_replay_ssm_buffered(a) == 5);
        CHECK(wubu_replay_ssm_checkpoints(a) == 0);
        for (int t = 7; t < 10; ++t) {
            input_at(t, &alpha, v, k, q);
            CHECK(wubu_replay_ssm_step(a, alpha, v, k, q, oa) == 0);
            CHECK(wubu_replay_ssm_step(b, alpha, v, k, q, ob) == 0);
            for (int i = 0; i < DIM; ++i) {
                double e = fabs(oa[i] - ob[i]);
                if (e > rb_err) rb_err = e;
            }
        }
        CHECK(rb_err < 1e-3);
        CHECK(wubu_replay_ssm_rollback(a, 9) == -1);
        CHECK(wubu_replay_ssm_rollback(a, -1) == -1);
        CHECK(wubu_replay_ssm_step(NULL, alpha, v, k, q, oa) == -1);
        CHECK(wubu_replay_ssm_create(0, BUF) == NULL);
        CHECK(wubu_replay_ssm_create(DIM, 0) == NULL);
        CHECK(wubu_replay_ssm_buffered(NULL) == -1);
        printf("EG-69 rollback 7->5 inputs: continuation maxerr=%.8f checkpoints=%d\n",
               rb_err, wubu_replay_ssm_checkpoints(a));
        wubu_replay_ssm_free(a);
        wubu_replay_ssm_free(b);
    }

    wubu_replay_ssm_free(p);
    if (failures) { fprintf(stderr, "test_eg69_replayssm: FAIL (%d)\n", failures); return 1; }
    printf("test_eg69_replayssm: PASS\n");
    return 0;
}
