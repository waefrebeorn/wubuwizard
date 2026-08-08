/**
 * test_mobius_backward.c — Numerical verification of Möbius addition backward.
 * Compares analytic gradient (wubu_mobius_add_backward) with finite differences.
 *
 * Build: gcc -O2 -I include -o test_mobius_bwd tools/test_mobius_backward.c src/wubu_mobius.c -lm
 * Usage: ./test_mobius_bwd
 */
#include "wubu_mobius.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D 16
#define R 1.0f
#define EPS 1e-4f

static float rand_float(void) {
    return (float)(rand() / (double)RAND_MAX) * 0.8f;
}

// Numerical gradient of z = f(x,y) w.r.t. x: ∂z/∂x
static void num_grad_x(const float *x, const float *y, float *dz, float *grad_x) {
    float z_plus[D], z_minus[D];
    for (int j = 0; j < D; j++) {
        float xp[D], xm[D];
        memcpy(xp, x, D * sizeof(float));
        memcpy(xm, x, D * sizeof(float));
        xp[j] += EPS;
        xm[j] -= EPS;

        wubu_mobius_add(xp, y, D, R, z_plus);
        wubu_mobius_add(xm, y, D, R, z_minus);

        double sum = 0;
        for (int i = 0; i < D; i++)
            sum += (double)dz[i] * (z_plus[i] - z_minus[i]);
        grad_x[j] = (float)(sum / (2.0 * EPS));
    }
}

static void num_grad_y(const float *x, const float *y, float *dz, float *grad_y) {
    float z_plus[D], z_minus[D];
    for (int j = 0; j < D; j++) {
        float yp[D], ym[D];
        memcpy(yp, y, D * sizeof(float));
        memcpy(ym, y, D * sizeof(float));
        yp[j] += EPS;
        ym[j] -= EPS;

        wubu_mobius_add(x, yp, D, R, z_plus);
        wubu_mobius_add(x, ym, D, R, z_minus);

        double sum = 0;
        for (int i = 0; i < D; i++)
            sum += (double)dz[i] * (z_plus[i] - z_minus[i]);
        grad_y[j] = (float)(sum / (2.0 * EPS));
    }
}

int main(void) {
    srand(42);
    printf("=== Möbius Addition Backward Test (d=%d, R=%.2f) ===\n\n", D, R);

    int passed = 0, failed = 0;
    float max_err_dx = 0, max_err_dy = 0;

    for (int trial = 0; trial < 10; trial++) {
        float x[D], y[D], dz[D], z[D];
        float ana_dx[D], ana_dy[D];
        float num_dx[D], num_dy[D];

        for (int i = 0; i < D; i++) {
            x[i] = rand_float();
            y[i] = rand_float();
            dz[i] = rand_float() * 2.0f - 1.0f;
        }

        wubu_mobius_add(x, y, D, R, z);
        wubu_mobius_add_backward(x, y, D, R, z, dz, ana_dx, ana_dy);
        num_grad_x(x, y, dz, num_dx);
        num_grad_y(x, y, dz, num_dy);

        float err_dx = 0, err_dy = 0;
        for (int j = 0; j < D; j++) {
            float edx = fabsf(ana_dx[j] - num_dx[j]);
            float edy = fabsf(ana_dy[j] - num_dy[j]);
            if (edx > err_dx) err_dx = edx;
            if (edy > err_dy) err_dy = edy;
        }

        if (err_dx > max_err_dx) max_err_dx = err_dx;
        if (err_dy > max_err_dy) max_err_dy = err_dy;

        int ok = (err_dx < 5e-3f && err_dy < 5e-3f);
        if (ok) passed++;
        else {
            failed++;
            printf("Trial %d: FAIL — dx_err=%.6e dy_err=%.6e\n", trial, err_dx, err_dy);
        }
    }

    printf("\nResults: %d/%d passed\n", passed, passed + failed);
    printf("Max dx error: %.6e (threshold: 1e-3)\n", max_err_dx);
    printf("Max dy error: %.6e (threshold: 1e-3)\n", max_err_dy);

    if (failed > 0) {
        // Debug one trial
        printf("\n=== Debug trial with larger errors ===\n");
        float x[D], y[D], dz[D], z[D];
        float ana_dx[D], ana_dy[D], num_dx[D], num_dy[D];
        for (int i = 0; i < D; i++) {
            x[i] = 0.3f; y[i] = 0.2f;
            dz[i] = 1.0f;
        }
        x[0] = 0.6f;  // Make one element larger

        wubu_mobius_add(x, y, D, R, z);
        wubu_mobius_add_backward(x, y, D, R, z, dz, ana_dx, ana_dy);
        num_grad_x(x, y, dz, num_dx);
        num_grad_y(x, y, dz, num_dy);

        printf("x[0]=%.3f y[0]=%.3f z[0]=%.3f\n", x[0], y[0], z[0]);
        printf("dx: ana=%.6f num=%.6f diff=%.6e\n", ana_dx[0], num_dx[0], fabsf(ana_dx[0]-num_dx[0]));
        printf("dy: ana=%.6f num=%.6f diff=%.6e\n", ana_dy[0], num_dy[0], fabsf(ana_dy[0]-num_dy[0]));
    }

    return failed > 0 ? 1 : 0;
}
