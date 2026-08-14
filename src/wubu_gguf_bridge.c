#include "gguf_reader.h"
#include "wubu_gguf_tensor.h"
#include <math.h>
#include <string.h>

// ========== Poincare / Hyperbolic Ops ==========

float wubu_norm(const float *v, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)v[i] * v[i];
    return (float)sqrt(sum);
}

float wubu_dot(const float *a, const float *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)a[i] * b[i];
    return (float)sum;
}

void wubu_exp_map(const float *input, int dim, float R, float *output) {
    // Map Euclidean vector to Poincare ball:
    // exp_map(v) = R * tanh(||v||/R) x v/||v||
    // Output norm = R * tanh(||v||/R) < R
    float norm = wubu_norm(input, dim);
    if (norm < 1e-8f) {
        memcpy(output, input, dim * sizeof(float));
        return;
    }
    float factor = R * tanhf(norm / R) / norm;
    for (int i = 0; i < dim; i++) {
        output[i] = factor * input[i];
    }
}

void wubu_log_map(const float *input, int dim, float R, float *output) {
    // Map Poincare ball point back to Euclidean tangent space:
    // log_map(x) = artanh(||x||) x x/||x|| x R
    float norm = wubu_norm(input, dim);
    if (norm < 1e-8f) {
        memcpy(output, input, dim * sizeof(float));
        return;
    }
    if (norm >= 1.0f) norm = 0.99f; // clamp
    float factor = (float)(atanh((double)norm) / norm) * R;
    for (int i = 0; i < dim; i++) {
        output[i] = factor * input[i];
    }
}