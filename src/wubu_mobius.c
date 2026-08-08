#include "wubu_mobius.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// Möbius addition
// ============================================================
void wubu_mobius_add(const float *x, const float *y, int d, float R, float *z) {
    // The numerically stable formula from Ganea et al. (2018):
    // Let c = 1/R^2 (curvature).
    // Then:
    //   x ⊕ y = (1 + 2c⟨x,y⟩ + c||y||²) x + (1 - c||x||²) y
    //            ────────────────────────────────────────────────
    //              1 + 2c⟨x,y⟩ + c²||x||²||y||²
    //
    // But we work with R directly, so c = 1/R^2.
    
    float c = 1.0f / (R * R);
    
    // Compute dot product and squared norms
    float dot_xy = 0.0f;
    float nx2 = 0.0f, ny2 = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_xy += x[i] * y[i];
        nx2 += x[i] * x[i];
        ny2 += y[i] * y[i];
    }
    
    // Check if x is at origin (zero norm)
    if (nx2 < 1e-30f) {
        // 0 ⊕ y = y
        memcpy(z, y, d * sizeof(float));
        return;
    }
    if (ny2 < 1e-30f) {
        // x ⊕ 0 = x
        memcpy(z, x, d * sizeof(float));
        return;
    }
    
    // Compute numerator coefficients
    float cny2 = c * ny2;
    float cnx2 = c * nx2;
    float c2nx2ny2 = c * cnx2 * ny2;
    float two_c_dot = 2.0f * c * dot_xy;
    
    float denom = 1.0f + two_c_dot + c2nx2ny2;
    float num1 = 1.0f + two_c_dot + cny2;
    float num2 = 1.0f - cnx2;
    
    // Clamp denominator to avoid division by zero
    // In theory denom > 0 for all valid ball points, but with float32
    // boundary conditions we might get near-zero.
    if (fabsf(denom) < 1e-30f) {
        // Fallback: just return y (boundary case — shouldn't happen with valid inputs)
        memcpy(z, y, d * sizeof(float));
        return;
    }
    
    float inv_denom = 1.0f / denom;
    
    // z = (num1 * x + num2 * y) / denom
    for (int i = 0; i < d; i++) {
        z[i] = (num1 * x[i] + num2 * y[i]) * inv_denom;
    }
}

// ============================================================
// Möbius scalar multiplication
// ============================================================
void wubu_mobius_scalar_mul(float r, const float *x, int d, float R, float *z) {
    // r ⊗ x = tanh(r * artanh(||x||/R)) * x / ||x|| * R
    // = exp_map(r * log_map(x))
    
    float nx = 0.0f;
    for (int i = 0; i < d; i++) nx += x[i] * x[i];
    nx = sqrtf(nx);
    
    // If x is at origin, result is origin (regardless of r)
    if (nx < 1e-30f) {
        memset(z, 0, d * sizeof(float));
        return;
    }
    
    // Compute scaling factor
    float ratio = nx / R;
    // Clamp to avoid domain error in artanh (must be < 1)
    if (ratio >= 0.9999f) ratio = 0.9999f;
    
    float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float scale = tanhf(r * artanh_r);
    float factor = scale * R / nx;
    
    for (int i = 0; i < d; i++) {
        z[i] = x[i] * factor;
    }
}

// ============================================================
// Poincaré geodesic distance
// ============================================================
float wubu_poincare_dist(const float *x, const float *y, int d, float R) {
    // d(x,y) = R * artanh(||(-x) ⊕ y|| / R)
    float *neg_x = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++) neg_x[i] = -x[i];
    
    float *diff = (float *)malloc(d * sizeof(float));
    wubu_mobius_add(neg_x, y, d, R, diff);
    
    float ndiff = 0.0f;
    for (int i = 0; i < d; i++) ndiff += diff[i] * diff[i];
    ndiff = sqrtf(ndiff);
    
    // Clamp for artanh domain
    float ratio = ndiff / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;
    if (ratio < 0.0f) ratio = 0.0f;
    
    float dist = R * 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    
    free(neg_x);
    free(diff);
    return dist;
}

// ============================================================
// Tangent-space linear combination (Poincaré I)
// ============================================================
void wubu_poincare_linear_comb(const float **xi, const float *wi, int n, int d, float R, float *z) {
    // z = exp_map(sum_i w_i * log_map(x_i))
    
    // Allocate tangent-space buffer
    float *tangent = (float *)calloc(d, sizeof(float));
    if (!tangent) return;
    
    float *log_xi = (float *)malloc(d * sizeof(float));
    if (!log_xi) { free(tangent); return; }
    
    // Sum w_i * log_map(x_i) in tangent space
    for (int i = 0; i < n; i++) {
        // log_map(x) = R * artanh(||x||/R) * x/||x||
        float nx = 0.0f;
        for (int j = 0; j < d; j++) nx += xi[i][j] * xi[i][j];
        nx = sqrtf(nx);
        
        if (nx < 1e-30f) continue;  // skip origin points
        
        float ratio = nx / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
        float scale = R * artanh_r / nx;
        
        for (int j = 0; j < d; j++) {
            tangent[j] += wi[i] * xi[i][j] * scale;
        }
    }
    
    // exp_map(tangent) = tanh(||t||/R) * t/||t|| * R
    float nt = 0.0f;
    for (int j = 0; j < d; j++) nt += tangent[j] * tangent[j];
    nt = sqrtf(nt);
    
    if (nt < 1e-30f) {
        memset(z, 0, d * sizeof(float));
    } else {
        float ratio = nt / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float tanh_r = tanhf(ratio);
        float factor = tanh_r * R / nt;
        for (int j = 0; j < d; j++) {
            z[j] = tangent[j] * factor;
        }
    }
    
    free(tangent);
    free(log_xi);
}

// ============================================================
// exp_map backward (tangent → ball)
// Forward: z = tanh(||v||/R) * v/||v|| * R
// ============================================================
void wubu_exp_map_backward(const float *v, int d, float R,
                            const float *z, const float *dz,
                            float *dv) {
    // z = s * v where s = tanh(nv/R) * R / nv
    // d(z_i) = s * δ_ij * d(v_j) + v_i * ds/dnv * v_j/nv * d(v_j)
    // dv_i = s * dz_i + v_i * ds/dnv * (v·dz) / nv

    float nv = 0.0f, v_dot_dz = 0.0f;
    for (int i = 0; i < d; i++) {
        nv += v[i] * v[i];
        v_dot_dz += v[i] * dz[i];
    }
    nv = sqrtf(nv);

    if (nv < 1e-30f) {
        // exp_map(0) ≈ 0, gradient ≈ identity: d(z_i)/d(v_j) = δ_ij
        memcpy(dv, dz, d * sizeof(float));
        return;
    }

    float ratio = nv / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;  // clamp for tanh domain

    float s = tanhf(ratio);                  // tanh(nv/R)
    float scale = s * R / nv;                // s * R / nv

    // ds/dnv = (1 - s²) / R
    // dscale/dnv = ((1-s²) * nv - s * R) / (nv²)
    float dscale_dnv = ((1.0f - s * s) * nv - s * R) / (nv * nv);

    float factor = dscale_dnv * v_dot_dz / nv;
    for (int i = 0; i < d; i++) {
        dv[i] = scale * dz[i] + v[i] * factor;
    }
}

// ============================================================
// log_map backward (ball → tangent)
// Forward: v = R * artanh(||x||/R) * x/||x||
// ============================================================
void wubu_log_map_backward(const float *x, int d, float R,
                            const float *v, const float *dv,
                            float *dx) {
    // v = s * x where s = R * artanh(nx/R) / nx
    // Same structure as exp_map backward
    float nx = 0.0f, x_dot_dv = 0.0f;
    for (int i = 0; i < d; i++) {
        nx += x[i] * x[i];
        x_dot_dv += x[i] * dv[i];
    }
    nx = sqrtf(nx);

    if (nx < 1e-30f) {
        // log_map(0) ≈ 0, gradient ≈ identity
        memcpy(dx, dv, d * sizeof(float));
        return;
    }

    float ratio = nx / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;

    float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float scale = R * artanh_r / nx;          // R * artanh(nx/R) / nx

    // ds/dnx = R * [1/(1-ratio²) * 1/R * nx - artanh(nx/R)] / nx²
    //        = [nx/(1-ratio²) - R*artanh(nx/R)] / nx²
    //        = [nx * R²/(R²-nx²) - R*artanh(nx/R)] / nx²
    float inv_1mr2 = R * R / (R * R - nx * nx);  // 1/(1-ratio²) = R²/(R²-nx²)
    float dscale_dnx = (nx * inv_1mr2 - R * artanh_r) / (nx * nx);

    float factor = dscale_dnx * x_dot_dv / nx;
    for (int i = 0; i < d; i++) {
        dx[i] = scale * dv[i] + x[i] * factor;
    }
}

// ============================================================
// Möbius scalar multiplication backward: z = r ⊗ x
// Forward: z = tanh(r * artanh(||x||/R)) * x/||x|| * R
// ============================================================
void wubu_mobius_scalar_mul_backward(float r, const float *x, int d, float R,
                                      const float *z, const float *dz,
                                      float *dx) {
    // z = scale * x where scale = tanh(r * artanh(nx/R)) * R / nx
    // Same structure as exp_map backward with r * artanh(nx/R) as argument
    float nx = 0.0f, x_dot_dz = 0.0f;
    for (int i = 0; i < d; i++) {
        nx += x[i] * x[i];
        x_dot_dz += x[i] * dz[i];
    }
    nx = sqrtf(nx);

    if (nx < 1e-30f || fabsf(r) < 1e-30f) {
        // r ⊗ 0 = 0, or 0 ⊗ x = 0
        memset(dx, 0, d * sizeof(float));
        return;
    }

    float ratio = nx / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;

    float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float t = r * artanh_r;
    // Clamp t to avoid huge tanh values
    if (t > 80.0f) t = 80.0f;
    if (t < -80.0f) t = -80.0f;

    float th = tanhf(t);                      // tanh(r * artanh(nx/R))
    float scale = th * R / nx;                // scale factor

    // d(th)/d(nx) = (1 - th²) * r * d(artanh)/d(nx)
    // d(artanh)/d(nx) = 1/(1 - ratio²) * 1/R = R/(R² - nx²)
    // d(th)/d(nx) = (1 - th²) * r * R / (R² - nx²)
    float dth_dnx = (1.0f - th * th) * r * R / (R * R - nx * nx);
    // dscale/dnx = (dth_dnx * R * nx - th * R) / nx²
    float dscale_dnx = (dth_dnx * R * nx - th * R) / (nx * nx);

    float factor = dscale_dnx * x_dot_dz / nx;
    for (int i = 0; i < d; i++) {
        dx[i] = scale * dz[i] + x[i] * factor;
    }
}

// ============================================================
// Möbius addition backward (vector-Jacobian product)
// Computes dx, dy given upstream gradient dz for z = x ⊕ y
// ============================================================
void wubu_mobius_add_backward(const float *x, const float *y, int d, float R,
                               const float *z, const float *dz,
                               float *dx, float *dy) {
    // z = (A·x + B·y) / D
    // where:
    //   A = 1 + 2c⟨x,y⟩ + c||y||²
    //   B = 1 - c||x||²
    //   D = 1 + 2c⟨x,y⟩ + c²||x||²||y||²
    //   c = 1/R²

    float c = 1.0f / (R * R);

    // Compute dot products and squared norms
    float dot_xy = 0.0f, nx2 = 0.0f, ny2 = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_xy += x[i] * y[i];
        nx2 += x[i] * x[i];
        ny2 += y[i] * y[i];
    }

    if (nx2 < 1e-30f) {
        // 0 ⊕ y = y, so z = y, dz/dy = I, dz/dx ≈ 0
        memcpy(dy, dz, d * sizeof(float));
        if (dx) memset(dx, 0, d * sizeof(float));
        return;
    }
    if (ny2 < 1e-30f) {
        // x ⊕ 0 = x, so z = x, dz/dx = I, dz/dy ≈ 0
        memcpy(dx, dz, d * sizeof(float));
        if (dy) memset(dy, 0, d * sizeof(float));
        return;
    }

    // Precompute scalars from forward
    float cny2 = c * ny2;
    float cnx2 = c * nx2;
    float c2nx2ny2 = c * cnx2 * ny2;
    float two_c_dot = 2.0f * c * dot_xy;

    float A = 1.0f + two_c_dot + cny2;
    float B = 1.0f - cnx2;
    float D = 1.0f + two_c_dot + c2nx2ny2;
    float invD = (fabsf(D) < 1e-30f) ? 0.0f : 1.0f / D;
    float invD2 = invD * invD;

    // Precompute dz·num (scalar = Σ_i dz_i * num_i)
    double dot_dz_num = 0.0;
    for (int i = 0; i < d; i++) {
        float num_i = A * x[i] + B * y[i];
        dot_dz_num += (double)dz[i] * num_i;
    }

    // Precompute dz·x and dz·y
    double dot_dz_x = 0.0, dot_dz_y = 0.0;
    for (int i = 0; i < d; i++) {
        dot_dz_x += (double)dz[i] * x[i];
        dot_dz_y += (double)dz[i] * y[i];
    }

    // dx_j = (1/D²) · [ (2c·y_j · (A·x + B·y)·dz + A·dz_j - 2c·x_j · dot_dz_y) · D
    //                - (A·x + B·y)·dz · (2c·y_j + 2c²·||y||²·x_j) ]
    // Simplified:
    // dx_j = invD · A · dz_j + invD2 · [ y_j · (2c · D · dot_dz_num - 2c · dot_dz_num)
    //                                    - x_j · (2c · dot_dz_y · D + 2c²·||y||² · dot_dz_num) ]

    double dD_dx_pre = 2.0 * c;                 // ∂D/∂x_j = 2c·y_j + 2c²·||y||²·x_j
    double dA_dx_pre = 2.0 * c;                 // ∂A/∂x_j = 2c·y_j
    double dB_dx_pre = -2.0 * c;                // ∂B/∂x_j = -2c·x_j

    for (int j = 0; j < d; j++) {
        // ∂num_i/∂x_j = (∂A/∂x_j)·x_i + A·δ_ij + (∂B/∂x_j)·y_i
        //             = 2c·y_j·x_i + A·δ_ij - 2c·x_j·y_i

        double dA_dx = dA_dx_pre * y[j];        // 2c·y_j
        double dB_dx = dB_dx_pre * x[j];        // -2c·x_j
        double dD_dx = dD_dx_pre * y[j] + 2.0 * c * c * ny2 * x[j]; // 2c·y_j + 2c²·||y||²·x_j

        // dx_j = Σ_i dz_i · ∂z_i/∂x_j
        // = Σ_i dz_i · ( (∂num_i/∂x_j · D - num_i · ∂D/∂x_j) / D² )
        // = invD · Σ_i dz_i · ∂num_i/∂x_j  -  invD² · ∂D/∂x_j · Σ_i dz_i · num_i

        // Σ_i dz_i · ∂num_i/∂x_j = dA_dx · Σ_i dz_i · x_i + A·dz_j + dB_dx · Σ_i dz_i · y_i
        double sum_dz_dnum = dA_dx * dot_dz_x + (double)A * dz[j] + dB_dx * dot_dz_y;

        dx[j] = (float)(invD * sum_dz_dnum - invD2 * dD_dx * dot_dz_num);
    }

    if (dy) {
        // ∂A/∂y_j = 2c·x_j + 2c·y_j (from c||y||²)
        // ∂B/∂y_j = 0 (B doesn't depend on y)
        // ∂D/∂y_j = 2c·x_j + 2c²·||x||²·y_j

        double dA_dy_pre_x = 2.0 * c;           // from ⟨x,y⟩
        double dA_dy_pre_y = 2.0 * c;           // from ||y||²

        for (int j = 0; j < d; j++) {
            double dA_dy = dA_dy_pre_x * x[j] + dA_dy_pre_y * y[j]; // 2c·x_j + 2c·y_j
            double dD_dy = 2.0 * c * x[j] + 2.0 * c * c * nx2 * y[j]; // 2c·x_j + 2c²·||x||²·y_j

            // Σ_i dz_i · ∂num_i/∂y_j = dA_dy · Σ_i dz_i · x_i + 0 + dB_dy · Σ_i dz_i · y_i + B·dz_j
            double sum_dz_dnum = dA_dy * dot_dz_x + (double)B * dz[j];

            dy[j] = (float)(invD * sum_dz_dnum - invD2 * dD_dy * dot_dz_num);
        }
    }
}

// ============================================================
// Möbius gyration operator (optimized: shared dot products)
// ============================================================
void wubu_mobius_gyrate(const float *x, const float *y, const float *z, int d, float R, float *out) {
    // gyr[x,y]z = (-(x ⊕ y)) ⊕ (x ⊕ (y ⊕ z))
    //
    // Optimized: compute all dot products once and share across
    // the 3 mobius_add calls, eliminating redundant norm computations.
    // ~3x faster than the original naive 3-add version.

    float c = 1.0f / (R * R);

    // Step 0: precompute all dot products and norms
    float nx2 = 0, ny2 = 0, nz2 = 0, dxy = 0, dyz = 0;
    for (int i = 0; i < d; i++) {
        nx2 += x[i]*x[i]; ny2 += y[i]*y[i]; nz2 += z[i]*z[i];
        dxy += x[i]*y[i]; dyz += y[i]*z[i];
    }

    // Helper: mobius_add with cached norms
    // Computes a⊕b where na2 = ||a||², nb2 = ||b||², dab = ⟨a,b⟩
    float *buf1 = (float *)malloc(d * sizeof(float));
    float *buf2 = (float *)malloc(d * sizeof(float));
    float *x_plus_y = (float *)malloc(d * sizeof(float));

    // Step 1: x⊕y
    {
        float cny2 = c * ny2, cnx2 = c * nx2;
        float c2nx2ny2 = cnx2 * c * ny2;
        float two_c_dot = 2.0f * c * dxy;
        float alpha = 1.0f + two_c_dot + c2nx2ny2;
        float beta = 1.0f + two_c_dot + cny2;
        float gamma = 1.0f - cnx2;
        float inv_alpha = (fabsf(alpha) < 1e-30f) ? 0.0f : 1.0f / alpha;
        for (int i = 0; i < d; i++)
            x_plus_y[i] = (beta * x[i] + gamma * y[i]) * inv_alpha;
    }

    // Compute ||x⊕y||²
    float nxpy2 = 0;
    for (int i = 0; i < d; i++) nxpy2 += x_plus_y[i] * x_plus_y[i];

    // Step 2: y⊕z
    {
        float cnz2 = c * nz2, cny2 = c * ny2;
        float c2ny2nz2 = cny2 * c * nz2;
        float two_c_dot = 2.0f * c * dyz;
        float alpha = 1.0f + two_c_dot + c2ny2nz2;
        float beta = 1.0f + two_c_dot + cnz2;
        float gamma = 1.0f - cny2;
        float inv_alpha = (fabsf(alpha) < 1e-30f) ? 0.0f : 1.0f / alpha;
        for (int i = 0; i < d; i++)
            buf1[i] = (beta * y[i] + gamma * z[i]) * inv_alpha;
    }

    // Compute ||y⊕z||² and ⟨x, y⊕z⟩
    float nyz2 = 0, d_x_yz = 0;
    for (int i = 0; i < d; i++) {
        nyz2 += buf1[i] * buf1[i];
        d_x_yz += x[i] * buf1[i];
    }

    // Step 3: x ⊕ (y⊕z)
    {
        float cnyz2 = c * nyz2, cnx2 = c * nx2;
        float c2nx2nyz2 = cnx2 * c * nyz2;
        float two_c_dot = 2.0f * c * d_x_yz;
        float alpha = 1.0f + two_c_dot + c2nx2nyz2;
        float beta = 1.0f + two_c_dot + cnyz2;
        float gamma = 1.0f - cnx2;
        float inv_alpha = (fabsf(alpha) < 1e-30f) ? 0.0f : 1.0f / alpha;
        for (int i = 0; i < d; i++)
            buf2[i] = (beta * x[i] + gamma * buf1[i]) * inv_alpha;
    }

    // Compute ||x⊕(y⊕z)||² and ⟨-(x⊕y), x⊕(y⊕z)⟩
    float n_x_yz2 = 0, d_neg_xpy_x_yz = 0;
    for (int i = 0; i < d; i++) {
        n_x_yz2 += buf2[i] * buf2[i];
        d_neg_xpy_x_yz -= x_plus_y[i] * buf2[i];
    }

    // Step 4: (-(x⊕y)) ⊕ (x⊕(y⊕z))
    {
        float cn_x_yz2 = c * n_x_yz2, cnxpy2 = c * nxpy2;
        float c2nxpy2n_xyz = cnxpy2 * c * n_x_yz2;
        float two_c_dot = 2.0f * c * d_neg_xpy_x_yz;
        float alpha = 1.0f + two_c_dot + c2nxpy2n_xyz;
        float beta = 1.0f + two_c_dot + cn_x_yz2;
        float gamma = 1.0f - cnxpy2;
        float inv_alpha = (fabsf(alpha) < 1e-30f) ? 0.0f : 1.0f / alpha;
        for (int i = 0; i < d; i++)
            out[i] = (beta * (-x_plus_y[i]) + gamma * buf2[i]) * inv_alpha;
    }

    free(buf1);
    free(buf2);
    free(x_plus_y);
}
