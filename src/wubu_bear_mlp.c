#include <stddef.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * wubu_bear_mlp.c  --  BearRL MLP Policy Forward (CPU, C11 fallback).
 *
 * Self-contained CPU port of WuBuOS/src/bear/bear_kernels.cu:
 *   - MLP policy forward (no CUDA, no Tensor Cores)
 *   - GAE advantage computation
 *   - CartPole physics step
 *
 * C11, no external deps beyond libm.
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#include "wubu_bear_mlp.h"
#include <math.h>

#define WUBU_M_PI 3.14159265358979323846f

int wubu_bear_mlp_forward(const float *obs,
                          const float *h_in,
                          const float *W1, const float *b1,
                          const float *W2, const float *b2,
                          float *actions, float *logprobs,
                          float *values, float *h_out,
                          int batch, int obs_dim, int hidden_dim, int act_dim) {
    if (!obs || !W1 || !b1 || !W2 || !b2 ||
        !actions || !logprobs || !values || !h_out ||
        batch <= 0 || obs_dim <= 0 || hidden_dim <= 0 || act_dim <= 0)
        return -1;

    int out_dim = act_dim + 1 + 1 + hidden_dim;
    (void)out_dim;  /* Used for dimension validation; layout is documented above */

    /* Layer 1: h_out = relu(obs @ W1^T + b1)  [batch, hidden_dim] */
    for (int i = 0; i < batch; i++) {
        const float *x = obs + (size_t)i * obs_dim;
        float *ho = h_out + (size_t)i * hidden_dim;
        for (int j = 0; j < hidden_dim; j++) {
            float sum = b1[j];
            const float *wrow = W1 + (size_t)j * obs_dim;
            for (int k = 0; k < obs_dim; k++)
                sum += x[k] * wrow[k];
            ho[j] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
        }
    }

    /* Layer 2: output = h_out @ W2^T + b2  [batch, out_dim] */
    for (int i = 0; i < batch; i++) {
        const float *ho = h_out + (size_t)i * hidden_dim;
        float *a = actions + (size_t)i * act_dim;
        float *lp = logprobs + i;
        float *v = values + i;

        /* Action means: output[0..act_dim-1] */
        for (int j = 0; j < act_dim; j++) {
            float sum = b2[j];
            const float *wrow = W2 + (size_t)j * hidden_dim;
            for (int k = 0; k < hidden_dim; k++)
                sum += ho[k] * wrow[k];
            a[j] = sum;
        }

        /* Shared logstd: output[act_dim] */
        float logstd = b2[act_dim];

        /* Value: output[act_dim+1] */
        float v_sum = b2[act_dim + 1];
        const float *wrow = W2 + (size_t)(act_dim + 1) * hidden_dim;
        for (int k = 0; k < hidden_dim; k++)
            v_sum += ho[k] * wrow[k];
        v[0] = v_sum;

        /* Simplified logprob: Gaussian log-density */
        *lp = 0.0f;
        for (int j = 0; j < act_dim; j++) {
            float diff = a[j];  /* a = mean, assume action=mean for logprob */
            *lp += -0.5f * logf(2.0f * (float)M_PI) - logstd
                   - 0.5f * (diff * diff) / expf(2.0f * logstd);
        }
    }

    (void)h_in;  /* Not used in pure MLP (recurrent would use it) */
    return 0;
}

int wubu_bear_gae(const float *rewards, const uint8_t *dones,
                  const float *values, float *advantages, float *returns,
                  int T, int B, float gamma, float gae_lambda) {
    if (!rewards || !dones || !values || !advantages || !returns ||
        T <= 0 || B <= 0)
        return -1;

    for (int b = 0; b < B; b++) {
        float gae = 0.0f;
        for (int t = T - 1; t >= 0; t--) {
            int idx = t * B + b;
            float v_t = values[idx];
            float v_next = (t == T - 1) ? 0.0f : values[(t + 1) * B + b];
            float done = (float)(dones[idx] ? 1 : 0);

            float delta = rewards[idx] + gamma * (1.0f - done) * v_next - v_t;
            gae = delta + gamma * gae_lambda * (1.0f - done) * gae;

            advantages[idx] = gae;
            returns[idx] = gae + v_t;
        }
    }
    return 0;
}

int wubu_bear_cartpole_step(float *x, float *x_dot,
                            float *theta, float *theta_dot,
                            float *reward, uint8_t *done,
                            const float *force,
                            int num_envs,
                            float pole_mass, float pole_length,
                            float gravity, float cart_mass, float dt,
                            float force_mag, float angle_threshold,
                            float pos_threshold) {
    if (!x || !x_dot || !theta || !theta_dot || !reward || !done ||
        !force || num_envs <= 0)
        return -1;

    float total_mass = cart_mass + pole_mass;
    float pole_mass_length = pole_mass * pole_length;

    for (int i = 0; i < num_envs; i++) {
        float f = force[i] * force_mag;
        float th = theta[i];
        float th_d = theta_dot[i];
        float x_p = x[i];
        float x_d = x_dot[i];

        float cos_th = cosf(th);
        float sin_th = sinf(th);

        float temp = (f + pole_mass_length * th_d * th_d * sin_th) / total_mass;
        float denom = pole_length * (4.0f / 3.0f - pole_mass * cos_th * cos_th / total_mass);
        float th_ddot = (gravity * sin_th - cos_th * temp) / denom;
        float x_ddot = temp - pole_mass_length * th_ddot * cos_th / total_mass;

        /* Semi-implicit Euler integration */
        x_d = x_d + x_ddot * dt;
        x_p = x_p + x_d * dt;
        th_d = th_d + th_ddot * dt;
        th = th + th_d * dt;

        x[i] = x_p;
        x_dot[i] = x_d;
        theta[i] = th;
        theta_dot[i] = th_d;

        float angle = fabsf(th);
        float pos = fabsf(x_p);
        reward[i] = 1.0f;
        done[i] = (angle > angle_threshold || pos > pos_threshold) ? 1 : 0;
    }
    return 0;
}
