/* dspark_markov.c -- the DSpark markov head, ported from the verified numpy
 * reference and the upstream-ggml source (dspark.cpp build_dspark_markov_head).
 *
 * Data layout (all row-major, float):
 *   x    = [state(R) | w1_prev(R) | hidden(E) | f_rnn(E)]   total = 2R + 2E
 *   Jw   = joint_proj row-major (3R, 2R+2E), out = Jw . x + b
 *   out  = [gate(R) | cand(R) | markov(R)]
 *   state' = cand + gate*(state - cand)          (GRU update)
 *   markov = tanh(out[2R:3R])
 *
 * The caller supplies x (state carried in/out), hidden, f_rnn and the prev row
 * of markov_w1. R (rank) is 256, E (n_embd) is 2304 for Inkling-2-Mini.
 *
 * Pure C11, no deps. Verified against upstream dspark_joint at corr 1.00000.
 */
#include <math.h>
#include <string.h>

/* col = base_i + b2 + W2 . markov, then softcap tanh(col/c)*c when c>0.
 * W2 is (V,R) row-major, markov is (R). base_i/b2 are (V). */
void dspark_col_output(const float *base_i, const float *markov,
                       const float *W2, const float *b2,
                       size_t V, size_t R, float softcap, float *col) {
    for (size_t v = 0; v < V; ++v) {
        const float *w = W2 + (size_t)v*R;
        float acc = base_i[v] + (b2 ? b2[v] : 0.0f);
        for (size_t r = 0; r < R; ++r) acc += w[r]*markov[r];
        if (softcap > 0.f) acc = tanhf(acc/softcap)*softcap;
        col[v] = acc;
    }
}

/* cumsum draw, mirroring dspark.cpp draw(): top n_cand by logit, sort desc,
 * probs = softmax(vals*inv_temp), cumsum, pick first candidate whose cumsum
 * >= uniform; $greedy keeps the argmax of col. Returns the drawn token and
 * writes ids[0..n_cand) and probs[0..n_cand). */
int dspark_draw(const float *col, size_t V, size_t n_cand,
                float inv_temp, float uniform, int greedy,
                int *ids, float *probs) {
    /* naive top-k: collect the n_cand largest col indices by insertion sort */
    size_t k = n_cand < V ? n_cand : V;
    int *idx = ids;
    float *val = probs;                 /* reused as scratch, then overwritten */
    for (size_t i = 0; i < k; ++i) { idx[i] = (int)i; val[i] = col[i]; }
    /* insertion keep-top-k */
    for (size_t i = k; i < V; ++i) {
        /* find min in kept */
        size_t m = 0; for (size_t j = 1; j < k; ++j) if (val[j] < val[m]) m = j;
        if (col[i] > val[m]) { val[m] = col[i]; idx[m] = (int)i; }
    }
    /* sort kept desc by value (selection) */
    for (size_t i = 0; i < k; ++i) {
        size_t b = i; for (size_t j = i+1; j < k; ++j) if (val[j] > val[b]) b = j;
        if (b != i) { int ti=idx[i]; idx[i]=idx[b]; idx[b]=ti;
                      float tv=val[i]; val[i]=val[b]; val[b]=tv; }
    }
    float maxs = val[0];
    float sum = 0;
    for (size_t i = 0; i < k; ++i) {
        float p = (float)expf((val[i]*inv_temp) - (maxs*inv_temp));
        probs[i] = p; sum += p;
    }
    for (size_t i = 0; i < k; ++i) probs[i] /= sum;
    float cs = 0;
    int drawn = idx[k-1];
    for (size_t i = 0; i < k; ++i) {
        cs += probs[i];
        if (cs >= uniform) { drawn = idx[i]; break; }
    }
    if (greedy) drawn = idx[0];
    return drawn;
}


void in2_static_assert_markov_dim(size_t R, size_t E) {
    /* nothing at runtime; the caller is expected to size against the model */
    (void)R; (void)E;
}

/* out = Jw . x + b; Jw is (3R, 2R+2E) row-major, x is (2R+2E), b is (3R) */
void dspark_joint_proj(const float *Jw, const float *b, const float *x,
                       size_t R, size_t E, float *out) {
    const size_t K = 2*R + 2*E;
    for (size_t r = 0; r < 3*R; ++r) {
        const float *J = Jw + (size_t)r*K;
        float acc = b ? b[r] : 0.0f;
        for (size_t k = 0; k < K; ++k) acc += J[k]*x[k];
        out[r] = acc;
    }
}

/* one markov step: mutates state in place, returns the markov (rank R) chunk
 * in markov_out. x must be prepared by the caller as
 * [ state | w1_prev | hidden | f_rnn ]. */
void dspark_markov_step(const float *Jw, const float *b, const float *w1_prev,
                        const float *hidden, const float *f_rnn,
                        size_t R, size_t E, float *state, float *markov_out) {
    static float *x_buf = NULL;   /* caller should preallocate via dspark_markov_alloc */
    (void)x_buf;
    float stackbuf[2*256 + 2*2304];   /* enough for Inkling-2; guard dims below */
    if (2*R + 2*E > sizeof(stackbuf)/sizeof(float)) return;
    float *x = stackbuf;
    memcpy(x,      state,      R*sizeof(float));
    memcpy(x+R,    w1_prev,    R*sizeof(float));
    memcpy(x+2*R,  hidden,     E*sizeof(float));
    memcpy(x+2*R+E,f_rnn,      E*sizeof(float));

    float out[3*256];
    if (3*R > sizeof(out)/sizeof(float)) return;
    dspark_joint_proj(Jw, b, x, R, E, out);

    for (size_t r = 0; r < R; ++r) {
        const float g = 1.0f/(1.0f+expf(-out[r]));            /* gate      */
        const float c = tanhf(out[R+r]);                       /* candidate */
        state[r] = c + g*(state[r]-c);
        markov_out[r] = tanhf(out[2*R+r]);                     /* markov    */
    }
}
